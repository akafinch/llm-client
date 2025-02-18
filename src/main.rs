#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use anyhow::{Result, Context};
use eframe::egui;
use futures_util::StreamExt;
use poll_promise::Promise;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use std::time::Duration;
use std::sync::mpsc::{self, sync_channel, SyncSender};
use tokio::runtime::Runtime;

const DEFAULT_API_URL: &str = "http://localhost:1234/v1/chat/completions";
const OLLAMA_API_URL: &str = "http://localhost:11434/v1/chat/completions";

#[derive(Debug, Clone, PartialEq, Copy)]
enum EndpointType {
    LMStudio,
    Ollama,
}

impl EndpointType {
    fn default_url(&self) -> &'static str {
        match self {
            EndpointType::LMStudio => DEFAULT_API_URL,
            EndpointType::Ollama => OLLAMA_API_URL,
        }
    }
    
    fn default_port(&self) -> &'static str {
        match self {
            EndpointType::LMStudio => "1234",
            EndpointType::Ollama => "11434",
        }
    }

    fn default_endpoint(&self) -> &'static str {
        match self {
            EndpointType::LMStudio => "v1/chat/completions",
            EndpointType::Ollama => "v1/chat/completions",
        }
    }
    
    fn models_endpoint(&self, endpoint: &str) -> String {
        match self {
            EndpointType::LMStudio => {
                // For LM Studio, always use /v1/models
                "v1/models".to_string()
            }
            EndpointType::Ollama => {
                // For Ollama, use /api/tags but respect any custom base path
                if endpoint.is_empty() {
                    "api/tags".to_string()
                } else {
                    // Strip the chat completions part if present and add api/tags
                    let base = endpoint.trim_end_matches("v1/chat/completions");
                    format!("{}api/tags", base.trim_end_matches('/'))
                        .trim_start_matches('/')
                        .to_string()
                }
            }
        }
    }

    fn chat_endpoint(&self, endpoint: &str) -> String {
        match self {
            EndpointType::LMStudio => {
                // For LM Studio, always use /v1/chat/completions
                "v1/chat/completions".to_string()
            }
            EndpointType::Ollama => {
                // For Ollama, use /api/chat but respect any custom base path
                if endpoint.is_empty() {
                    "api/chat".to_string()
                } else {
                    // Strip the chat completions part if present and add api/chat
                    let base = endpoint.trim_end_matches("v1/chat/completions");
                    format!("{}api/chat", base.trim_end_matches('/'))
                        .trim_start_matches('/')
                        .to_string()
                }
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct ModelData {
    id: String,
    // Add other fields as needed
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelData>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct DeltaContent {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    delta: DeltaContent,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct ModelDetails {
    name: String,
}

#[derive(Debug, Deserialize)]
struct OllamaModelsResponse {
    models: Vec<ModelDetails>,
}

#[derive(Clone)]
struct LLMClient {
    client: Client,
    protocol: String,
    server: String,
    port: String,
    endpoint: String,
    endpoint_type: EndpointType,
}

impl LLMClient {
    fn new(protocol: String, server: String, port: String, endpoint: String, endpoint_type: EndpointType) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(5))  // 5 second timeout
            .build()
            .unwrap_or_else(|_| Client::new());
            
        Self {
            client,
            protocol,
            server,
            port,
            endpoint,
            endpoint_type,
        }
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let models_url = format!("{}://{}:{}/{}",
            self.protocol,
            self.server,
            self.port,
            self.endpoint_type.models_endpoint(&self.endpoint)
        ).trim_end_matches('/').to_string();
        
        println!("Fetching models from: {}", models_url);
        
        let response = self.client
            .get(&models_url)
            .send()
            .await
            .context(format!("Failed to fetch models from {}. Please check if the server is running and accessible", &models_url))?;
            
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Server returned error {}: {}",
                response.status(),
                response.text().await.unwrap_or_else(|_| "No error message".to_string())
            ));
        }
            
        match self.endpoint_type {
            EndpointType::LMStudio => {
                let models: ModelsResponse = response
                    .json()
                    .await
                    .context("Failed to parse models response")?;
                    
                Ok(models.data.into_iter().map(|m| m.id).collect())
            }
            EndpointType::Ollama => {
                // First print the raw response for debugging
                let text = response.text().await?;
                println!("Raw Ollama response: {}", text);
                
                // Parse the response from the text
                let models: OllamaModelsResponse = serde_json::from_str(&text)
                    .context("Failed to parse Ollama response")?;
                    
                Ok(models.models.into_iter().map(|m| m.name).collect())
            }
        }
    }

    async fn chat_stream(&self, chat_history: &[(String, String)], prompt: &str, model: &str, tx: SyncSender<String>) -> Result<()> {
        let chat_url = format!("{}://{}:{}/{}",
            self.protocol,
            self.server,
            self.port,
            self.endpoint_type.chat_endpoint(&self.endpoint)
        ).trim_end_matches('/').to_string();
        
        // Convert chat history to messages format
        let mut messages = Vec::new();
        // Add all messages except the last one (which is the current prompt)
        for (role, content) in chat_history.iter().take(chat_history.len().saturating_sub(1)) {
            messages.push(serde_json::json!({
                "role": role,
                "content": content
            }));
        }
        // Add current prompt
        messages.push(serde_json::json!({
            "role": "user",
            "content": prompt
        }));

        // Different request format for different endpoints
        let request_body = match self.endpoint_type {
            EndpointType::LMStudio => {
                let request = ChatRequest {
                    model: model.to_string(),
                    messages: messages.iter().map(|m| ChatMessage {
                        role: m["role"].as_str().unwrap().to_string(),
                        content: m["content"].as_str().unwrap().to_string(),
                    }).collect(),
                    temperature: 0.7,
                    stream: true,
                };
                serde_json::to_value(request).unwrap()
            }
            EndpointType::Ollama => {
                serde_json::json!({
                    "model": model,
                    "messages": messages,
                    "stream": true
                })
            }
        };

        let response = self.client
            .post(&chat_url)
            .json(&request_body)
            .timeout(Duration::from_secs(300))  // 5 minute timeout for the entire stream
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send request: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Request failed with status {}: {}", status, error_text));
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut done = false;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    let chunk_str = String::from_utf8_lossy(&chunk);
                    
                    match self.endpoint_type {
                        EndpointType::LMStudio => {
                            // LMStudio uses SSE format with "data: " prefix
                            for line in chunk_str.lines() {
                                let line = line.trim();
                                if !line.starts_with("data: ") {
                                    continue;
                                }
                                let line = &line["data: ".len()..];
                                if line == "[DONE]" {
                                    done = true;
                                    continue;
                                }

                                if let Ok(response) = serde_json::from_str::<ChatResponse>(line) {
                                    if let Some(choice) = response.choices.first() {
                                        if let Some(content) = &choice.delta.content {
                                            buffer.push_str(content);
                                            tx.send(buffer.clone()).ok();
                                        }
                                        
                                        if choice.finish_reason.is_some() {
                                            done = true;
                                        }
                                    }
                                }
                            }
                        }
                        EndpointType::Ollama => {
                            // Ollama returns each chunk as a complete JSON object
                            if let Ok(response) = serde_json::from_str::<serde_json::Value>(&chunk_str) {
                                // Get content from message.content
                                if let Some(message) = response.get("message") {
                                    if let Some(content) = message.get("content") {
                                        if let Some(text) = content.as_str() {
                                            // Skip the thinking tokens but preserve newlines
                                            if text != "<think>" && text != "</think>" {
                                                // If we get pure newlines, add just one
                                                if text.trim().is_empty() && text.contains('\n') {
                                                    buffer.push('\n');
                                                } else {
                                                    buffer.push_str(text);
                                                }
                                                tx.send(buffer.clone()).ok();
                                            }
                                        }
                                    }
                                }
                                
                                if response.get("done").and_then(|v| v.as_bool()).unwrap_or(false) {
                                    done = true;
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    println!("Error reading chunk: {:?}", e);
                    return Err(e).context("Failed to read chunk");
                }
            }
        }

        if done {
            Ok(())
        } else {
            Err(anyhow::anyhow!("Stream ended without completion"))
        }
    }
}

struct ChatApp {
    client: LLMClient,
    runtime: Runtime,
    input: String,
    chat_history: Vec<(String, String)>,
    pending_response: Option<Promise<Result<()>>>,
    response_receiver: Option<mpsc::Receiver<String>>,
    current_response: String,
    show_settings: bool,
    protocol: String,
    server: String,
    port: String,
    endpoint: String,
    endpoint_type: EndpointType,
    available_models: Vec<String>,
    selected_model: String,
    models_loading: bool,
    error_message: Option<String>,
}

impl ChatApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let endpoint_type = EndpointType::Ollama;
        let protocol = "http".to_string();
        let server = "localhost".to_string();
        let port = "11434".to_string();
        let endpoint = "v1/chat/completions".to_string();
        Self {
            client: LLMClient::new(protocol.clone(), server.clone(), port.clone(), endpoint.clone(), endpoint_type),
            runtime: Runtime::new().unwrap(),
            input: String::new(),
            chat_history: Vec::new(),
            pending_response: None,
            response_receiver: None,
            current_response: String::new(),
            show_settings: true,
            protocol,
            server,
            port,
            endpoint,
            endpoint_type,
            available_models: Vec::new(),
            selected_model: "local-model".to_string(),
            models_loading: false,
            error_message: None,
        }
    }

    fn refresh_models(&mut self, ctx: &egui::Context) {
        self.models_loading = true;
        self.error_message = None;  // Clear any previous errors
        
        let client = self.client.clone();
        let ctx = ctx.clone();
        
        tokio::spawn(async move {
            match client.list_models().await {
                Ok(models) => {
                    ctx.memory_mut(|mem| {
                        mem.data.insert_temp(egui::Id::new("available_models").into(), models);
                    });
                }
                Err(e) => {
                    let error_msg = format!("Failed to fetch models: {}", e);
                    println!("{}", error_msg);
                    ctx.memory_mut(|mem| {
                        mem.data.insert_temp(egui::Id::new("models_error").into(), error_msg);
                    });
                }
            }
        });
    }

    fn send_message(&mut self, _ctx: &egui::Context) {
        if self.input.is_empty() || self.pending_response.is_some() {
            return;
        }

        let prompt = std::mem::take(&mut self.input);
        self.chat_history.push(("user".to_string(), prompt.clone()));

        let client = self.client.clone();
        let model = self.selected_model.clone();
        let chat_history = self.chat_history.clone();
        
        // Create a channel with a large buffer for fast chunks
        let (tx, rx) = sync_channel(16384); // 16K buffer
        self.response_receiver = Some(rx);
        
        self.pending_response = Some(Promise::spawn_thread("llm_response".to_string(), move || {
            let rt = Runtime::new().unwrap();
            rt.block_on(async move {
                client.chat_stream(&chat_history, &prompt, &model, tx).await
            })
        }));
    }

    fn show_settings_window(&mut self, ctx: &egui::Context) {
        // Check for model list updates or errors
        if let Some(error) = ctx.memory_mut(|mem| mem.data.remove_temp::<String>(egui::Id::new("models_error"))) {
            self.error_message = Some(error);
            self.models_loading = false;
        }
        if let Some(models) = ctx.memory_mut(|mem| mem.data.remove_temp::<Vec<String>>(egui::Id::new("available_models"))) {
            self.available_models = models;
            self.models_loading = false;
            
            // Select the first model if none selected
            if self.selected_model == "local-model" && !self.available_models.is_empty() {
                self.selected_model = self.available_models[0].clone();
            }
        }

        let mut show_settings = self.show_settings;
        let endpoint_type = self.endpoint_type;
        let mut protocol = self.protocol.clone();  // Use the edit buffer, not the current URL
        let mut server = self.server.clone();  // Use the edit buffer, not the current URL
        let mut port = self.port.clone();  // Use the edit buffer, not the current URL
        let mut endpoint = self.endpoint.clone();  // Use the edit buffer, not the current URL
        let available_models = self.available_models.clone();
        let selected_model = self.selected_model.clone();
        let models_loading = self.models_loading;
        let mut url_changed = false;

        egui::Window::new("Settings")
            .open(&mut show_settings)
            .resizable(false)
            .default_width(400.0)
            .show(ctx, |ui| {
                ui.heading("API Configuration");
                ui.add_space(8.0);
                
                // Endpoint type selection
                ui.horizontal(|ui| {
                    ui.label("Endpoint Type:");
                    let mut new_endpoint = endpoint_type;  
                    if ui.radio_value(&mut new_endpoint, EndpointType::LMStudio, "OpenAI-Compatible").clicked() {
                        self.endpoint_type = new_endpoint;
                        self.port = new_endpoint.default_port().to_string();
                        port = self.port.clone(); // Update the UI-bound variable
                        self.endpoint = new_endpoint.default_endpoint().to_string();
                        endpoint = self.endpoint.clone(); // Update the UI-bound variable
                    }
                    if ui.radio_value(&mut new_endpoint, EndpointType::Ollama, "Ollama").clicked() {
                        self.endpoint_type = new_endpoint;
                        self.port = new_endpoint.default_port().to_string();
                        port = self.port.clone(); // Update the UI-bound variable
                        self.endpoint = new_endpoint.default_endpoint().to_string();
                        endpoint = self.endpoint.clone(); // Update the UI-bound variable
                    }
                });
                
                ui.add_space(8.0);
                
                // Model selection
                ui.horizontal(|ui| {
                    ui.label("Model:");
                    let mut new_model = selected_model.clone();
                    egui::ComboBox::from_id_source("model_select")
                        .selected_text(&new_model)
                        .show_ui(ui, |ui| {
                            for model in &available_models {
                                ui.selectable_value(&mut new_model, model.clone(), model);
                            }
                        });
                    if new_model != selected_model {
                        self.selected_model = new_model;
                    }
                        
                    if ui.button("⟳").on_hover_text("Refresh model list").clicked() {
                        self.refresh_models(ctx);
                    }
                });
                
                if models_loading {
                    ui.spinner();
                }
                
                ui.add_space(8.0);
                
                // API URL components in a grid layout
                egui::Grid::new("api_settings_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        // Protocol dropdown
                        ui.label("Protocol:");
                        egui::ComboBox::from_id_source("protocol_select")
                            .selected_text(&protocol)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut protocol, "http".to_string(), "http");
                                ui.selectable_value(&mut protocol, "https".to_string(), "https");
                            });
                        if protocol != self.protocol {
                            self.protocol = protocol.clone();
                        }
                        ui.end_row();

                        // Server
                        ui.label("Server:");
                        let response = ui.text_edit_singleline(&mut server);
                        if response.changed() {
                            self.server = server.clone();
                        }
                        ui.end_row();

                        // Port (with smaller width)
                        ui.label("Port:");
                        ui.add(egui::TextEdit::singleline(&mut port)
                            .desired_width(60.0))
                            .on_hover_text("Port number");
                        if port != self.port {
                            self.port = port.clone();
                        }
                        ui.end_row();

                        // Endpoint
                        ui.label("Endpoint:");
                        let response = ui.text_edit_singleline(&mut endpoint);
                        if response.changed() {
                            self.endpoint = endpoint.clone();
                        }
                        ui.end_row();
                    });
                
                ui.add_space(8.0);
                
                // Test Connection button
                if ui.button("Test Connection").clicked() {
                    println!("Testing connection to: {}://{}:{}/{}", protocol, server, port, endpoint);
                    match reqwest::Url::parse(&format!("{}://{}:{}/{}", protocol, server, port, endpoint)) {
                        Ok(_) => {
                            // Update the actual URL and try to connect
                            self.client = LLMClient::new(protocol.clone(), server.clone(), port.clone(), endpoint.clone(), self.endpoint_type);
                            self.refresh_models(ctx);
                        }
                        Err(e) => {
                            self.error_message = Some(format!("Invalid URL: {}", e));
                        }
                    }
                }
                
                ui.add_space(4.0);
                ui.label("Current: ").on_hover_text("The URL currently in use");
                ui.label(&format!("{}://{}:{}/{}", self.protocol, self.server, self.port, self.endpoint));
                
                // Display error message if present
                if let Some(error) = &self.error_message {
                    ui.colored_label(egui::Color32::RED, error);
                }
                
                ui.separator();
                
                if ui.button("Reset to Default").clicked() {
                    self.protocol = "http".to_string();
                    self.server = "localhost".to_string();
                    self.port = "11434".to_string();
                    self.endpoint = "v1/chat/completions".to_string();
                    self.client = LLMClient::new(self.protocol.clone(), self.server.clone(), self.port.clone(), self.endpoint.clone(), self.endpoint_type);
                    self.refresh_models(ctx);
                }
            });
            
        self.show_settings = show_settings;
        self.protocol = protocol;
        self.server = server;
        self.port = port;
        self.endpoint = endpoint;
    }
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request a repaint after a short delay (16ms = ~60 FPS)
        ctx.request_repaint_after(Duration::from_millis(16));

        // Check for new response chunks
        if let Some(rx) = &self.response_receiver {
            if let Ok(new_content) = rx.try_recv() {
                self.current_response = new_content;
                ctx.request_repaint();
            }
        }

        // Check if the response is complete
        if let Some(promise) = &self.pending_response {
            if let Some(result) = promise.ready() {
                match result {
                    Err(e) => {
                        // Only show error if we don't have content
                        if self.current_response.is_empty() {
                            self.chat_history.push(("error".to_string(), format!("Error: {}", e)));
                        } else {
                            // If we have content, treat as success despite error
                            self.chat_history.push(("assistant".to_string(), self.current_response.clone()));
                        }
                    }
                    Ok(()) => {
                        // Only move to history if we have content
                        if !self.current_response.is_empty() {
                            self.chat_history.push(("assistant".to_string(), self.current_response.clone()));
                        }
                    }
                }
                self.current_response.clear();
                self.pending_response = None;
                self.response_receiver = None;
                // Request repaint to show final state
                ctx.request_repaint();
            }
        }

        // Top menu bar
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("☰").clicked() {
                    self.show_settings = !self.show_settings;
                    if self.show_settings && self.available_models.is_empty() {
                        self.refresh_models(ctx);
                    }
                }
                ui.label("LLM Chat");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("New Chat").clicked() {
                        // Clear all chat state
                        self.chat_history.clear();
                        self.current_response.clear();
                        self.input.clear();
                        self.pending_response = None;
                        self.response_receiver = None;
                        self.error_message = None;
                    }
                });
            });
        });

        // Settings window
        if self.show_settings {
            self.show_settings_window(ctx);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let available_height = ui.available_height();
            let input_area_height = 100.0; // Fixed height for input area
            
            // Use vertical layout to separate chat history and input
            ui.vertical(|ui| {
                // Chat history area with calculated height
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .stick_to_bottom(true)
                    .max_height(available_height - input_area_height)
                    .show(ui, |ui| {
                        for (role, content) in &self.chat_history {
                            let is_user = role == "user";
                            let is_error = role == "error";
                            ui.horizontal(|ui| {
                                if is_user {
                                    ui.label(egui::RichText::new("You: ").strong());
                                } else if is_error {
                                    ui.label(egui::RichText::new("Error: ").strong().color(egui::Color32::RED));
                                } else {
                                    ui.label(egui::RichText::new("LLM: ").strong());
                                }
                            });
                            if is_error {
                                ui.label(egui::RichText::new(content).color(egui::Color32::RED));
                            } else {
                                ui.label(content);
                            }
                            ui.add_space(8.0);
                        }

                        // Show current response if any
                        if !self.current_response.is_empty() {
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("LLM: ").strong());
                            });
                            ui.label(&self.current_response);
                        }
                    });

                ui.add_space(8.0);

                // Input area with fixed height
                ui.group(|ui| {
                    ui.set_min_height(input_area_height);
                    
                    ui.vertical(|ui| {
                        // Text input
                        ui.add_sized(
                            [ui.available_width(), 70.0],
                            egui::TextEdit::multiline(&mut self.input)
                                .hint_text("Type your message here... (Press Enter to send, Shift+Enter for new line)")
                                .desired_rows(3),
                        );

                        // Send button
                        ui.horizontal(|ui| {
                            if ui.button("Send").clicked() || 
                               (ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift)) {
                                self.send_message(ctx);
                            }
                        });
                    });
                });
            });
        });
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "LLM Chat",
        options,
        Box::new(|cc| Box::new(ChatApp::new(cc))),
    )
    .map_err(|e| anyhow::anyhow!("Failed to run app: {}", e))
}
