use anyhow::{Result, Context};
use eframe::egui;
use futures_util::StreamExt;
use poll_promise::Promise;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use std::sync::mpsc::{channel, Sender};
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
    
    fn models_endpoint(&self, base_url: &str) -> String {
        match self {
            EndpointType::LMStudio | EndpointType::Ollama => {
                base_url.replace("/chat/completions", "/models")
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

#[derive(Clone)]
struct LLMClient {
    client: Client,
    api_url: String,
    endpoint_type: EndpointType,
}

impl LLMClient {
    fn new(api_url: String, endpoint_type: EndpointType) -> Self {
        Self {
            client: Client::new(),
            api_url,
            endpoint_type,
        }
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let models_url = self.endpoint_type.models_endpoint(&self.api_url);
        let response = self.client
            .get(&models_url)
            .send()
            .await
            .context("Failed to fetch models")?;
            
        let models: ModelsResponse = response
            .json()
            .await
            .context("Failed to parse models response")?;
            
        Ok(models.data.into_iter().map(|m| m.id).collect())
    }

    async fn chat_stream(&self, prompt: &str, model: &str, tx: Sender<String>) -> Result<()> {
        let request = ChatRequest {
            model: model.to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: 0.7,
            stream: true,
        };

        let response = self.client
            .post(&self.api_url)
            .json(&request)
            .send()
            .await
            .context("Failed to send request")?;

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Failed to read chunk")?;
            let chunk_str = String::from_utf8_lossy(&chunk);
            
            for line in chunk_str.split("data: ") {
                let line = line.trim();
                if line.is_empty() || line == "[DONE]" {
                    continue;
                }

                if let Ok(response) = serde_json::from_str::<ChatResponse>(line) {
                    if let Some(choice) = response.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            buffer.push_str(content);
                            tx.send(buffer.clone()).ok();
                        }
                        
                        if choice.finish_reason.is_some() {
                            return Ok(());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

struct ChatApp {
    client: LLMClient,
    runtime: Runtime,
    input: String,
    chat_history: Vec<(String, String)>,
    pending_response: Option<Promise<Result<()>>>,
    response_receiver: Option<std::sync::mpsc::Receiver<String>>,
    current_response: String,
    show_settings: bool,
    api_url: String,
    api_url_edit: String,
    endpoint_type: EndpointType,
    available_models: Vec<String>,
    selected_model: String,
    models_loading: bool,
}

impl ChatApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let endpoint_type = EndpointType::LMStudio;
        let api_url = endpoint_type.default_url().to_string();
        Self {
            client: LLMClient::new(api_url.clone(), endpoint_type),
            runtime: Runtime::new().unwrap(),
            input: String::new(),
            chat_history: Vec::new(),
            pending_response: None,
            response_receiver: None,
            current_response: String::new(),
            show_settings: false,
            api_url,
            api_url_edit: endpoint_type.default_url().to_string(),
            endpoint_type,
            available_models: Vec::new(),
            selected_model: "local-model".to_string(), // Default model
            models_loading: false,
        }
    }

    fn refresh_models(&mut self, ctx: &egui::Context) {
        if self.models_loading {
            return;
        }
        
        self.models_loading = true;
        let client = self.client.clone();
        
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let rt = Runtime::new().unwrap();
            let models = rt.block_on(client.list_models());
            
            ctx.request_repaint();
            if let Ok(models) = models {
                ctx.memory_mut(|mem| {
                    mem.data.insert_temp("available_models".to_owned().into(), models);
                });
            }
        });
    }

    fn send_message(&mut self, _ctx: &egui::Context) {
        if self.input.trim().is_empty() || self.pending_response.is_some() {
            return;
        }

        let prompt = std::mem::take(&mut self.input);
        self.chat_history.push(("user".to_string(), prompt.clone()));
        self.current_response.clear();
        
        let (tx, rx) = channel();
        self.response_receiver = Some(rx);
        
        let client = self.client.clone();
        let model = self.selected_model.clone();
        
        self.pending_response = Some(Promise::spawn_thread(
            "llm_response".to_string(),
            move || {
                tokio::runtime::Runtime::new()
                    .unwrap()
                    .block_on(async move {
                        client.chat_stream(&prompt, &model, tx).await
                    })
            },
        ));
    }

    fn show_settings_window(&mut self, ctx: &egui::Context) {
        // Check for model list updates
        if let Some(models) = ctx.memory_mut(|mem| mem.data.remove_temp::<Vec<String>>("available_models".to_owned().into())) {
            self.available_models = models;
            self.models_loading = false;
            
            // Select the first model if none selected
            if self.selected_model == "local-model" && !self.available_models.is_empty() {
                self.selected_model = self.available_models[0].clone();
            }
        }

        let mut show_settings = self.show_settings;
        let endpoint_type = self.endpoint_type;
        let api_url_edit = self.api_url_edit.clone();
        let available_models = self.available_models.clone();
        let selected_model = self.selected_model.clone();
        let models_loading = self.models_loading;

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
                    if ui.radio_value(&mut new_endpoint, EndpointType::LMStudio, "LM Studio").clicked() {
                        self.endpoint_type = new_endpoint;
                        self.api_url = self.endpoint_type.default_url().to_string();
                        self.api_url_edit = self.api_url.clone();
                        self.client = LLMClient::new(self.api_url.clone(), self.endpoint_type);
                        self.refresh_models(ctx);
                    }
                    if ui.radio_value(&mut new_endpoint, EndpointType::Ollama, "Ollama").clicked() {
                        self.endpoint_type = new_endpoint;
                        self.api_url = self.endpoint_type.default_url().to_string();
                        self.api_url_edit = self.api_url.clone();
                        self.client = LLMClient::new(self.api_url.clone(), self.endpoint_type);
                        self.refresh_models(ctx);
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
                
                ui.horizontal(|ui| {
                    ui.label("API URL:");
                    let mut new_url = api_url_edit.clone();
                    if ui.text_edit_singleline(&mut new_url).lost_focus() {
                        if reqwest::Url::parse(&new_url).is_ok() {
                            self.api_url = new_url.clone();
                            self.api_url_edit = new_url;
                            self.client = LLMClient::new(self.api_url.clone(), self.endpoint_type);
                            self.refresh_models(ctx);
                        }
                    }
                });
                
                ui.add_space(4.0);
                ui.label("Current: ").on_hover_text("The URL currently in use");
                ui.label(&self.api_url);
                
                ui.separator();
                
                if ui.button("Reset to Default").clicked() {
                    self.api_url = self.endpoint_type.default_url().to_string();
                    self.api_url_edit = self.api_url.clone();
                    self.client = LLMClient::new(self.api_url.clone(), self.endpoint_type);
                    self.refresh_models(ctx);
                }
            });
            
        self.show_settings = show_settings;
    }
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request a repaint after a short delay (16ms = ~60 FPS)
        ctx.request_repaint_after(std::time::Duration::from_millis(16));

        // Check for new response chunks
        if let Some(rx) = &self.response_receiver {
            if let Ok(new_content) = rx.try_recv() {
                self.current_response = new_content;
            }
        }

        // Check if the response is complete
        if let Some(promise) = &self.pending_response {
            if let Some(result) = promise.ready() {
                if let Err(e) = result {
                    self.chat_history.push(("error".to_string(), format!("Error: {}", e)));
                } else {
                    // Add the final response to chat history before clearing
                    self.chat_history.push(("assistant".to_string(), self.current_response.clone()));
                }
                self.current_response.clear();
                self.pending_response = None;
                self.response_receiver = None;
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
