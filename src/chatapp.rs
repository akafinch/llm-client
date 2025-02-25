use anyhow::Result;
use eframe::egui;
use poll_promise::Promise;
use std::time::Duration;
use std::sync::mpsc::{self, sync_channel, SyncSender};
use tokio::runtime::Runtime;

use crate::endpoint_type::EndpointType;
use crate::llmclient::LLMClient;

const DEFAULT_API_URL: &str = "http://localhost:1234/v1/chat/completions";
const OLLAMA_API_URL: &str = "http://localhost:11434/v1/chat/completions";

pub struct ChatApp {
    pub client: LLMClient,
    pub runtime: Runtime,
    pub input: String,
    pub chat_history: Vec<(String, String)>,
    pub pending_response: Option<Promise<Result<()>>>,
    pub response_receiver: Option<mpsc::Receiver<String>>,
    pub current_response: String,
    pub show_settings: bool,
    pub protocol: String,
    pub server: String,
    pub port: String,
    pub endpoint: String,
    pub endpoint_type: EndpointType,
    pub available_models: Vec<String>,
    pub selected_model: String,
    pub models_loading: bool,
    pub error_message: Option<String>,
    pub active_tab: usize,
    pub active_settings_tab: usize,
}

impl ChatApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
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
            active_tab: 0,
            active_settings_tab: 0,
        }
    }

    pub fn refresh_models(&mut self, ctx: &egui::Context) {
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

    pub fn send_message(&mut self, _ctx: &egui::Context) {
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

    pub fn reset_to_defaults(&mut self) {
        self.protocol = "http".to_string();
        self.server = "localhost".to_string();
        self.port = "11434".to_string();
        self.endpoint = "v1/chat/completions".to_string();
        self.client = LLMClient::new(
            self.protocol.clone(),
            self.server.clone(),
            self.port.clone(),
            self.endpoint.clone(),
            self.endpoint_type
        );
    }

    pub fn update_endpoint_type(&mut self, new_endpoint_type: EndpointType) {
        self.endpoint_type = new_endpoint_type;
        self.port = new_endpoint_type.default_port().to_string();
        self.endpoint = new_endpoint_type.default_endpoint().to_string();
        self.selected_model = "local-model".to_string();
        self.available_models.clear();
    }

    pub fn clear_chat(&mut self) {
        self.chat_history.clear();
        self.current_response.clear();
        self.input.clear();
        self.pending_response = None;
        self.response_receiver = None;
        self.error_message = None;
    }

    pub fn update_client_url(&mut self) {
        self.client = LLMClient::new(
            self.protocol.clone(),
            self.server.clone(),
            self.port.clone(),
            self.endpoint.clone(),
            self.endpoint_type
        );
    }

    pub fn process_response_chunks(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.response_receiver {
            if let Ok(new_content) = rx.try_recv() {
                self.current_response.push_str(&new_content);
                ctx.request_repaint();
            }
        }

        if let Some(promise) = &self.pending_response {
            if let Some(result) = promise.ready() {
                match result {
                    Err(e) => {
                        if self.current_response.is_empty() {
                            self.chat_history.push(("error".to_string(), format!("Error: {}", e)));
                        } else {
                            self.chat_history.push(("assistant".to_string(), self.current_response.clone()));
                        }
                    }
                    Ok(()) => {
                        if !self.current_response.is_empty() {
                            self.chat_history.push(("assistant".to_string(), self.current_response.clone()));
                        }
                    }
                }
                self.current_response.clear();
                self.pending_response = None;
                self.response_receiver = None;
                ctx.request_repaint();
            }
        }
    }
} 