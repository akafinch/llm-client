use anyhow::Result;
use eframe::egui;
use poll_promise::Promise;
use std::time::Duration;
use std::sync::mpsc::{self, sync_channel};
use tokio::runtime::Runtime;

use crate::endpoint_type::EndpointType;
use crate::llmclient::LLMClient;
use crate::sdclient::{SDClient, TextToImageRequest, SDModel, LoRA, Sampler};

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
    pub sd_client: SDClient,
    pub sd_prompt: String,
    pub sd_generating: bool,
    pub sd_progress: f32,
    pub sd_image_bytes: Option<Vec<u8>>,
    pub sd_image_texture: Option<egui::TextureHandle>,
    pub sd_pending_generation: Option<Promise<Result<Vec<u8>>>>,
    pub sd_error_message: Option<String>,
    pub sd_models: Vec<SDModel>,
    pub sd_selected_model: String,
    pub sd_loras: Vec<LoRA>,
    pub sd_selected_lora: Option<String>,
    pub sd_lora_weight: f32,
    pub sd_samplers: Vec<Sampler>,
    pub sd_selected_sampler: String,
    pub sd_schedulers: Vec<String>,
    pub sd_selected_scheduler: String,
    pub sd_schedulers_loading: bool,
    pub sd_steps: u32,
    pub sd_cfg_scale: f32,
    pub sd_width: u32,
    pub sd_height: u32,
    pub sd_negative_prompt: String,
    pub sd_models_loading: bool,
    pub sd_loras_loading: bool,
    pub sd_samplers_loading: bool,
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
            sd_client: SDClient::new("http://localhost:7860".to_string()),
            sd_prompt: String::new(),
            sd_generating: false,
            sd_progress: 0.0,
            sd_image_bytes: None,
            sd_image_texture: None,
            sd_pending_generation: None,
            sd_error_message: None,
            sd_models: Vec::new(),
            sd_selected_model: "".to_string(),
            sd_loras: Vec::new(),
            sd_selected_lora: None,
            sd_lora_weight: 0.7,
            sd_samplers: Vec::new(),
            sd_selected_sampler: "Euler a".to_string(),
            sd_schedulers: Vec::new(),
            sd_selected_scheduler: "Automatic".to_string(),
            sd_schedulers_loading: false,
            sd_steps: 20,
            sd_cfg_scale: 7.0,
            sd_width: 512,
            sd_height: 512,
            sd_negative_prompt: "blurry, low quality, deformed, distorted".to_string(),
            sd_models_loading: false,
            sd_loras_loading: false,
            sd_samplers_loading: false,
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

    pub fn load_sd_options(&mut self, ctx: &egui::Context) {
        // Loading models
        if self.sd_models.is_empty() && !self.sd_models_loading {
            self.sd_models_loading = true;
            
            let sd_client = self.sd_client.clone();
            let ctx_clone = ctx.clone();
            
            tokio::spawn(async move {
                match sd_client.get_available_models().await {
                    Ok(models) => {
                        ctx_clone.memory_mut(|mem| {
                            mem.data.insert_temp(egui::Id::new("sd_models"), models);
                        });
                    }
                    Err(e) => {
                        println!("Failed to fetch SD models: {}", e);
                        ctx_clone.memory_mut(|mem| {
                            mem.data.insert_temp(egui::Id::new("sd_models_error"), format!("{}", e));
                        });
                    }
                }
            });
        }
        
        // Loading LoRAs
        if self.sd_loras.is_empty() && !self.sd_loras_loading {
            self.sd_loras_loading = true;
            
            let sd_client = self.sd_client.clone();
            let ctx_clone = ctx.clone();
            
            tokio::spawn(async move {
                match sd_client.get_available_loras().await {
                    Ok(loras) => {
                        ctx_clone.memory_mut(|mem| {
                            mem.data.insert_temp(egui::Id::new("sd_loras"), loras);
                        });
                    }
                    Err(e) => {
                        println!("Failed to fetch LoRAs: {}", e);
                        ctx_clone.memory_mut(|mem| {
                            mem.data.insert_temp(egui::Id::new("sd_loras_error"), format!("{}", e));
                        });
                    }
                }
            });
        }
        
        // Loading samplers
        if self.sd_samplers.is_empty() && !self.sd_samplers_loading {
            self.sd_samplers_loading = true;
            
            let sd_client = self.sd_client.clone();
            let ctx_clone = ctx.clone();
            
            tokio::spawn(async move {
                match sd_client.get_available_samplers().await {
                    Ok(samplers) => {
                        ctx_clone.memory_mut(|mem| {
                            mem.data.insert_temp(egui::Id::new("sd_samplers"), samplers);
                        });
                    }
                    Err(e) => {
                        println!("Failed to fetch samplers: {}", e);
                        ctx_clone.memory_mut(|mem| {
                            mem.data.insert_temp(egui::Id::new("sd_samplers_error"), format!("{}", e));
                        });
                    }
                }
            });
        }
        
        // Load scheduler types based on the selected sampler
        if !self.sd_schedulers_loading && !self.sd_selected_sampler.is_empty() {
            self.sd_schedulers_loading = true;
            
            let sd_client = self.sd_client.clone();
            let ctx_clone = ctx.clone();
            let sampler_name = self.sd_selected_sampler.clone();
            
            tokio::spawn(async move {
                match sd_client.get_available_schedulers(&sampler_name).await {
                    Ok(schedulers) => {
                        ctx_clone.memory_mut(|mem| {
                            mem.data.insert_temp(egui::Id::new("sd_schedulers"), schedulers);
                        });
                    }
                    Err(e) => {
                        println!("Failed to fetch schedulers: {}", e);
                        ctx_clone.memory_mut(|mem| {
                            mem.data.insert_temp(egui::Id::new("sd_schedulers_error"), format!("{}", e));
                        });
                    }
                }
            });
        }
        
        // Check if data is ready and process it
        if let Some(models) = ctx.memory_mut(|mem| mem.data.remove_temp::<Vec<SDModel>>(egui::Id::new("sd_models"))) {
            self.sd_models = models;
            self.sd_models_loading = false;
            
            if !self.sd_models.is_empty() && self.sd_selected_model.is_empty() {
                self.sd_selected_model = self.sd_models[0].model_name.clone();
            }
        }
        
        if let Some(loras) = ctx.memory_mut(|mem| mem.data.remove_temp::<Vec<LoRA>>(egui::Id::new("sd_loras"))) {
            self.sd_loras = loras;
            self.sd_loras_loading = false;
        }
        
        if let Some(samplers) = ctx.memory_mut(|mem| mem.data.remove_temp::<Vec<Sampler>>(egui::Id::new("sd_samplers"))) {
            self.sd_samplers = samplers;
            self.sd_samplers_loading = false;
            
            // Select Euler a by default if available
            if self.sd_selected_sampler.is_empty() {
                self.sd_selected_sampler = self.sd_samplers
                    .iter()
                    .find(|s| s.name == "Euler a")
                    .map(|s| s.name.clone())
                    .unwrap_or_else(|| {
                        if !self.sd_samplers.is_empty() {
                            self.sd_samplers[0].name.clone()
                        } else {
                            "Euler a".to_string()
                        }
                    });
            }
            
            // When samplers are loaded or changed, we reload schedulers
            self.sd_schedulers_loading = false;
        }
        
        if let Some(schedulers) = ctx.memory_mut(|mem| mem.data.remove_temp::<Vec<String>>(egui::Id::new("sd_schedulers"))) {
            self.sd_schedulers = schedulers;
            self.sd_schedulers_loading = false;
            
            // Select Automatic by default if empty
            if self.sd_selected_scheduler.is_empty() && !self.sd_schedulers.is_empty() {
                self.sd_selected_scheduler = self.sd_schedulers[0].clone();
            }
        }
    }

    pub fn generate_sd_image(&mut self, ctx: &egui::Context) {
        self.sd_generating = true;
        self.sd_progress = 0.0;
        self.sd_error_message = None; // Clear any previous errors
        
        let mut prompt = self.sd_prompt.clone();
        let negative_prompt = self.sd_negative_prompt.clone();
        let steps = self.sd_steps;
        let cfg_scale = self.sd_cfg_scale;
        let width = self.sd_width;
        let height = self.sd_height;
        let sampler_name = self.sd_selected_sampler.clone();
        let scheduler = Some(self.sd_selected_scheduler.clone());
        let model_name = self.sd_selected_model.clone();
        
        // Add LoRA to prompt instead of using alwayson_scripts
        if let Some(lora_name) = &self.sd_selected_lora {
            // Add the LoRA to the prompt with the weight
            // Format: <lora:name:weight>
            prompt = format!("{} <lora:{}:{:.1}>", prompt, lora_name, self.sd_lora_weight);
        }
        
        let sd_client = self.sd_client.clone();
        let ctx_clone = ctx.clone();
        
        // Start the image generation in a separate thread
        self.sd_pending_generation = Some(Promise::spawn_thread("sd_generation", move || {
            let rt = Runtime::new().unwrap();
            rt.block_on(async move {
                // Change model if needed
                if !model_name.is_empty() {
                    if let Err(e) = sd_client.change_model(&model_name).await {
                        println!("Failed to change model: {}", e);
                        return Err(anyhow::anyhow!("Failed to change model: {}", e));
                    }
                }
                
                // Create the request (without alwayson_scripts)
                let request = TextToImageRequest {
                    prompt,
                    negative_prompt: Some(negative_prompt),
                    steps,
                    cfg_scale,
                    width,
                    height,
                    sampler_name,
                    scheduler,
                    seed: None, // Random seed
                    // Add the new hires.fix fields as None, they'll be filled with default values in generate_image
                    enable_hr: None,
                    hr_scale: None,
                    hr_upscaler: None,
                    hr_second_pass_steps: None,
                    denoising_strength: None,
                    alwayson_scripts: serde_json::json!({}), // Empty, since we're using prompt-based LoRA
                };
                
                // Log the actual request for debugging
                println!("Sending request: {}", serde_json::to_string_pretty(&request).unwrap_or_default());
                
                // Start the generation
                let image_data_result = sd_client.generate_image(request).await;
                
                // Check progress periodically while waiting
                let progress_client = sd_client.clone();
                let ctx_progress = ctx_clone.clone();
                
                tokio::spawn(async move {
                    while let Ok(progress) = progress_client.check_progress().await {
                        // Send progress update to UI
                        ctx_progress.memory_mut(|mem| {
                            mem.data.insert_temp(egui::Id::new("sd_progress"), progress);
                        });
                        
                        if progress >= 100.0 {
                            break;
                        }
                        
                        tokio::time::sleep(Duration::from_millis(500)).await;
                    }
                });
                
                image_data_result
            })
        }));
    }
    
    pub fn save_sd_image(&self) {
        if let Some(image_data) = &self.sd_image_bytes {
            // Use a file dialog to select where to save the file
            // For simplicity, we're just saving to a fixed location
            std::fs::write("generated_image.png", image_data)
                .expect("Failed to save image");
        }
    }
    
    pub fn process_sd_generation(&mut self, ctx: &egui::Context) {
        // Check for progress updates
        if let Some(progress) = ctx.memory_mut(|mem| mem.data.remove_temp::<f32>(egui::Id::new("sd_progress"))) {
            self.sd_progress = progress;
            ctx.request_repaint();
        }
        
        // Check if generation is complete
        if let Some(promise) = &self.sd_pending_generation {
            if let Some(result) = promise.ready() {
                self.sd_generating = false;
                
                match result {
                    Ok(image_data) => {
                        self.sd_image_bytes = Some(image_data.clone());
                        
                        // Create texture from image bytes
                        let image = image::load_from_memory(&image_data)
                            .expect("Failed to create image from data");
                        let size = [image.width() as _, image.height() as _];
                        let image_buffer = image.to_rgba8();
                        let pixels = image_buffer.as_flat_samples();
                        
                        self.sd_image_texture = Some(ctx.load_texture(
                            "generated-image",
                            egui::ColorImage::from_rgba_unmultiplied(
                                size,
                                pixels.as_slice(),
                            ),
                            egui::TextureOptions::default(),
                        ));
                    },
                    Err(e) => {
                        println!("Image generation failed: {}", e);
                        self.sd_error_message = Some(format!("Error: {}", e));
                    }
                }
                
                self.sd_pending_generation = None;
            }
        }
    }
} 