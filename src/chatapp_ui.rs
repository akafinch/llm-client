use eframe::egui;
use reqwest;
use std::time::Duration;

use crate::chatapp::ChatApp;

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request a repaint after a short delay (16ms = ~60 FPS)
        ctx.request_repaint_after(Duration::from_millis(16));

        // Process any incoming response chunks
        self.process_response_chunks(ctx);
        
        // Process SD generation progress
        self.process_sd_generation(ctx);

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
                        self.clear_chat();
                    }
                });
            });
        });

        // Add tab bar below the top menu
        egui::TopBottomPanel::top("tab_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.active_tab, 0, "Chat");
                ui.selectable_value(&mut self.active_tab, 1, "Stable Diffusion");
                ui.selectable_value(&mut self.active_tab, 2, "Future Tab");
                // Add more tabs as needed
            });
        });

        // Settings window
        if self.show_settings {
            self.show_settings_window(ctx);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.active_tab {
                0 => self.render_chat_tab(ui, ctx),
                1 => self.render_stable_diffusion_tab(ui, ctx),
                2 => self.render_future_tab(ui),
                _ => self.render_chat_tab(ui, ctx), // Default to chat tab
            }
        });
    }
}

impl ChatApp {
    fn render_chat_tab(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
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
                    self.render_chat_history(ui);
                    self.render_current_response(ui);
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
    }

    fn render_stable_diffusion_tab(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let available_height = ui.available_height();
        
        // Divide the space: 70% for image area, 30% for prompt input
        let image_area_height = available_height * 0.7;
        let input_area_height = available_height * 0.3;
        
        // DEBUG
        // println!("Available height: {}, Image area: {}, Input area: {}", 
        //          available_height, image_area_height, input_area_height);
        
        ui.vertical(|ui| {
            // Image display area with fixed height
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), image_area_height),
                egui::Layout::centered_and_justified(egui::Direction::TopDown),
                |ui| {
                    // Add a visible frame around the image area
                    egui::Frame::dark_canvas(ui.style())
                        .stroke(egui::Stroke::new(1.0, egui::Color32::WHITE))
                        .show(ui, |ui| {
                            if let Some(texture) = &self.sd_image_texture {
                                // Display the generated image with proper scaling
                                let size = texture.size_vec2();
                                let max_size = ui.available_size();
                                let ratio = (max_size.x / size.x).min(max_size.y / size.y);
                                let display_size = egui::vec2(size.x * ratio, size.y * ratio);
                                
                                // Using the texture handle directly
                                let image = egui::Image::from_texture(texture)
                                    .fit_to_exact_size(display_size);
                                ui.add(image);
                            } else if self.sd_generating {
                                // Show progress
                                ui.spinner();
                                ui.label(format!("Generating image... {}%", self.sd_progress));
                            } else if let Some(error) = &self.sd_error_message {
                                ui.vertical_centered(|ui| {
                                    ui.colored_label(egui::Color32::RED, "Generation failed!");
                                    ui.add_space(10.0);
                                    ui.label(error);
                                });
                            } else {
                                // Initial state
                                ui.heading("Enter a prompt below to generate an image");
                            }
                        });
                }
            );
            
            // Add a visible separator
            ui.separator();
            
            // Prompt input area with fixed height
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), input_area_height),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| {
                    // Add a visible frame around the input area
                    egui::Frame::none()
                        .fill(egui::Color32::from_rgb(40, 40, 40))
                        .stroke(egui::Stroke::new(1.0, egui::Color32::WHITE))
                        .show(ui, |ui| {
                            ui.heading("Create an image with Stable Diffusion");
                            
                            ui.add_space(10.0);
                            ui.label("Enter your prompt:");
                            
                            // Force the text edit to be a reasonable size
                            let prompt_edit = egui::TextEdit::multiline(&mut self.sd_prompt)
                                .desired_rows(3)
                                .hint_text("A beautiful landscape with mountains and lakes...");
                            
                            ui.add_sized(
                                [ui.available_width(), ui.available_height() * 0.5],
                                prompt_edit
                            );
                            
                            ui.horizontal(|ui| {
                                if ui.button("Generate Image").clicked() && !self.sd_prompt.is_empty() && !self.sd_generating {
                                    self.generate_sd_image(ctx);
                                }
                                
                                if let Some(_) = &self.sd_image_bytes {
                                    if ui.button("Save Image").clicked() {
                                        self.save_sd_image();
                                    }
                                }
                            });
                        });
                }
            );

            ui.add_space(5.0);
            ui.label("⚠️ Note: Make sure Automatic1111 WebUI is running with the --api flag enabled");
            ui.label("Default URL: http://localhost:7860");
        });
    }

    fn render_future_tab(&mut self, ui: &mut egui::Ui) {
        ui.centered_and_justified(|ui| {
            ui.heading("Future Feature Coming Soon!");
        });
    }

    fn render_chat_history(&self, ui: &mut egui::Ui) {
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
                self.render_message_content(ui, content);
            }
            ui.add_space(8.0);
        }
    }

    fn render_current_response(&self, ui: &mut egui::Ui) {
        if !self.current_response.is_empty() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("LLM: ").strong());
            });
            self.render_message_content(ui, &self.current_response);
        }
    }

    fn render_message_content(&self, ui: &mut egui::Ui, content: &str) {
        let parts: Vec<&str> = content.split("<think>").collect();
        for (i, part) in parts.iter().enumerate() {
            if i == 0 && !part.is_empty() {
                ui.label(*part);
            } else if !part.is_empty() {
                let think_parts: Vec<&str> = part.split("</think>").collect();
                if think_parts.len() > 0 {
                    if !think_parts[0].is_empty() {
                        egui::Frame::none()
                            .fill(egui::Color32::from_rgb(47, 45, 56))
                            .inner_margin(egui::style::Margin::same(8.0))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("🤔 Thinking...")
                                        .color(egui::Color32::from_rgb(167, 139, 250))
                                        .strong());
                                });
                                ui.label(
                                    egui::RichText::new(think_parts[0])
                                        .color(egui::Color32::LIGHT_GRAY)
                                );
                            });
                    }
                    if think_parts.len() > 1 && !think_parts[1].is_empty() {
                        ui.label(think_parts[1]);
                    }
                }
            }
        }
    }

    pub fn show_settings_window(&mut self, ctx: &egui::Context) {
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
        egui::Window::new("Settings")
            .open(&mut show_settings)
            .resizable(false)
            .default_width(400.0)
            .show(ctx, |ui| {
                // Add tab bar at the top of the settings window
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.active_settings_tab, 0, "API Configuration");
                    ui.selectable_value(&mut self.active_settings_tab, 1, "Advanced Settings");
                });
                
                ui.separator();
                ui.add_space(8.0);
                
                // Display the active settings tab content
                match self.active_settings_tab {
                    0 => self.render_api_settings_tab(ui, ctx),
                    1 => self.render_advanced_settings_tab(ui),
                    _ => self.render_api_settings_tab(ui, ctx), // Default to API settings
                }
            });
            
        self.show_settings = show_settings;
    }

    fn render_api_settings_tab(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("API Configuration");
        ui.add_space(8.0);
        
        // Endpoint type selection
        ui.horizontal(|ui| {
            ui.label("Endpoint Type:");
            let mut new_endpoint = self.endpoint_type;  
            if ui.radio_value(&mut new_endpoint, crate::endpoint_type::EndpointType::LMStudio, "OpenAI-Compatible").clicked() {
                self.update_endpoint_type(new_endpoint);
            }
            if ui.radio_value(&mut new_endpoint, crate::endpoint_type::EndpointType::Ollama, "Ollama").clicked() {
                self.update_endpoint_type(new_endpoint);
            }
        });
        
        ui.add_space(8.0);
        
        // Model selection
        ui.horizontal(|ui| {
            ui.label("Model:");
            let mut new_model = self.selected_model.clone();
            egui::ComboBox::from_id_source("model_select")
                .selected_text(&new_model)
                .show_ui(ui, |ui| {
                    for model in &self.available_models {
                        ui.selectable_value(&mut new_model, model.clone(), model);
                    }
                });
            if new_model != self.selected_model {
                self.selected_model = new_model;
            }
                
            if ui.button("⟳").on_hover_text("Refresh model list").clicked() {
                self.refresh_models(ctx);
            }
        });
        
        if self.models_loading {
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
                    .selected_text(&self.protocol)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.protocol, "http".to_string(), "http");
                        ui.selectable_value(&mut self.protocol, "https".to_string(), "https");
                    });
                ui.end_row();

                // Server
                ui.label("Server:");
                ui.text_edit_singleline(&mut self.server);
                ui.end_row();

                // Port
                ui.label("Port:");
                ui.add(egui::TextEdit::singleline(&mut self.port)
                    .desired_width(60.0))
                    .on_hover_text("Port number");
                ui.end_row();

                // Endpoint
                ui.label("Endpoint:");
                ui.text_edit_singleline(&mut self.endpoint);
                ui.end_row();
            });
        
        ui.add_space(8.0);
        
        // Test Connection button
        if ui.button("Test Connection").clicked() {
            println!("Testing connection to: {}://{}:{}/{}", self.protocol, self.server, self.port, self.endpoint);
            match reqwest::Url::parse(&format!("{}://{}:{}/{}", self.protocol, self.server, self.port, self.endpoint)) {
                Ok(_) => {
                    self.selected_model = "local-model".to_string();
                    self.available_models.clear();
                    self.update_client_url();
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
            self.reset_to_defaults();
            self.refresh_models(ctx);
        }
    }
    
    fn render_advanced_settings_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Advanced Settings");
        ui.add_space(8.0);
        
        // This is a placeholder - add your advanced settings here
        ui.label("These settings will be implemented in a future update.");
        
        // Example settings that could be added here:
        ui.group(|ui| {
            ui.label("Model Parameters");
            ui.add_space(4.0);
            
            let mut placeholder_temp = 0.7;
            ui.horizontal(|ui| {
                ui.label("Temperature:");
                ui.add(egui::Slider::new(&mut placeholder_temp, 0.0..=2.0).text(""));
            });
            
            let mut placeholder_tokens = 2048;
            ui.horizontal(|ui| {
                ui.label("Max Tokens:");
                ui.add(egui::Slider::new(&mut placeholder_tokens, 256..=4096).text(""));
            });
            
            let mut placeholder_presence = 0.0;
            ui.horizontal(|ui| {
                ui.label("Presence Penalty:");
                ui.add(egui::Slider::new(&mut placeholder_presence, -2.0..=2.0).text(""));
            });
        });
        
        ui.add_space(8.0);
        
        ui.group(|ui| {
            ui.label("Chat History");
            ui.add_space(4.0);
            
            if ui.button("Export Chat History").clicked() {
                // Placeholder for export functionality
            }
            
            if ui.button("Import Chat History").clicked() {
                // Placeholder for import functionality
            }
        });

        ui.group(|ui| {
            ui.label("Stable Diffusion Settings");
            ui.add_space(4.0);
            
            ui.horizontal(|ui| {
                ui.label("API URL:");
                let mut api_url = self.sd_client.base_url.clone();
                if ui.text_edit_singleline(&mut api_url).changed() {
                    self.sd_client = crate::sdclient::SDClient::new(api_url);
                }
            });
        });
    }
} 