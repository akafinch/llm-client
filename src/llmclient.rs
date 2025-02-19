use anyhow::{Result, Context};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::SyncSender;
use std::time::Duration;
use futures_util::StreamExt;
use crate::endpoint_type::EndpointType;

#[derive(Debug, Deserialize)]
pub struct ModelData {
    pub id: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelsResponse {
    pub data: Vec<ModelData>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: f32,
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct DeltaContent {
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub delta: DeltaContent,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
pub struct ModelDetails {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct OllamaModelsResponse {
    pub models: Vec<ModelDetails>,
}

#[derive(Clone)]
pub struct LLMClient {
    client: Client,
    protocol: String,
    server: String,
    port: String,
    endpoint: String,
    endpoint_type: EndpointType,
}

impl LLMClient {
    pub fn new(protocol: String, server: String, port: String, endpoint: String, endpoint_type: EndpointType) -> Self {
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

    pub async fn list_models(&self) -> Result<Vec<String>> {
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

    pub async fn chat_stream(&self, chat_history: &[(String, String)], prompt: &str, model: &str, tx: SyncSender<String>) -> Result<()> {
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
        
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| anyhow::anyhow!("Error reading stream: {}", e))?;
            let text = String::from_utf8_lossy(&chunk);
            
            match self.endpoint_type {
                EndpointType::LMStudio => {
                    // Split the text by lines and process each line
                    for line in text.lines() {
                        if line.is_empty() || line == "data: [DONE]" {
                            continue;
                        }
                        
                        if !line.starts_with("data: ") {
                            continue;
                        }
                        
                        let json_str = &line["data: ".len()..];
                        
                        match serde_json::from_str::<ChatResponse>(json_str) {
                            Ok(response) => {
                                if let Some(choice) = response.choices.first() {
                                    if let Some(content) = &choice.delta.content {
                                        buffer.push_str(content);
                                        
                                        // Try to send the content through the channel
                                        if tx.send(content.clone()).is_err() {
                                            // If sending fails, the receiver has been dropped
                                            return Ok(());
                                        }
                                    }
                                    
                                    if choice.finish_reason.is_some() {
                                        return Ok(());
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("Failed to parse response: {}. Raw JSON: {}", e, json_str);
                            }
                        }
                    }
                }
                EndpointType::Ollama => {
                    // Ollama returns each chunk as a complete JSON object
                    if let Ok(response) = serde_json::from_str::<serde_json::Value>(&text) {
                        // Get content from message.content
                        if let Some(message) = response.get("message") {
                            if let Some(content) = message.get("content") {
                                if let Some(text) = content.as_str() {
                                    // Skip the thinking tokens but preserve newlines
                                    if text != "<think>" && text != "</think>" {
                                        // If we get pure newlines, add just one
                                        if text.trim().is_empty() && text.contains('\n') {
                                            buffer.push('\n');
                                            if tx.send("\n".to_string()).is_err() {
                                                return Ok(());
                                            }
                                        } else {
                                            buffer.push_str(text);
                                            if tx.send(text.to_string()).is_err() {
                                                return Ok(());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        if response.get("done").and_then(|v| v.as_bool()).unwrap_or(false) {
                            return Ok(());
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
}
