use anyhow::{Result, Context};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::SyncSender;
use std::time::Duration;
use base64::{Engine as _, engine::general_purpose};

#[derive(Debug, Serialize)]
pub struct TextToImageRequest {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    pub steps: u32,
    pub cfg_scale: f32,
    pub width: u32,
    pub height: u32,
    pub sampler_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct TextToImageResponse {
    pub images: Vec<String>, // Base64 encoded images
    pub parameters: serde_json::Value,
    pub info: String,
}

#[derive(Debug, Deserialize)]
pub struct ProgressResponse {
    pub progress: f32,        // 0-1 progress value
    pub eta_relative: f32,    // estimated time remaining in seconds
    pub state: serde_json::Value,
}

#[derive(Clone)]
pub struct SDClient {
    client: Client,
    pub base_url: String,
}

impl SDClient {
    pub fn new(base_url: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))  // 5 minute timeout
            .build()
            .unwrap_or_else(|_| Client::new());
            
        Self {
            client,
            base_url,
        }
    }
    
    pub async fn generate_image(&self, request: TextToImageRequest) -> Result<Vec<u8>> {
        let url = format!("{}/sdapi/v1/txt2img", self.base_url.trim_end_matches('/'));
        
        println!("Sending request to Stable Diffusion API: {}", url);
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context(format!("Failed to connect to Stable Diffusion API at {}. Make sure Automatic1111 is running and the API is enabled.", url))?;
            
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "No error details".to_string());
            
            return Err(anyhow::anyhow!(
                "Stable Diffusion API returned error {}. \nDetails: {}\n\nCheck that:\n1. Automatic1111 WebUI is running\n2. The --api flag is enabled\n3. The URL is correct (default: http://localhost:7860)",
                status, error_text
            ));
        }
        
        let response_data: TextToImageResponse = response
            .json()
            .await
            .context("Failed to parse image response")?;
            
        if response_data.images.is_empty() {
            return Err(anyhow::anyhow!("No images returned from the server"));
        }
        
        // Decode the base64 image
        let image_data = general_purpose::STANDARD
            .decode(&response_data.images[0])
            .context("Failed to decode base64 image")?;
            
        Ok(image_data)
    }
    
    pub async fn check_progress(&self) -> Result<f32> {
        let url = format!("{}/sdapi/v1/progress", self.base_url.trim_end_matches('/'));
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .context("Failed to check progress")?;
            
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Server returned error {}: {}",
                response.status(),
                response.text().await.unwrap_or_else(|_| "No error message".to_string())
            ));
        }
        
        let progress_data: ProgressResponse = response
            .json()
            .await
            .context("Failed to parse progress response")?;
            
        Ok(progress_data.progress * 100.0) // Convert to percentage
    }
} 