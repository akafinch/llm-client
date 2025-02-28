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
    pub scheduler: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    // Hires.fix parameters (optional with skip_serializing_if)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_hr: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hr_scale: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hr_upscaler: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hr_second_pass_steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub denoising_strength: Option<f32>,
    #[serde(skip_serializing_if = "TextToImageRequest::is_empty_value")]
    pub alwayson_scripts: serde_json::Value,
}

impl TextToImageRequest {
    fn is_empty_value(value: &serde_json::Value) -> bool {
        value.as_object().map_or(false, |obj| obj.is_empty())
    }
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

#[derive(Debug, Deserialize, Clone)]
pub struct SDModel {
    pub title: String,
    pub model_name: String,
    pub hash: Option<String>,
    pub sha256: Option<String>,
    pub filename: Option<String>,
    pub config: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LoRA {
    pub name: String,
    pub alias: Option<String>,
    pub path: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Sampler {
    pub name: String,
    pub aliases: Option<Vec<String>>,
    pub options: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ScheduleType {
    pub name: String,
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
    
    pub async fn get_available_models(&self) -> Result<Vec<SDModel>> {
        let url = format!("{}/sdapi/v1/sd-models", self.base_url.trim_end_matches('/'));
        
        println!("Fetching available SD models from: {}", url);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch available SD models")?;
            
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to fetch models: {}", response.status()));
        }
        
        let models: Vec<SDModel> = response
            .json()
            .await
            .context("Failed to parse SD models response")?;
            
        Ok(models)
    }
    
    pub async fn get_available_loras(&self) -> Result<Vec<LoRA>> {
        let url = format!("{}/sdapi/v1/loras", self.base_url.trim_end_matches('/'));
        
        println!("Fetching available LoRAs from: {}", url);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch available LoRAs")?;
            
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to fetch LoRAs: {}", response.status()));
        }
        
        let loras: Vec<LoRA> = response
            .json()
            .await
            .context("Failed to parse LoRAs response")?;
            
        Ok(loras)
    }
    
    pub async fn get_available_samplers(&self) -> Result<Vec<Sampler>> {
        let url = format!("{}/sdapi/v1/samplers", self.base_url.trim_end_matches('/'));
        
        println!("Fetching available samplers from: {}", url);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch available samplers")?;
            
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to fetch samplers: {}", response.status()));
        }
        
        let samplers: Vec<Sampler> = response
            .json()
            .await
            .context("Failed to parse samplers response")?;
            
        Ok(samplers)
    }
    
    pub async fn get_available_schedulers(&self, sampler_name: &str) -> Result<Vec<String>> {
        // The Automatic1111 API doesn't directly expose a method to get schedule types
        // Based on the warning message, we know common ones are:
        // "Automatic", "Uniform", "Karras", "Exponential", "Polyexponential"
        
        // Hard-coded list of common schedulers - this could be improved with a more API-based approach
        // if the Automatic1111 API eventually exposes a method for this
        let default_schedulers = vec![
            "Automatic".to_string(),
            "Uniform".to_string(), 
            "Karras".to_string(), 
            "Exponential".to_string(), 
            "Polyexponential".to_string()
        ];

        // Optionally, we could try to parse these from sampler.options if available in the future
        // For now, return the default list
        Ok(default_schedulers)
    }
    
    pub async fn change_model(&self, model_name: &str) -> Result<()> {
        let url = format!("{}/sdapi/v1/options", self.base_url.trim_end_matches('/'));
        
        println!("Changing model to: {}", model_name);
        
        let request_body = serde_json::json!({
            "sd_model_checkpoint": model_name
        });
        
        let response = self.client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .context("Failed to change model")?;
            
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to change model: {}", response.status()));
        }
        
        Ok(())
    }
    
    pub async fn generate_image(&self, mut request: TextToImageRequest) -> Result<Vec<u8>> {
        let url = format!("{}/sdapi/v1/txt2img", self.base_url.trim_end_matches('/'));
        
        // Set default values for hires.fix
        request.enable_hr = Some(true);
        request.hr_scale = Some(2.0);
        request.hr_upscaler = Some("Latent".to_string());
        request.hr_second_pass_steps = Some(request.steps / 2);  // Half the original steps
        request.denoising_strength = Some(0.55);  // Good default value
        
        println!("Sending request to Stable Diffusion API: {}", url);
        
        // Print the request as JSON for debugging
        println!("Request payload: {}", serde_json::to_string_pretty(&request).unwrap_or_default());
        
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
                "Stable Diffusion API returned error {}. \nDetails: {}\n\nCheck that:\n1. Automatic1111 WebUI is running\n2. The --api flag is enabled\n3. The LoRA format is correct for your installation\n4. The URL is correct (default: http://localhost:7860)",
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