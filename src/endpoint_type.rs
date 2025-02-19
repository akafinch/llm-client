#[derive(Debug, Clone, PartialEq, Copy)]
pub enum EndpointType {
    LMStudio,
    Ollama,
}

impl EndpointType {
    pub fn default_url(&self) -> &'static str {
        match self {
            EndpointType::LMStudio => "http://localhost:1234/v1/chat/completions",
            EndpointType::Ollama => "http://localhost:11434/v1/chat/completions",
        }
    }
    
    pub fn default_port(&self) -> &'static str {
        match self {
            EndpointType::LMStudio => "1234",
            EndpointType::Ollama => "11434",
        }
    }

    pub fn default_endpoint(&self) -> &'static str {
        match self {
            EndpointType::LMStudio => "v1/chat/completions",
            EndpointType::Ollama => "v1/chat/completions",
        }
    }
    
    pub fn models_endpoint(&self, endpoint: &str) -> String {
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

    pub fn chat_endpoint(&self, endpoint: &str) -> String {
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
