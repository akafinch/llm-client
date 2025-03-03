# AI/ML Server Client

A Rust-based GUI client for interacting with Large Language Models and Stable Diffusion servers. Currently supports both LM Studio and Ollama endpoints through their OpenAI-compatible APIs, and Stable Diffusion servers via the Automatic1111 WebUI API.

## Features

- 🖥️ Modern, native GUI using egui
- 🔄 Real-time streaming responses
- 🔌 Support for multiple LLM backends:
  - LM Studio
  - Ollama
- 🎨 Stable Diffusion integration:
  - Text-to-image generation
  - Model selection
  - LoRA support
  - Customizable parameters (steps, CFG scale, dimensions, etc.)
- ⚙️ Configurable settings:
  - API endpoint selection
  - Model selection
  - Custom API URLs
  - Sampler options
- 💬 Chat-style interface with message history
- 📊 Real-time generation progress tracking
- 🎨 Clean, intuitive design with tabbed interface

## Prerequisites

- Rust (latest stable version)
- One of the following LLM servers:
  - [LM Studio](https://lmstudio.ai/) running locally
  - [Ollama](https://ollama.ai/) with at least one model installed
- For image generation:
  - [Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) running with the ` --listen --xformers --api --enable-insecure-extension-access --server-name 0.0.0.0` flags enabled

## Building

```bash
# Clone the repository
git clone https://github.com/akafinch/llm-client.git
cd llm-client

# Build and run
cargo run
```

## Usage

### LLM Chat

1. Start your LLM server (LM Studio or Ollama)
2. Launch the client
3. Click the hamburger menu (☰) to configure:
   - Select your endpoint type (LM Studio or Ollama)
   - Choose your model from the dropdown
   - Optionally customize the API URL
4. Type your message and press Enter or click Send
5. Watch as the LLM responds in real-time!

### Stable Diffusion

1. Start the Automatic1111 WebUI with the `--api` flag
2. Click the "Stable Diffusion" tab
3. Configure SD settings from the hamburger menu (☰) → Stable Diffusion tab:
   - Select your model
   - Choose LoRA (optional)
   - Adjust generation parameters
4. Enter your prompt and click "Generate Image"
5. Watch the progress indicator as your image is created
6. Save generated images with the "Save Image" button

## Default Endpoints

- LM Studio: `http://localhost:1234/v1/chat/completions`
- Ollama: `http://localhost:11434/v1/chat/completions`
- Stable Diffusion (Automatic1111): `http://localhost:7860`

## Dependencies

- `eframe`: GUI framework
- `reqwest`: HTTP client
- `tokio`: Async runtime
- `serde`: Serialization/deserialization
- `poll-promise`: Async state management
- `futures-util`: Future utilities
- `anyhow`: Error handling
- `image`: Image processing
- `base64`: Encoding/decoding

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
