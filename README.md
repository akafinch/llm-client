# LLM Chat Client

A Rust-based GUI client for interacting with local Large Language Models. Currently supports both LM Studio and Ollama endpoints through their OpenAI-compatible APIs.

## Features

- 🖥️ Modern, native GUI using egui
- 🔄 Real-time streaming responses
- 🔌 Support for multiple LLM backends:
  - LM Studio
  - Ollama
- ⚙️ Configurable settings:
  - API endpoint selection
  - Model selection
  - Custom API URLs
- 💬 Chat-style interface with message history
- 🎨 Clean, intuitive design

## Prerequisites

- Rust (latest stable version)
- One of the following LLM servers:
  - [LM Studio](https://lmstudio.ai/) running locally
  - [Ollama](https://ollama.ai/) with at least one model installed

## Building

```bash
# Clone the repository
git clone [your-repo-url]
cd llm-client

# Build and run
cargo run
```

## Usage

1. Start your LLM server (LM Studio or Ollama)
2. Launch the client
3. Click the hamburger menu (☰) to configure:
   - Select your endpoint type (LM Studio or Ollama)
   - Choose your model from the dropdown
   - Optionally customize the API URL
4. Type your message and press Enter or click Send
5. Watch as the LLM responds in real-time!

## Default Endpoints

- LM Studio: `http://localhost:1234/v1/chat/completions`
- Ollama: `http://localhost:11434/v1/chat/completions`

## Dependencies

- `eframe`: GUI framework
- `reqwest`: HTTP client
- `tokio`: Async runtime
- `serde`: Serialization/deserialization
- `poll-promise`: Async state management
- `futures-util`: Future utilities
- `anyhow`: Error handling

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
