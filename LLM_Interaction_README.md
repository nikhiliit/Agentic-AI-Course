# Interactive LLM Chat Interface

This project provides an interactive command-line interface for chatting with multiple Large Language Models (LLMs) including OpenAI GPT, Google Gemini, and local Ollama models.

## Features

- **Multi-Provider Support**: Chat with OpenAI GPT, Google Gemini, and Ollama models
- **Automatic Detection**: Only enables models for which you have API keys configured
- **Interactive CLI**: Easy-to-use command-line interface with multiple options
- **Model Comparison**: Compare responses from all available models to the same question
- **AI Judge**: Use AI to evaluate and rank model responses
- **Chat History**: Keep track of your conversations

## Prerequisites

### API Keys Setup

Create a `.env` file in the project root with your API keys:

```bash
# OpenAI (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Google Gemini (optional)
GOOGLE_API_KEY=your_google_api_key_here

# Anthropic (optional - not used in this script)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Ollama Setup (for local models)

1. Install Ollama from https://ollama.com
2. Pull a model (recommended: qwen3:1.7b for good balance of speed/quality):
   ```bash
   ollama pull qwen3:1.7b
   ```
   Or for better performance (but larger):
   ```bash
   ollama pull qwen3:4b
   ```
3. Start Ollama server:
   ```bash
   ollama serve
   ```

#### Qwen3 Model Options

The script currently uses **qwen3:1.7b** as the default. Here are all available Qwen3 models from [Ollama Library](https://ollama.com/library/qwen3):

| Model | Size | Pull Command | Best For |
|-------|------|-------------|----------|
| `qwen3:0.6b` | 523MB | `ollama pull qwen3:0.6b` | Fastest, minimal quality |
| `qwen3:1.7b` | 1.4GB | `ollama pull qwen3:1.7b` | **Current default** - good balance |
| `qwen3:4b` | 2.5GB | `ollama pull qwen3:4b` | Better quality, reasonable speed |
| `qwen3:8b` | 5.2GB | `ollama pull qwen3:8b` | High quality, moderate speed |
| `qwen3:14b` | 9.3GB | `ollama pull qwen3:14b` | Very high quality |
| `qwen3:30b` | 19GB | `ollama pull qwen3:30b` | Excellent quality, slower |
| `qwen3:32b` | 20GB | `ollama pull qwen3:32b` | Top tier quality |
| `qwen3:235b` | 142GB | `ollama pull qwen3:235b` | Maximum quality (requires powerful hardware) |

#### Switching Between Qwen3 Models

To use a different Qwen3 model, simply change the model name in the script:

```python
# In multiple_llm_interaction.py, around line 92
available["ollama"] = {
    "client": self.ollama_client,
    "model": "qwen3:4b",  # Change this to any model from the table above
    "name": "Ollama Qwen3 4B"
}
```

**Important:** Make sure to pull the model first with `ollama pull <model_name>` before changing the script configuration.

#### About Qwen3 Models

Qwen3 is Alibaba's latest generation of large language models with significant improvements over previous versions:

- **Enhanced Reasoning**: Superior performance in mathematics, code generation, and logical reasoning
- **Better Alignment**: Improved creative writing, role-playing, and instruction following
- **Agent Capabilities**: Strong integration with external tools and complex agent-based tasks
- **Multilingual Support**: Excellent performance across 100+ languages
- **Competitive Performance**: Matches or exceeds models like DeepSeek-R1, o1, and Gemini-2.5-Pro

The smaller models (1.7B, 4B) offer a great balance of speed and capability for most interactive use cases.

### Google Gemini Setup

Make sure your Google API key has access to the Gemini API. You can get a key from:
https://makersuite.google.com/app/apikey

## Installation

The required dependencies are already included in the project's `requirements.txt`. If using uv:

```bash
uv sync
```

## Usage

### Interactive Mode

Run the main interactive script:

```bash
uv run python interactive_llm.py
```

This will start an interactive command-line interface where you can:

- Choose specific models to chat with
- Send messages to all available models at once
- Compare model responses to the same question
- Use AI to judge and rank responses
- View chat history

### Programmatic Usage

You can also use the LLM manager directly in your code:

```python
from interactive_llm import LLMManager

# Initialize manager (automatically detects available models)
llm = LLMManager()

# Chat with a specific model
response = llm.chat_with_model('openai', 'Hello, how are you?')
print(response)

# Get available models
print(llm.available_models.keys())
```

### Demo Script

Run the demo to see the system in action:

```bash
uv run python demo_llm.py
```

## Available Models

| Provider | Model | Requirements |
|----------|-------|--------------|
| OpenAI | GPT-4o-mini | `OPENAI_API_KEY` environment variable |
| Google | Gemini 2.5 Flash | `GOOGLE_API_KEY` environment variable |
| Ollama | Qwen3 1.7B | Ollama installed and running locally |

## Menu Options

When running the interactive interface, you'll see these options:

1. **[number]** - Chat with a specific model (e.g., `1` for OpenAI)
2. **all** - Send a message to all available models
3. **compare** - Compare responses from all models to a question
4. **chain** - Chain multiple models (each response becomes next input)
5. **history** - Show chat history
6. **clear** - Clear chat history
7. **exit** - Exit the program

## Commands During Chat

When chatting with a specific model:
- Type your message and press Enter
- Type `back` or `b` to return to main menu
- Type `exit`, `quit`, or `q` to exit

## Error Handling

The script gracefully handles:
- Missing API keys (models are skipped automatically)
- Network errors and API failures
- Invalid user input
- Keyboard interrupts (Ctrl+C)

## Model Comparison Feature

The `compare` command allows you to:
1. Ask the same question to all available models
2. Get responses from each model
3. Optionally have AI judge and rank the responses based on:
   - Accuracy and correctness
   - Clarity and coherence
   - Completeness of answer
   - Helpfulness and relevance

## Architecture

The system consists of:

- **LLMManager**: Core class that handles model initialization and API calls
- **InteractiveCLI**: Command-line interface for user interaction
- **Environment Detection**: Automatic detection of available API keys and services

## LLM Chaining Feature

The **chain** command allows you to create sequential interactions between multiple LLMs:

1. **Select Models**: Choose which models to include in your chain (e.g., "1 2 3")
2. **Initial Query**: Provide the starting question or prompt
3. **Sequential Processing**: Each model's response becomes the input for the next model
4. **Chain Summary**: View a complete summary of the entire interaction chain

**Example Chain Flow:**
```
User Query → OpenAI → Response → Google Gemini → Response → Ollama → Final Response
```

This enables advanced workflows like:
- **Iterative Refinement**: One model generates ideas, another refines them
- **Multi-Perspective Analysis**: Different models provide complementary viewpoints
- **Quality Enhancement**: Models can critique and improve each other's outputs

## Security Notes

- API keys are loaded from environment variables only
- Keys are validated but not displayed (only first few characters shown)
- No API keys are stored or transmitted except to their respective services

## Troubleshooting

### Common Issues

1. **"No module named 'openai'"**: Make sure dependencies are installed:
   ```bash
   uv sync
   ```

2. **"No LLM clients could be initialized"**: Check your `.env` file and API keys

3. **"Ollama client failed to initialize"**: Make sure Ollama is running:
   ```bash
   ollama serve
   ```

4. **API errors**: Check your API keys are valid and have sufficient credits

### Getting API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **Google Gemini**: https://makersuite.google.com/app/apikey

## Contributing

This script is based on the lab exercises from the course. Feel free to extend it with:
- Additional model providers
- New features like conversation memory
- Different evaluation criteria
- Export functionality for chat history
