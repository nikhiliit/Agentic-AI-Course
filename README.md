# 🤖 Agentic AI Course

Welcome to the comprehensive **Agentic AI Course**! This course will guide you through every step of building intelligent AI agents, from foundational concepts to advanced implementations.

## 🎯 What You'll Learn

This course takes you on a complete journey through Agentic AI development:

### **Foundations (Weeks 1-2)**
- **Week 1**: Multi-LLM Interactions & API Mastery
  - Working with OpenAI, Anthropic, Google Gemini, and local models
  - Understanding different AI model capabilities and trade-offs
  - Building robust API integrations with error handling
- **Week 2**: Agent Design Patterns & Tool Integration
  - Creating agents that can use tools and APIs
  - Memory management and conversation persistence
  - Multi-agent collaboration systems

### **Advanced Agentic AI (Weeks 3-4)**
- **Week 3**: CrewAI & Multi-Agent Systems
  - Building complex agent teams with specialized roles
  - Task delegation and coordination
  - Quality assurance through agent evaluation
- **Week 4**: LangGraph & State Management
  - Advanced conversation flows and state machines
  - Building agents with complex decision trees
  - Integrating external data sources and APIs

### **Production-Ready Agents (Weeks 5-6)**
- **Week 5**: AutoGen & Distributed Agents
  - Scalable agent architectures
  - Distributed computing for AI agents
  - Real-time agent communication
- **Week 6**: MCP (Model Context Protocol) & Advanced Integrations
  - Building agents that integrate with external tools
  - Context management across sessions
  - Production deployment strategies

## 🚀 Featured Project: Interactive LLM Chat Interface

This repository includes a powerful **Interactive LLM Chat Interface** (`1_multiple_llm_interaction.py`) that demonstrates advanced Agentic AI concepts:

### **Key Features:**
- 🤖 **Multi-Provider Support**: Chat with OpenAI GPT, Google Gemini, and local Ollama models
- 🔗 **LLM Chaining**: Create conversational chains where each model's response becomes the next model's input
- 🛡️ **Smart Filtering**: Automatically handles content filtering and API limitations
- 📚 **Model Comparison**: Compare responses from different AI models to the same query
- 💬 **Interactive CLI**: User-friendly command-line interface with multiple options

### **How to Use:**
```bash
# Install dependencies
pip install openai python-dotenv requests

# Set up your API keys in .env file
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key

# Install and run Ollama (optional)
ollama pull qwen3:1.7b
ollama serve

# Run the interactive interface
python 1_multiple_llm_interaction.py
```

### **Available Commands:**
- `[number]` - Chat with specific model (1=OpenAI, 2=Google Gemini, 3=Ollama)
- `all` - Send message to all available models
- `compare` - Compare model responses side-by-side
- `chain` - Chain multiple models (response → input flow)
- `history` - View conversation history

## 📁 Course Structure

```
Agentic-AI-Course/
├── 1_foundations/          # Basic concepts and API integrations
├── 2_openai/              # Advanced OpenAI integrations
├── 3_crew/                # Multi-agent systems with CrewAI
├── 4_langgraph/           # State management and complex flows
├── 5_autogen/             # Distributed agent architectures
├── 6_mcp/                 # Model Context Protocol implementations
├── guides/                # Learning guides and tutorials
├── assets/                # Course images and resources
└── 1_multiple_llm_interaction.py  # Interactive LLM demo
```

## 🛠️ Prerequisites

- Python 3.8+
- Basic understanding of Python programming
- Familiarity with APIs and HTTP requests
- (Optional) Ollama for local AI model experimentation

## 🎓 Learning Outcomes

By the end of this course, you'll be able to:

- ✅ Build AI agents that can interact with multiple LLM providers
- ✅ Create conversational chains between different AI models
- ✅ Implement robust error handling for AI API integrations
- ✅ Design multi-agent systems with specialized roles
- ✅ Deploy production-ready AI agent applications
- ✅ Integrate AI agents with external tools and APIs

## 🤝 Contributing

This course encourages hands-on learning! We welcome:

- **Bug fixes** and **improvements** to existing code
- **New agent implementations** and **examples**
- **Documentation enhancements**
- **Community contributions** in the respective folders

## 📄 License

This course is provided for educational purposes. Please respect API usage policies and terms of service when building with these examples.

---

**Ready to build your first AI agent?** Start with the Interactive LLM Chat Interface and explore the course materials step by step! 🚀
