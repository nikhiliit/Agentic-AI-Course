#!/usr/bin/env python3
"""
Lab Technician AI Assistant - CLI Interface

This script provides a command-line interface for interacting with multiple LLMs
(OpenAI GPT, Google Gemini, and Ollama models) as a lab technician assistant.
The AI loads lab report data from PDF files and answers questions about blood tests,
medical diagnostics, and lab results in a professional, compassionate manner.

Features:
- Multi-provider LLM support (OpenAI, Gemini, Ollama)
- PDF lab report loading and text extraction
- Lab technician persona with medical expertise
- Conversation history management
- Command-line interface with interactive commands
- Automatic model availability detection

Based on the lab3.ipynb pattern but adapted for medical lab technician use case.
"""

import os
import sys
import threading
import time
import itertools
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Dict, Any, List
import requests
from pypdf import PdfReader

# Global variables for model selection and system prompt
llm_manager = None
current_model = None
current_system_prompt = ""
lab_report_content = ""
spinner_active = False

def show_processing_spinner():
    """Show a processing spinner while LLM is thinking."""
    global spinner_active
    spinner_chars = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    spinner_active = True

    while spinner_active:
        sys.stdout.write(f'\r{next(spinner_chars)} Analyzing your question... ')
        sys.stdout.flush()
        time.sleep(0.1)

    # Clear the spinner line
    sys.stdout.write('\r' + ' ' * 30 + '\r')
    sys.stdout.flush()

def stop_spinner():
    """Stop the processing spinner."""
    global spinner_active
    spinner_active = False

def load_lab_report(pdf_path: str = "MADHU labreportnew.pdf") -> str:
    """Load and extract text from lab report PDF file."""
    try:
        reader = PdfReader(pdf_path)
        lab_report = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                lab_report += text
        print(f"✅ Lab report loaded from {pdf_path} ({len(lab_report)} characters)")
        return lab_report
    except FileNotFoundError:
        print(f"❌ Lab report file not found: {pdf_path}")
        return ""
    except Exception as e:
        print(f"❌ Error loading lab report: {e}")
        return ""

def create_lab_technician_system_prompt(lab_report: str) -> str:
    """Create system prompt for lab technician persona."""
    name = "Dr. LLM"  # You can change this to the actual lab technician name

    system_prompt = f"""You are acting as {name}, a skilled lab technician and medical professional.
You are answering questions on {name}'s lab reports and medical test results website,
particularly questions related to blood reports, lab test results, medical diagnostics, and health analysis.

Your responsibility is to represent {name} for interactions on the website as faithfully as possible,
providing accurate, professional, and helpful explanations of lab results and medical information.

You are given detailed lab report data from blood tests and medical examinations which you can use to answer questions.

Be professional, accurate, and compassionate, as if talking to a patient or healthcare provider who needs to understand their lab results.
Always explain medical terms in simple, understandable language while maintaining accuracy.
If you don't have specific information about a particular test or result, say so clearly.

## Lab Report Data:
{lab_report}

With this context, please assist users with understanding their lab results, explaining medical terminology, and providing general guidance about blood tests and medical diagnostics."""

    return system_prompt

def prompt_model_selection(available_models: List[str]) -> str:
    """Prompt user to select a model from available options."""
    print("\n🤖 Available LLM Models:")
    for i, model_name in enumerate(available_models, 1):
        print(f"  {i}. {model_name}")

    # Check if running in interactive mode
    import sys
    if not sys.stdin.isatty():
        print(f"⚠️  Non-interactive environment detected. Using default model: {available_models[0]}")
        return available_models[0]

    while True:
        try:
            choice = input(f"\nSelect model (1-{len(available_models)}) or enter model name: ").strip()

            # Try numeric selection first
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_models):
                    selected = available_models[idx]
                    print(f"✅ Selected: {selected}")
                    return selected

            # Try name matching
            for model_name in available_models:
                if choice.lower() == model_name.lower():
                    print(f"✅ Selected: {model_name}")
                    return model_name

            print(f"❌ Invalid choice. Please select 1-{len(available_models)} or enter a valid model name.")

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")

def initialize_llm_manager():
    """Initialize the LLM manager globally."""
    global llm_manager, current_model

    load_dotenv(override=True)

    class LLMManager:
        def __init__(self):
            self.openai_key = os.getenv('OPENAI_API_KEY')
            self.google_key = os.getenv('GOOGLE_API_KEY')
            self.ollama_available = self._check_ollama()
            self.available_models = self._initialize_clients()

        def _check_ollama(self) -> bool:
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                return response.status_code == 200
            except:
                return False

        def _initialize_clients(self) -> Dict[str, Any]:
            available = {}

            if self.openai_key:
                try:
                    client = OpenAI(api_key=self.openai_key)
                    available["openai"] = {
                        "client": client,
                        "model": "gpt-4o-mini",
                        "name": "OpenAI GPT-4o-mini"
                    }
                    print("✅ OpenAI client initialized")
                except Exception as e:
                    print(f"❌ Failed to initialize OpenAI: {e}")

            if self.google_key:
                try:
                    client = OpenAI(
                        api_key=self.google_key,
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                    )
                    available["google"] = {
                        "client": client,
                        "model": "gemini-2.5-flash",
                        "name": "Google Gemini 2.5 Flash"
                    }
                    print("✅ Google Gemini client initialized")
                except Exception as e:
                    print(f"❌ Failed to initialize Google Gemini: {e}")

            if self.ollama_available:
                try:
                    client = OpenAI(
                        base_url='http://localhost:11434/v1',
                        api_key='ollama'
                    )
                    available["ollama"] = {
                        "client": client,
                        "model": "qwen3:1.7b",
                        "name": "Ollama Qwen3 1.7B"
                    }
                    print("✅ Ollama client initialized")
                except Exception as e:
                    print(f"❌ Failed to initialize Ollama: {e}")

            if not available:
                print("❌ No LLM clients could be initialized.")
                sys.exit(1)

            return available

        def chat_with_model(self, model_key: str, messages: List[Dict[str, str]]) -> Optional[str]:
            if model_key not in self.available_models:
                return None

            model_info = self.available_models[model_key]
            client = model_info["client"]
            model = model_info["model"]

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages
                )

                message_obj = response.choices[0].message
                content = message_obj.content

                if not content and getattr(message_obj, 'reasoning', None):
                    content = message_obj.reasoning

                return content if content else None
            except Exception as e:
                print(f"❌ Error with {model_info['name']}: {e}")
                return None

        def get_available_model_names(self) -> List[str]:
            return [info["name"] for info in self.available_models.values()]

        def get_model_key_from_name(self, display_name: str) -> Optional[str]:
            for key, info in self.available_models.items():
                if info["name"] == display_name:
                    return key
            return None

    llm_manager = LLMManager()
    available_models = llm_manager.get_available_model_names()

    if available_models:
        # Prompt user to select model before launching interface
        current_model = prompt_model_selection(available_models)
    else:
        print("❌ No models available")
        sys.exit(1)

    # Load lab report and create system prompt
    global lab_report_content, current_system_prompt
    lab_report_content = load_lab_report()
    if lab_report_content:
        current_system_prompt = create_lab_technician_system_prompt(lab_report_content)
        print("✅ Lab technician system prompt created")
    else:
        current_system_prompt = "You are a helpful lab technician assistant. Please ask me questions about lab reports and medical tests."
        print("⚠️ Using default system prompt (no lab report loaded)")

    print("🤖 LLM Manager Initialized")
    print(f"Available models: {', '.join(llm_manager.available_models.keys())}")
    print(f"Selected model: {current_model}")

def update_system_prompt(system_prompt: str):
    """Update the current system prompt."""
    global current_system_prompt
    current_system_prompt = system_prompt
    return f"System prompt updated! ({len(system_prompt)} chars)"


def chat(message, history):
    """Chat function compatible with gr.ChatInterface - exactly like lab3.ipynb pattern."""
    global llm_manager, current_model, current_system_prompt

    if not message.strip():
        return ""

    if not llm_manager or not current_model:
        return "❌ No LLM available. Please check your setup."

    # Get model key
    model_key = llm_manager.get_model_key_from_name(current_model)
    if not model_key:
        return f"❌ Model '{current_model}' not found."

    # Prepare messages for API call
    messages = []
    if current_system_prompt.strip():
        messages.append({"role": "system", "content": current_system_prompt})

    # Convert history from gr.ChatInterface format to API format
    # gr.ChatInterface uses [[user_msg, bot_msg], [user_msg, bot_msg], ...]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    # Add current user message
    messages.append({"role": "user", "content": message})

    # Get response
    response = llm_manager.chat_with_model(model_key, messages)

    if response:
        return response
    else:
        return f"❌ Failed to get response from {current_model}. Please try again."

def cli_chat_interface():
    """Run the CLI chat interface."""
    global llm_manager, current_model, current_system_prompt

    if not llm_manager:
        initialize_llm_manager()

    print(f"\n🩺 Lab Technician AI Assistant - {current_model}")
    print("=" * 60)
    print("I'm Dr. LLM, your lab technician assistant!")
    print("Ask me questions about blood reports, lab test results, and medical diagnostics.")
    print()
    print("Commands:")
    print("  /help     - Show this help")
    print("  /system   - Set system prompt")
    print("  /clear    - Clear conversation history")
    print("  /quit     - Exit the chat")
    print("=" * 60)

    conversation_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith('/'):
                if user_input == '/quit':
                    print("👋 Goodbye!")
                    break
                elif user_input == '/help':
                    print("Commands:")
                    print("  /help     - Show this help")
                    print("  /system   - Set system prompt")
                    print("  /clear    - Clear conversation history")
                    print("  /quit     - Exit the chat")
                    continue
                elif user_input == '/clear':
                    conversation_history = []
                    print("🧹 Conversation history cleared.")
                    continue
                elif user_input == '/system':
                    new_prompt = input("Enter new system prompt: ").strip()
                    update_system_prompt(new_prompt)
                    print(f"✅ System prompt updated! ({len(new_prompt)} chars)")
                    continue
                else:
                    print("❌ Unknown command. Type /help for available commands.")
                    continue

            # Add user message to history
            conversation_history.append({"role": "user", "content": user_input})

            # Start processing spinner in a separate thread
            spinner_thread = threading.Thread(target=show_processing_spinner, daemon=True)
            spinner_thread.start()

            try:
                # Get response from LLM
                response = chat(user_input, conversation_history)
            finally:
                # Stop the spinner
                stop_spinner()
                # Wait a moment for spinner to clear
                time.sleep(0.1)

            if response:
                print(f"🩺 Dr. LLM: {response}")
                conversation_history.append({"role": "assistant", "content": response})
            else:
                print(f"\n❌ Failed to get response from {current_model}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    """Main entry point."""
    try:
        print("🩺 Starting Lab Technician AI Assistant...")
        print(f"Selected model: {current_model}")
        print("Available models:", ", ".join(llm_manager.get_available_model_names()) if llm_manager else "None")
        print()

        cli_chat_interface()

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
