
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
from pydantic import BaseModel

# Create a Pydantic model for the Evaluation
class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str

# Global variables for model selection and system prompt
llm_manager = None
chat_model = None  # Model for lab technician responses
evaluator_model = None  # Model for evaluating responses
current_system_prompt = ""
lab_report_content = ""
spinner_active = False
evaluation_enabled = True  # Enable response evaluation by default

# Evaluator system prompt for medical context
evaluator_system_prompt = """You are an evaluator that decides whether a response to a medical question is acceptable.
You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality.
The Agent is playing the role of Dr. LLM, a skilled lab technician and medical professional.
The Agent has been instructed to be professional, accurate, and compassionate, as if talking to a patient or healthcare provider who needs to understand their lab results.
The Agent should explain medical terms in simple, understandable language while maintaining accuracy.
The Agent has been provided with detailed lab report data from blood tests and medical examinations.

With this context, please evaluate the latest response, checking for:
- Medical accuracy and appropriateness
- Professional and compassionate tone
- Clear explanation of medical terms
- Relevance to the user's question about lab results or medical diagnostics
- Whether the response addresses the user's concerns adequately

Reply with whether the response is acceptable and your detailed feedback."""

def evaluate_response(reply: str, message: str, history: List[Dict[str, str]]) -> Evaluation:
    """Evaluate whether a response from the LLM is acceptable quality."""
    global llm_manager, evaluator_model, evaluator_system_prompt

    if not llm_manager:
        # Fallback evaluation if no LLM manager available
        return Evaluation(is_acceptable=True, feedback="No evaluator available - response accepted by default")

    if not evaluator_model:
        return Evaluation(is_acceptable=True, feedback="No evaluator model selected - response accepted by default")

    # Get model key for evaluator
    model_key = llm_manager.get_model_key_from_name(evaluator_model)
    if not model_key:
        return Evaluation(is_acceptable=True, feedback="Evaluator model not found - response accepted by default")

    # Format conversation history for evaluation
    conversation_text = ""
    for item in history:
        if item["role"] == "user":
            conversation_text += f"User: {item['content']}\n"
        elif item["role"] == "assistant":
            conversation_text += f"Agent: {item['content']}\n"

    conversation_text += f"User: {message}\n"
    conversation_text += f"Agent: {reply}\n"

    evaluator_user_prompt = f"""Please evaluate the following conversation:

{conversation_text}

Is the Agent's latest response (the last "Agent:" message) acceptable quality for a medical lab technician assistant?"""

    messages = [
        {"role": "system", "content": evaluator_system_prompt},
        {"role": "user", "content": evaluator_user_prompt}
    ]

    try:
        # Try to use structured output if available (OpenAI style)
        if model_key in ["openai", "google"]:
            model_info = llm_manager.available_models[model_key]
            client = model_info["client"]
            model_name = model_info["model"]

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            if content:
                # Parse JSON response
                import json
                parsed = json.loads(content)
                return Evaluation(
                    is_acceptable=parsed.get("is_acceptable", True),
                    feedback=parsed.get("feedback", "Evaluation completed")
                )
        else:
            # For Ollama or other models, get text response and parse manually
            response = llm_manager.chat_with_model(model_key, messages)
            if response:
                # Simple parsing - look for keywords
                lower_response = response.lower()
                is_acceptable = "acceptable" in lower_response and "not acceptable" not in lower_response
                return Evaluation(is_acceptable=is_acceptable, feedback=response)
            else:
                return Evaluation(is_acceptable=True, feedback="Evaluation failed - response accepted by default")

    except Exception as e:
        print(f"⚠️ Evaluation error: {e}")
        return Evaluation(is_acceptable=True, feedback=f"Evaluation failed: {e} - response accepted by default")

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

def prompt_model_selection(available_models: List[str], purpose: str = "model") -> str:
    """Prompt user to select a model from available options."""
    print(f"\n🤖 Available LLM Models for {purpose}:")
    for i, model_name in enumerate(available_models, 1):
        print(f"  {i}. {model_name}")

    # Check if running in interactive mode
    import sys
    if not sys.stdin.isatty():
        print(f"⚠️  Non-interactive environment detected. Using default model: {available_models[0]}")
        return available_models[0]

    while True:
        try:
            choice = input(f"\nSelect {purpose} (1-{len(available_models)}) or enter model name: ").strip()

            # Try numeric selection first
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_models):
                    selected = available_models[idx]
                    print(f"✅ Selected {purpose}: {selected}")
                    return selected

            # Try name matching
            for model_name in available_models:
                if choice.lower() == model_name.lower():
                    print(f"✅ Selected {purpose}: {model_name}")
                    return model_name

            print(f"❌ Invalid choice. Please select 1-{len(available_models)} or enter a valid model name.")

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")

def prompt_dual_model_selection(available_models: List[str]) -> tuple[str, str]:
    """Prompt user to select both chat and evaluator models."""
    print("\n🩺 Lab Technician AI Assistant - Model Selection")
    print("=" * 60)
    print("You can select different models for:")
    print("1. Chat Model: Generates responses as the lab technician")
    print("2. Evaluator Model: Evaluates response quality")
    print("=" * 60)

    # Select chat model
    chat_model = prompt_model_selection(available_models, "Chat Model")

    # Select evaluator model
    evaluator_model = prompt_model_selection(available_models, "Evaluator Model")

    print(f"\n🎯 Configuration:")
    print(f"   Chat Model: {chat_model}")
    print(f"   Evaluator Model: {evaluator_model}")

    return chat_model, evaluator_model

def initialize_llm_manager():
    """Initialize the LLM manager globally."""
    global llm_manager, chat_model, evaluator_model

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
        # Prompt user to select both chat and evaluator models
        chat_model, evaluator_model = prompt_dual_model_selection(available_models)
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
    print(f"Chat model: {chat_model}")
    print(f"Evaluator model: {evaluator_model}")

def update_system_prompt(system_prompt: str):
    """Update the current system prompt."""
    global current_system_prompt
    current_system_prompt = system_prompt
    return f"System prompt updated! ({len(system_prompt)} chars)"


def chat(message, history):
    """Chat function compatible with gr.ChatInterface - exactly like lab3.ipynb pattern."""
    global llm_manager, chat_model, current_system_prompt, evaluation_enabled

    if not message.strip():
        return ""

    if not llm_manager or not chat_model:
        return "❌ No LLM available. Please check your setup."

    # Get model key for chat
    model_key = llm_manager.get_model_key_from_name(chat_model)
    if not model_key:
        return f"❌ Chat model '{chat_model}' not found."

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

    if not response:
        return f"❌ Failed to get response from {chat_model}. Please try again."

    # Evaluate response if evaluation is enabled
    if evaluation_enabled:
        # Convert gr.ChatInterface history format to evaluation format
        eval_history = []
        for user_msg, bot_msg in history:
            eval_history.append({"role": "user", "content": user_msg})
            eval_history.append({"role": "assistant", "content": bot_msg})

        evaluation = evaluate_response(response, message, eval_history)

        if not evaluation.is_acceptable:
            # Response is not acceptable - append evaluation feedback
            response += f"\n\n⚠️ **Quality Check:** {evaluation.feedback}"
        else:
            # Response is acceptable - optionally show positive feedback
            if "Evaluation completed" not in evaluation.feedback:
                response += f"\n\n✅ **Quality Check:** {evaluation.feedback}"

    return response

def cli_chat_interface():
    """Run the CLI chat interface."""
    global llm_manager, chat_model, evaluator_model, current_system_prompt

    if not llm_manager:
        initialize_llm_manager()

    print(f"\n🩺 Lab Technician AI Assistant")
    print("=" * 60)
    print(f"Chat Model: {chat_model}")
    print(f"Evaluator Model: {evaluator_model}")
    print()
    print("I'm Dr. LLM, your lab technician assistant!")
    print("Ask me questions about blood reports, lab test results, and medical diagnostics.")
    print()
    print("Commands:")
    print("  /help     - Show this help")
    print("  /system   - Set system prompt")
    print("  /clear    - Clear conversation history")
    print("  /eval     - Toggle response evaluation on/off")
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
                    print("  /eval     - Toggle response evaluation on/off")
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
                elif user_input == '/eval':
                    global evaluation_enabled
                    evaluation_enabled = not evaluation_enabled
                    status = "enabled" if evaluation_enabled else "disabled"
                    print(f"🔍 Response evaluation {status}.")
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
                # Evaluate response if evaluation is enabled
                if evaluation_enabled:
                    evaluation = evaluate_response(response, user_input, conversation_history)

                    if not evaluation.is_acceptable:
                        print(f"🩺 Dr. LLM: {response}")
                        print(f"⚠️ **Quality Check:** {evaluation.feedback}")
                    else:
                        print(f"🩺 Dr. LLM: {response}")
                        if "Evaluation completed" not in evaluation.feedback:
                            print(f"✅ **Quality Check:** {evaluation.feedback}")
                else:
                    print(f"🩺 Dr. LLM: {response}")

                conversation_history.append({"role": "assistant", "content": response})
            else:
                print(f"\n❌ Failed to get response from {chat_model}")

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
        print(f"Chat model: {chat_model}")
        print(f"Evaluator model: {evaluator_model}")
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
