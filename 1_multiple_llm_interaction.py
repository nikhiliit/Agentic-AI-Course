#!/usr/bin/env python3
"""
Interactive LLM Chat Interface

This script provides an interactive command-line interface for chatting with multiple LLMs.
Supports OpenAI GPT, Google Gemini, and local Ollama models.
Only enables models for which API keys are available in the environment.
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Dict, Any
import requests


class LLMManager:
    """Manages multiple LLM providers with automatic availability detection."""

    def __init__(self):
        # Load environment variables
        load_dotenv(override=True)

        # Initialize API keys
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.google_key = os.getenv('GOOGLE_API_KEY')
        self.ollama_available = self._check_ollama()

        # Initialize clients
        self.openai_client = None
        self.google_client = None
        self.ollama_client = None

        # Track available models
        self.available_models = self._initialize_clients()

        print("🤖 LLM Manager Initialized")
        print(f"Available models: {', '.join(self.available_models.keys())}")
        print()

    def _check_ollama(self) -> bool:
        """Check if Ollama is running locally."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def _initialize_clients(self) -> Dict[str, Any]:
        """Initialize available LLM clients based on API keys."""
        available = {}

        # OpenAI
        if self.openai_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_key)
                available["openai"] = {
                    "client": self.openai_client,
                    "model": "gpt-4o-mini",  # Fast and reliable
                    "name": "OpenAI GPT-4o-mini"
                }
                print("✅ OpenAI client initialized")
            except Exception as e:
                print(f"❌ Failed to initialize OpenAI: {e}")

        # Google Gemini
        if self.google_key:
            try:
                self.google_client = OpenAI(
                    api_key=self.google_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
                available["google"] = {
                    "client": self.google_client,
                    "model": "gemini-2.5-flash",  # As shown in notebook
                    "name": "Google Gemini 2.5 Flash"
                }
                print("✅ Google Gemini client initialized")
            except Exception as e:
                print(f"❌ Failed to initialize Google Gemini: {e}")

        # Ollama (local)
        if self.ollama_available:
            try:
                self.ollama_client = OpenAI(
                    base_url='http://localhost:11434/v1',
                    api_key='ollama'
                )
                available["ollama"] = {
                    "client": self.ollama_client,
                    "model": "qwen3:1.7b",
                    "name": "Ollama Qwen3 1.7B"
                }
                print("✅ Ollama client initialized")
            except Exception as e:
                print(f"❌ Failed to initialize Ollama: {e}")
        else:
            # Ollama not available - show installation message in red
            print("\033[91m" + "#!ollama pull qwen3:1.7b  # Current default model" + "\033[0m")
            print("\033[91m" + "#!Other options: qwen3:0.6b (fastest), qwen3:4b (better quality), qwen3:8b (high quality)" + "\033[0m")

        if not available:
            print("❌ No LLM clients could be initialized. Please check your API keys and Ollama setup.")
            sys.exit(1)

        return available

    def chat_with_model(self, model_key: str, message: str) -> Optional[str]:
        """Send a message to a specific model and return the response."""
        if model_key not in self.available_models:
            print(f"❌ Model '{model_key}' is not available.")
            return None

        model_info = self.available_models[model_key]
        client = model_info["client"]
        model = model_info["model"]
        name = model_info["name"]

        try:
            print(f"🤔 Thinking with {name}...")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}]
            )

            # Handle Qwen3 models which put content in 'reasoning' field
            message_obj = response.choices[0].message
            content = message_obj.content

            # If content is None or empty, try reasoning field (Qwen3 models)
            if not content and getattr(message_obj, 'reasoning', None):
                content = message_obj.reasoning

            # If content is still None, this is a failed response (likely filtered)
            if content is None:
                return None

            return content
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
            return None


class InteractiveCLI:
    """Interactive command-line interface for LLM chat."""

    def __init__(self):
        self.llm_manager = LLMManager()
        self.chat_history = []

    def display_menu(self):
        """Display the main menu."""
        print("\n" + "="*50)
        print("🤖 Interactive LLM Chat Interface")
        print("="*50)
        print("\nAvailable models:")
        for i, (key, info) in enumerate(self.llm_manager.available_models.items(), 1):
            print(f"  {i}. {info['name']} ({key})")

        print("\nCommands:")
        print("  [number] - Chat with specific model")
        print("  all      - Chat with all available models")
        print("  compare  - Compare responses from all models")
        print("  chain    - Chain multiple models (response to response)")
        print("  history  - Show chat history")
        print("  clear    - Clear chat history")
        print("  exit     - Exit the program")
        print()

    def get_user_choice(self) -> str:
        """Get user choice from menu."""
        while True:
            choice = input("Choose an option: ").strip().lower()
            if choice in ['exit', 'quit', 'q']:
                return 'exit'
            elif choice in ['all', 'compare', 'chain', 'history', 'clear']:
                return choice
            elif choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(self.llm_manager.available_models):
                    return str(choice_num)
            print("❌ Invalid choice. Please try again.")

    def chat_with_model_interactive(self, model_key: str):
        """Interactive chat with a specific model."""
        model_info = self.llm_manager.available_models[model_key]
        model_name = model_info["name"]

        print(f"\n🗣️  Starting chat with {model_name}")
        print("Type 'back' to return to main menu, 'exit' to quit.")

        while True:
            try:
                user_input = input(f"\nYou ({model_name}): ").strip()
                if user_input.lower() in ['back', 'b']:
                    break
                elif user_input.lower() in ['exit', 'quit', 'q']:
                    sys.exit(0)
                elif user_input:
                    response = self.llm_manager.chat_with_model(model_key, user_input)
                    if response:
                        print(f"\n🤖 {model_name}:")
                        print(response)
                        self.chat_history.append({
                            "model": model_name,
                            "user": user_input,
                            "response": response
                        })
                    else:
                        print("❌ Failed to get response. Try again or choose a different model.")
                else:
                    print("Please enter a message or command.")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                sys.exit(0)
            except EOFError:
                print("\n👋 Goodbye!")
                sys.exit(0)

    def chat_with_all_models(self, message: str):
        """Send the same message to all available models."""
        if not message.strip():
            print("❌ Please provide a message.")
            return

        print(f"\n📤 Sending to all models: '{message}'")
        print("-" * 50)

        for model_key, model_info in self.llm_manager.available_models.items():
            print(f"\n🤖 {model_info['name']}:")
            response = self.llm_manager.chat_with_model(model_key, message)
            if response:
                print(response)
                self.chat_history.append({
                    "model": model_info["name"],
                    "user": message,
                    "response": response
                })
            else:
                print("❌ Failed to get response")
            print("-" * 30)

    def compare_models(self):
        """Compare model responses to the same question."""
        question = input("Enter a question to compare across all models: ").strip()
        if not question:
            print("❌ Question cannot be empty.")
            return

        self.chat_with_all_models(question)

        # Ask if user wants to judge the responses
        judge = input("\n🤔 Would you like to have an AI judge these responses? (y/n): ").strip().lower()
        if judge == 'y':
            self.judge_responses(question)

    def judge_responses(self, original_question: str):
        """Use AI to judge and rank the responses."""
        if not self.chat_history:
            print("❌ No responses to judge.")
            return

        # Get the most recent set of responses
        recent_responses = []
        seen_models = set()

        for entry in reversed(self.chat_history):
            if entry["model"] not in seen_models:
                recent_responses.append(entry)
                seen_models.add(entry["model"])

        if len(recent_responses) < 2:
            print("❌ Need at least 2 different models to compare.")
            return

        # Create judge prompt
        judge_prompt = f"""You are judging a competition between {len(recent_responses)} AI models.
Each model answered this question: "{original_question}"

Please evaluate each response for:
1. Accuracy and correctness
2. Clarity and coherence
3. Completeness of answer
4. Helpfulness and relevance

Rank them from best to worst and briefly explain your reasoning.

"""

        for i, response in enumerate(recent_responses, 1):
            judge_prompt += f"\nModel {i} ({response['model']}):\n{response['response']}\n"

        judge_prompt += "\n\nProvide your ranking and explanation:"

        print("\n🧑‍⚖️ Asking GPT to judge the responses...")

        # Use OpenAI to judge if available, otherwise first available model
        judge_model = "openai" if "openai" in self.llm_manager.available_models else list(self.llm_manager.available_models.keys())[0]

        judgment = self.llm_manager.chat_with_model(judge_model, judge_prompt)
        if judgment:
            print(f"\n🎯 Judgment from {self.llm_manager.available_models[judge_model]['name']}:")
            print(judgment)
        else:
            print("❌ Failed to get judgment.")

    def show_history(self):
        """Display chat history."""
        if not self.chat_history:
            print("📝 No chat history yet.")
            return

        print("\n📜 Chat History:")
        print("="*50)
        for i, entry in enumerate(self.chat_history[-10:], 1):  # Show last 10
            print(f"{i}. {entry['model']}")
            print(f"   Q: {entry['user']}")
            print(f"   A: {entry['response'][:100]}{'...' if len(entry['response']) > 100 else ''}")
            print("-" * 30)

    def clear_history(self):
        """Clear chat history."""
        self.chat_history.clear()
        print("🧹 Chat history cleared.")

    def chain_models(self):
        """Chain multiple models where each response becomes the next input."""
        available_models = list(self.llm_manager.available_models.keys())

        print("\n🔗 LLM Chaining Mode")
        print("Each model's response will become the input for the next model.")
        model_list = [f'{i+1}. {self.llm_manager.available_models[m]["name"]}' for i, m in enumerate(available_models)]
        print(f"Available models: {', '.join(model_list)}")

        # Get chain sequence
        chain_input = input("\nEnter model numbers to chain (e.g., '1 2 3' for first→second→third): ").strip()
        try:
            chain_indices = [int(x.strip()) - 1 for x in chain_input.split()]
            if not chain_indices:
                print("❌ No models selected.")
                return

            # Validate indices
            for idx in chain_indices:
                if idx < 0 or idx >= len(available_models):
                    print(f"❌ Invalid model number: {idx + 1}")
                    return

            chain_models = [available_models[idx] for idx in chain_indices]

        except ValueError:
            print("❌ Invalid input format. Use numbers separated by spaces.")
            return

        # Get initial query
        initial_query = input("Enter the initial query: ").strip()
        if not initial_query:
            print("❌ Query cannot be empty.")
            return

        print(f"\n🔗 Starting chain: {' → '.join([self.llm_manager.available_models[m]['name'] for m in chain_models])}")
        print(f"Initial query: {initial_query}")
        print("-" * 60)

        current_message = initial_query
        chain_responses = []

        for i, model_key in enumerate(chain_models):
            model_info = self.llm_manager.available_models[model_key]
            model_name = model_info["name"]

            print(f"\n🤖 Step {i+1}: {model_name}")
            if i > 0:
                print(f"Input from previous: {current_message[:100]}{'...' if len(current_message) > 100 else ''}")

            response = self.llm_manager.chat_with_model(model_key, current_message)

            if response:
                print(f"Response: {response}")
                chain_responses.append({
                    "step": i+1,
                    "model": model_name,
                    "input": current_message,
                    "response": response
                })
                current_message = response  # Pass response to next model
            else:
                print(f"⚠️  No response from {model_name}, stopping chain")
                print("❌ Failed to get response. Chain stopped.")
                break

        print("\n🔗 Chain completed!")
        print(f"Total steps: {len(chain_responses)}")

        # Ask if user wants to see full chain summary
        show_summary = input("\nShow full chain summary? (y/n): ").strip().lower()
        if show_summary == 'y':
            self.show_chain_summary(chain_responses)

    def show_chain_summary(self, chain_responses):
        """Show a summary of the entire chain."""
        print("\n📋 Chain Summary:")
        print("=" * 60)
        for response in chain_responses:
            print(f"Step {response['step']}: {response['model']}")
            print(f"  Input: {response['input'][:80]}{'...' if len(response['input']) > 80 else ''}")
            print(f"  Output: {response['response'][:80]}{'...' if len(response['response']) > 80 else ''}")
            print("-" * 40)

    def run(self):
        """Main CLI loop."""
        print("Welcome to the Interactive LLM Chat Interface!")
        print("This tool lets you chat with multiple AI models interactively.")

        while True:
            self.display_menu()
            choice = self.get_user_choice()

            if choice == 'exit':
                print("👋 Goodbye!")
                break
            elif choice == 'all':
                message = input("Enter your message for all models: ").strip()
                self.chat_with_all_models(message)
            elif choice == 'compare':
                self.compare_models()
            elif choice == 'chain':
                self.chain_models()
            elif choice == 'history':
                self.show_history()
            elif choice == 'clear':
                self.clear_history()
            elif choice.isdigit():
                model_keys = list(self.llm_manager.available_models.keys())
                model_index = int(choice) - 1
                self.chat_with_model_interactive(model_keys[model_index])


def main():
    """Main entry point."""
    try:
        cli = InteractiveCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
