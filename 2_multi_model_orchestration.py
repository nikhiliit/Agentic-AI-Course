#!/usr/bin/env python3
"""
Multi-Model LLM Orchestration System

This script orchestrates multiple LLMs in a coordinated workflow:
1. Question Generator: Creates challenging questions
2. Answer Providers: Multiple LLMs provide responses (2 by default)
3. Evaluator: Judges and ranks all responses

Features:
- Manual and automatic model selection for each role (Question Generator, Answer Providers, Evaluator)
- Support for OpenAI, Google Gemini, and Ollama models
- Independent script with no external dependencies
- Interactive CLI for model selection and orchestration
- Flexible model assignment: same model can be used for multiple roles
- Robust error handling with automatic retry logic
- Graceful handling of API failures and model overloads
- Auto-selection prioritizes stable models (Ollama > OpenAI > Gemini)
- Connection testing and fallback model selection for Gemini

Based on the workflow Orchestrator-Worker Workflow from our notes with 4-agent orchestration.
"""

import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Dict, Any, List
import requests
import dotenv
load_dotenv(override=True)

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
                # Test the connection briefly
                try:
                    test_response = self.google_client.chat.completions.create(
                        model="gemini-2.5-flash",
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=10
                    )
                    available["google"] = {
                        "client": self.google_client,
                        "model": "gemini-2.5-flash",  # As shown in notebook
                        "name": "Google Gemini 2.5 Flash"
                    }
                    print("✅ Google Gemini client initialized")
                except Exception as test_e:
                    print(f"⚠️  Google Gemini connection test failed: {test_e}")
                    print("   Gemini 2.5 Flash is often overloaded. Trying more stable model...")

                    # Try a more stable Gemini model
                    try:
                        stable_test = self.google_client.chat.completions.create(
                            model="gemini-1.5-flash",
                            messages=[{"role": "user", "content": "test"}],
                            max_tokens=10
                        )
                        available["google"] = {
                            "client": self.google_client,
                            "model": "gemini-1.5-flash",  # More stable alternative
                            "name": "Google Gemini 1.5 Flash (Stable)"
                        }
                        print("✅ Google Gemini client initialized (using stable 1.5 Flash model)")
                    except Exception as stable_e:
                        print(f"⚠️  Stable Gemini model also failed: {stable_e}")
                        print("   You can still select Gemini, but it might fail during orchestration.")
                        available["google"] = {
                            "client": self.google_client,
                            "model": "gemini-2.5-flash",
                            "name": "Google Gemini 2.5 Flash ⚠️ (Unstable)"
                        }
                        print("✅ Google Gemini client initialized (with warnings - may be unstable)")
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
                    "model": "llama3.2:latest",
                    "name": "Ollama Llama3.2 Latest"
                }
                print("✅ Ollama client initialized")
            except Exception as e:
                print(f"❌ Failed to initialize Ollama: {e}")
        else:
            # Ollama not available - show installation message in red
            print("\033[91m" + "#!ollama pull llama3.2:latest  # Current default model" + "\033[0m")
            print("\033[91m" + "#!Other options: llama3.2:1b (fastest), llama3.2:3b (balanced), llama3.3:70b (high quality)" + "\033[0m")

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

            # Extract response content
            message_obj = response.choices[0].message
            content = message_obj.content

            # If content is still None, this is a failed response (likely filtered)
            if content is None:
                return None

            return content
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
            return None


class QuestionGenerator:
    """Generates challenging questions for LLM evaluation."""

    def __init__(self, llm_manager, model_key: str):
        self.llm_manager = llm_manager
        self.model_key = model_key

    def generate_question(self, topic: str = "general AI/ML", max_retries: int = 2) -> Optional[str]:
        """Generate a challenging question on the given topic with retry logic."""
        prompt = f"""Please come up with a challenging, nuanced question about {topic} that I can ask multiple LLMs to evaluate their intelligence and reasoning capabilities.

Answer only with the question, no explanation or additional text."""

        model_info = self.llm_manager.available_models[self.model_key]
        print(f"🤔 Question Generator ({model_info['name']}) creating question about: {topic}")

        import time
        for attempt in range(max_retries + 1):
            response = self.llm_manager.chat_with_model(self.model_key, prompt)

            if response:
                print(f"✅ Generated question: {response}")
                return response
            else:
                if attempt < max_retries:
                    print(f"⚠️  Question generation attempt {attempt+1} failed, retrying...")
                    time.sleep(2)
                else:
                    print("❌ Failed to generate question after all attempts")
                    return None


class AnswerProvider:
    """Handles getting answers from multiple LLMs."""

    def __init__(self, llm_manager, model_keys: List[str]):
        self.llm_manager = llm_manager
        self.model_keys = model_keys

    def get_answers(self, question: str, max_retries: int = 2, retry_delay: int = 3) -> List[Dict[str, Any]]:
        """Get answers from the specified LLMs with retry logic."""
        answers = []
        import time

        print(f"🗣️  Getting answers from {len(self.model_keys)} selected models...")

        for i, model_key in enumerate(self.model_keys):
            model_info = self.llm_manager.available_models[model_key]
            success = False

            print(f"🤖 Provider {i+1}: {model_info['name']}")

            # Try multiple times with delay
            for attempt in range(max_retries + 1):
                response = self.llm_manager.chat_with_model(model_key, question)

                if response:
                    answers.append({
                        "provider_id": i+1,
                        "model_name": model_info['name'],
                        "model_key": model_key,
                        "question": question,
                        "answer": response
                    })
                    print("✅ Answer received" + (f" (after {attempt+1} attempts)" if attempt > 0 else ""))
                    success = True
                    break
                else:
                    if attempt < max_retries:
                        print(f"⚠️  Attempt {attempt+1} failed, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print(f"❌ Failed to get answer after {max_retries+1} attempts")

            if not success:
                # Add a placeholder entry for failed model
                answers.append({
                    "provider_id": i+1,
                    "model_name": model_info['name'],
                    "model_key": model_key,
                    "question": question,
                    "answer": "[FAILED] Unable to get response from this model",
                    "failed": True
                })

        return answers


class Evaluator:
    """Evaluates and ranks responses from multiple LLMs."""

    def __init__(self, llm_manager, model_key: str):
        self.llm_manager = llm_manager
        self.model_key = model_key

    def evaluate_responses(self, question: str, responses: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Evaluate and rank the responses using the specified LLM."""
        if not responses:
            print("❌ No responses to evaluate")
            return None

        # Check for successful responses
        successful_responses = [r for r in responses if not r.get("failed", False)]
        failed_count = len(responses) - len(successful_responses)

        if failed_count > 0:
            print(f"⚠️  {failed_count} model(s) failed to respond. Evaluating with {len(successful_responses)} successful responses.")

        if len(successful_responses) < 2:
            print("⚠️  Need at least 2 successful responses for meaningful evaluation.")
            return {
                "warning": "Insufficient responses for ranking",
                "successful_responses": len(successful_responses),
                "failed_responses": failed_count
            }

        # Prepare the evaluation prompt
        evaluation_prompt = self._create_evaluation_prompt(question, successful_responses)

        model_info = self.llm_manager.available_models[self.model_key]
        print(f"🧑‍⚖️  Evaluator ({model_info['name']}) judging {len(successful_responses)} responses...")

        # Try evaluation with retry logic
        import time
        max_retries = 2
        judgment = None

        for attempt in range(max_retries + 1):
            judgment = self.llm_manager.chat_with_model(self.model_key, evaluation_prompt)

            if judgment:
                break
            else:
                if attempt < max_retries:
                    print(f"⚠️  Evaluation attempt {attempt+1} failed, retrying...")
                    time.sleep(3)
                else:
                    print("❌ Failed to get evaluation after all attempts")

        if judgment:
            try:
                # Try to parse JSON response
                result = json.loads(judgment)
                return result
            except json.JSONDecodeError:
                print("⚠️  Evaluator didn't return valid JSON, returning raw response")
                return {"raw_judgment": judgment}
        else:
            print("❌ Failed to get evaluation")
            return None

    def _create_evaluation_prompt(self, question: str, responses: List[Dict[str, Any]]) -> str:
        """Create the evaluation prompt for the judge LLM."""
        prompt = f"""You are judging a competition between {len(responses)} AI models.
Each model has been given this question:

"{question}"

Your job is to evaluate each response for:
1. Accuracy and correctness
2. Clarity and coherence
3. Completeness of answer
4. Helpfulness and relevance
5. Originality and insight

Rank them from best to worst and provide a brief explanation for your ranking.

Respond with JSON in this format:
{{
    "ranking": [1, 2, 3, ...],  // provider IDs in ranked order (best to worst)
    "explanation": "Brief explanation of your ranking criteria and reasoning"
}}

Here are the responses from each provider:

"""

        for response in responses:
            prompt += f"\nProvider {response['provider_id']} ({response['model_name']}):\n{response['answer']}\n"

        prompt += "\n\nProvide your ranking and explanation in JSON format:"
        return prompt


class Orchestrator:
    """Coordinates the multi-agent LLM workflow."""

    def __init__(self):
        self.llm_manager = LLMManager()
        self.question_generator = None
        self.answer_provider = None
        self.evaluator = None
        self.selected_models = {}

    def select_models_manually(self):
        """Allow user to manually select models for each role."""
        available_models = self.llm_manager.available_models

        if len(available_models) < 1:
            print("❌ No models available for orchestration.")
            return

        print("\n🤖 Select models for each orchestration role:")
        print("=" * 50)
        print("💡 You can select the same model for multiple roles if desired.")
        print("💡 Models marked with ⚠️ may be unstable due to server overload.")

        # Role 1: Question Generator
        print("\n1. 📝 Question Generator - Creates challenging questions")
        qg_model = self._select_model_for_role("Question Generator", available_models)
        self.selected_models['question_generator'] = qg_model
        self.question_generator = QuestionGenerator(self.llm_manager, qg_model)

        # Role 2: Answer Provider 1
        print("\n2. 🗣️  Answer Provider 1 - First model to answer questions")
        ap1_model = self._select_model_for_role("Answer Provider 1", available_models)
        self.selected_models['answer_provider_1'] = ap1_model

        # Role 3: Answer Provider 2
        print("\n3. 🗣️  Answer Provider 2 - Second model to answer questions")
        ap2_model = self._select_model_for_role("Answer Provider 2", available_models)
        self.selected_models['answer_provider_2'] = ap2_model

        self.answer_provider = AnswerProvider(self.llm_manager, [ap1_model, ap2_model])

        # Role 4: Evaluator
        print("\n4. 🧑‍⚖️  Evaluator - Judges and ranks all responses")
        eval_model = self._select_model_for_role("Evaluator", available_models)
        self.selected_models['evaluator'] = eval_model
        self.evaluator = Evaluator(self.llm_manager, eval_model)

        print("\n✅ Model selection complete!")
        print("Selected models:")
        for role, model_key in self.selected_models.items():
            model_info = available_models[model_key]
            print(f"  {role.replace('_', ' ').title()}: {model_info['name']}")

        # Show duplicate usage if any
        model_usage = {}
        for role, model_key in self.selected_models.items():
            model_usage[model_key] = model_usage.get(model_key, []) + [role]

        duplicates = [model for model, roles in model_usage.items() if len(roles) > 1]
        if duplicates:
            print("\n📊 Note: The following models are used for multiple roles:")
            for model_key in duplicates:
                model_info = available_models[model_key]
                roles = model_usage[model_key]
                print(f"  {model_info['name']}: {', '.join([r.replace('_', ' ').title() for r in roles])}")

    def _select_model_for_role(self, role_name: str, available_models: Dict) -> str:
        """Helper method to select a model for a specific role."""
        print(f"Available models for {role_name}:")
        for i, (key, info) in enumerate(available_models.items(), 1):
            print(f"  {i}. {info['name']}")

        while True:
            try:
                choice = input(f"Select model number for {role_name} (1-{len(available_models)}): ").strip()
                choice_num = int(choice)
                if 1 <= choice_num <= len(available_models):
                    selected_key = list(available_models.keys())[choice_num - 1]
                    return selected_key
                else:
                    print(f"❌ Please enter a number between 1 and {len(available_models)}")
            except ValueError:
                print("❌ Please enter a valid number")

    def select_models_automatically(self):
        """Automatically select stable models for orchestration."""
        available_models = self.llm_manager.available_models

        if len(available_models) < 1:
            print("❌ No models available for orchestration.")
            return

        print("\n🤖 Auto-selecting stable models for orchestration...")
        print("=" * 50)

        # Priority order for stability: Ollama (most stable) -> OpenAI -> Gemini (least stable)
        priority_order = []
        if "ollama" in available_models:
            priority_order.append("ollama")
        if "openai" in available_models:
            priority_order.append("openai")
        if "google" in available_models:
            # Only include Gemini if it's not marked as unstable
            if "⚠️" not in available_models["google"]["name"]:
                priority_order.append("google")

        if len(priority_order) < 1:
            print("❌ No stable models available. Please use manual selection.")
            return

        print(f"Available stable models: {', '.join([available_models[k]['name'] for k in priority_order])}")

        # Auto-assign based on priority
        self.selected_models['question_generator'] = priority_order[0]
        self.selected_models['answer_provider_1'] = priority_order[min(1, len(priority_order)-1)]
        self.selected_models['answer_provider_2'] = priority_order[min(2, len(priority_order)-1)]
        self.selected_models['evaluator'] = priority_order[min(3, len(priority_order)-1)]

        # Initialize components
        self.question_generator = QuestionGenerator(self.llm_manager, self.selected_models['question_generator'])
        self.answer_provider = AnswerProvider(self.llm_manager, [
            self.selected_models['answer_provider_1'],
            self.selected_models['answer_provider_2']
        ])
        self.evaluator = Evaluator(self.llm_manager, self.selected_models['evaluator'])

        print("\n✅ Auto-selection complete!")
        print("Selected models:")
        for role, model_key in self.selected_models.items():
            model_info = available_models[model_key]
            print(f"  {role.replace('_', ' ').title()}: {model_info['name']}")

        # Show if any models are reused
        model_usage = {}
        for role, model_key in self.selected_models.items():
            model_usage[model_key] = model_usage.get(model_key, []) + [role]

        duplicates = [model for model, roles in model_usage.items() if len(roles) > 1]
        if duplicates:
            print("\n📊 Note: The following models are used for multiple roles (limited stable models):")
            for model_key in duplicates:
                model_info = available_models[model_key]
                roles = model_usage[model_key]
                print(f"  {model_info['name']}: {', '.join([r.replace('_', ' ').title() for r in roles])}")

    def run_orchestration(self, topic: str = "general AI/ML") -> Dict[str, Any]:
        """Run the complete orchestration workflow."""
        # Check if models have been selected
        if not self.selected_models:
            print("❌ No models selected! Please run select_models_manually() first.")
            return {"error": "Models not selected"}

        print("🎭 Starting Multi-Model Orchestration")
        print("=" * 50)

        # Step 1: Generate question
        print("\n📝 Step 1: Question Generation")
        question = self.question_generator.generate_question(topic)

        if not question:
            return {"error": "Failed to generate question"}

        # Step 2: Get answers from multiple providers
        print("\n🗣️  Step 2: Gathering Responses")
        answers = self.answer_provider.get_answers(question)

        if not answers:
            return {"error": "Failed to get any answers"}

        # Step 3: Evaluate and rank responses
        print("\n🧑‍⚖️  Step 3: Evaluation & Ranking")
        evaluation = self.evaluator.evaluate_responses(question, answers)

        # Prepare final results
        results = {
            "question": question,
            "topic": topic,
            "responses": answers,
            "evaluation": evaluation,
            "timestamp": str(os.times())
        }

        self.display_results(results)
        return results

    def display_results(self, results: Dict[str, Any]):
        """Display the orchestration results in a nice format."""
        print("\n" + "="*60)
        print("🎭 MULTI-MODEL ORCHESTRATION RESULTS")
        print("="*60)

        print(f"\n📝 Question: {results['question']}")
        print(f"🏷️  Topic: {results['topic']}")

        print("\n🗣️  Responses:")
        failed_count = 0
        for response in results['responses']:
            status = "❌" if response.get("failed", False) else "✅"
            print(f"\n{status} Provider {response['provider_id']}: {response['model_name']}")
            if response.get("failed", False):
                print("   [FAILED] Unable to get response from this model")
                failed_count += 1
            else:
                print(f"   {response['answer'][:200]}{'...' if len(response['answer']) > 200 else ''}")

        if failed_count > 0:
            print(f"\n⚠️  {failed_count} model(s) failed to respond. This may affect evaluation quality.")

        if results['evaluation']:
            print("\n🧑‍⚖️  Evaluation:")
            if 'warning' in results['evaluation']:
                print(f"⚠️  Warning: {results['evaluation']['warning']}")
                print(f"   Successful responses: {results['evaluation'].get('successful_responses', 0)}")
                print(f"   Failed responses: {results['evaluation'].get('failed_responses', 0)}")
            elif 'ranking' in results['evaluation']:
                print(f"🏆 Ranking: {results['evaluation']['ranking']}")
            if 'explanation' in results['evaluation']:
                print(f"📋 Explanation: {results['evaluation']['explanation']}")
            elif 'raw_judgment' in results['evaluation']:
                print(f"📋 Raw Judgment: {results['evaluation']['raw_judgment']}")

        print("\n" + "="*60)


class OrchestrationCLI:
    """Command-line interface for the orchestration system."""

    def __init__(self):
        self.orchestrator = Orchestrator()

    def display_menu(self):
        """Display the main menu."""
        print("\n" + "="*50)
        print("🎭 Multi-Model LLM Orchestration")
        print("="*50)
        print("\nThis system coordinates multiple LLMs in a workflow:")
        print("1. 📝 Question Generator creates challenging questions")
        print("2. 🗣️  Multiple Answer Providers respond")
        print("3. 🧑‍⚖️  Evaluator judges and ranks responses")
        print("\nCommands:")
        print("  select  - Select models for each role manually")
        print("  auto    - Auto-select stable models (avoids potentially unstable ones)")
        print("  run     - Run the full orchestration workflow")
        print("  topic   - Run with custom topic")
        print("  help    - Show this help")
        print("  exit    - Exit the program")
        print()

    def get_user_choice(self) -> str:
        """Get user choice from menu."""
        while True:
            choice = input("Choose an option (select/auto/run/topic/help/exit): ").strip().lower()
            if choice in ['exit', 'quit', 'q']:
                return 'exit'
            elif choice in ['select', 'auto', 'run', 'topic', 'help', 'h']:
                return choice
            print("❌ Invalid choice. Please choose: select, auto, run, topic, help, or exit.")

    def select_models(self):
        """Allow user to select models for orchestration."""
        try:
            self.orchestrator.select_models_manually()
            print("\n✅ Models selected successfully! You can now run the orchestration.")
        except Exception as e:
            print(f"❌ Error during model selection: {e}")

    def auto_select_models(self):
        """Automatically select stable models for orchestration."""
        try:
            self.orchestrator.select_models_automatically()
            print("\n✅ Stable models auto-selected successfully! You can now run the orchestration.")
        except Exception as e:
            print(f"❌ Error during auto-selection: {e}")

    def run_workflow(self, topic: str = "general AI/ML"):
        """Run the orchestration workflow."""
        try:
            results = self.orchestrator.run_orchestration(topic)
            return results
        except Exception as e:
            print(f"❌ Error during orchestration: {e}")
            return None

    def run(self):
        """Main CLI loop."""
        print("Welcome to the Multi-Model LLM Orchestration System!")

        while True:
            self.display_menu()
            choice = self.get_user_choice()

            if choice == 'exit':
                print("👋 Goodbye!")
                break
            elif choice == 'help':
                continue  # Menu already displayed
            elif choice == 'select':
                self.select_models()
            elif choice == 'auto':
                self.auto_select_models()
            elif choice == 'run':
                if not self.orchestrator.selected_models:
                    print("❌ Please select models first using 'select' command.")
                    continue
                self.run_workflow()
            elif choice == 'topic':
                if not self.orchestrator.selected_models:
                    print("❌ Please select models first using 'select' command.")
                    continue
                topic = input("Enter a topic for the question: ").strip()
                if topic:
                    self.run_workflow(topic)
                else:
                    print("❌ Topic cannot be empty.")


def main():
    """Main entry point."""
    try:
        cli = OrchestrationCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
