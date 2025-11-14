from dotenv import load_dotenv
from agents import Agent, Runner, trace, function_tool
from openai.types.responses import ResponseTextDeltaEvent
import asyncio
import json

# Load environment variables
load_dotenv(override=True)

# ============================================================================
# CONTENT GENERATOR AGENTS (different writing styles)
# ============================================================================

professional_instructions = """You are a professional content writer specializing in technical articles.
Write clear, informative content that explains complex topics in an accessible way.
Focus on facts, structure, and educational value."""

creative_instructions = """You are a creative storyteller who makes topics engaging and memorable.
Write content that uses analogies, stories, and vivid language to captivate readers.
Make learning fun and exciting."""

concise_instructions = """You are a concise technical writer who gets straight to the point.
Write clear, brief content that delivers maximum information with minimum words.
Be direct and efficient."""

# Create the content generator agents
professional_writer = Agent(
    name="Professional Writer",
    instructions=professional_instructions,
    model="gpt-4o-mini"
)

creative_writer = Agent(
    name="Creative Writer",
    instructions=creative_instructions,
    model="gpt-4o-mini"
)

concise_writer = Agent(
    name="Concise Writer",
    instructions=concise_instructions,
    model="gpt-4o-mini"
)

# ============================================================================
# FORMATTER AGENT (receives handoffs)
# ============================================================================

formatter_instructions = """You are a content formatter and publisher.
You receive raw content and format it into a professional publication-ready format.

Your tasks:
1. Add proper headings and structure
2. Format with markdown for readability
3. Add publication metadata (title, author, date)
4. Make it ready for publication

You are the final step in the content creation pipeline."""

formatter_agent = Agent(
    name="Content Formatter",
    instructions=formatter_instructions,
    model="gpt-4o-mini",
    handoff_description="Format and publish the final content"
)

# ============================================================================
# SIMPLE TOOL FUNCTION (instead of complex email sending)
# ============================================================================

@function_tool
def publish_content(title: str, formatted_content: str) -> dict:
    """
    Publish the formatted content by saving it to a file and returning publication info.

    Args:
        title: The title of the content
        formatted_content: The fully formatted content ready for publication
    """
    # Simple file output instead of email sending
    filename = f"published_content_{title.lower().replace(' ', '_')[:20]}.md"

    with open(filename, 'w') as f:
        f.write(formatted_content)

    print(f"✅ Content published to: {filename}")

    return {
        "status": "published",
        "filename": filename,
        "word_count": len(formatted_content.split()),
        "title": title
    }

# ============================================================================
# CONTENT MANAGER (main orchestrating agent)
# ============================================================================

content_manager_instructions = """
You are a Content Manager responsible for creating high-quality published content.

Follow these steps carefully:

1. Generate Drafts: Use all three writer agents to generate different versions of the requested content. Wait for all drafts to be ready.

2. Evaluate and Select: Review all drafts and choose the single best one based on quality, engagement, and suitability for the topic. You can regenerate drafts if needed.

3. Handoff for Publishing: Pass ONLY the winning content draft to the 'Content Formatter' agent. The formatter will handle final formatting and publishing.

Critical Rules:
- You must use the writer agents to generate content — do not write it yourself.
- You must hand off exactly ONE piece of content to the Content Formatter.
- Focus on selecting content that will engage and inform the target audience.
"""

# Convert writer agents to tools
writer_tool_1 = professional_writer.as_tool(
    tool_name="professional_writer",
    tool_description="Generate professional, technical content"
)
writer_tool_2 = creative_writer.as_tool(
    tool_name="creative_writer",
    tool_description="Generate creative, engaging content with stories"
)
writer_tool_3 = concise_writer.as_tool(
    tool_name="concise_writer",
    tool_description="Generate clear, concise content"
)

# Set up tools and handoffs
tools = [writer_tool_1, writer_tool_2, writer_tool_3]
handoffs = [formatter_agent]

content_manager = Agent(
    name="Content Manager",
    instructions=content_manager_instructions,
    tools=tools,
    handoffs=handoffs,
    model="gpt-4o-mini"
)

# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

async def demo_parallel_generation():
    """Demo: Generate content in parallel using all writers"""
    print("🚀 Demo 1: Parallel Content Generation")
    print("=" * 50)

    message = "Write a 300-word article about the benefits of renewable energy"

    with trace("Parallel content generation"):
        results = await asyncio.gather(
            Runner.run(professional_writer, message),
            Runner.run(creative_writer, message),
            Runner.run(concise_writer, message),
        )

    outputs = [result.final_output for result in results]

    for i, output in enumerate(outputs, 1):
        print(f"\n📝 Writer {i} Output:")
        print("-" * 30)
        print(output[:200] + "..." if len(output) > 200 else output)
        print()

async def demo_with_selection():
    """Demo: Generate content and let an agent select the best"""
    print("\n🎯 Demo 2: Content Selection by Agent")
    print("=" * 50)

    message = "Write a 300-word article about the benefits of renewable energy"

    # Create a simple selector agent (like the sales picker)
    selector_instructions = """You are a content quality evaluator. Review the provided content options and select the single best one that would most effectively inform and engage readers about renewable energy. Reply with only the selected content, no explanation."""

    selector = Agent(
        name="Content Selector",
        instructions=selector_instructions,
        model="gpt-4o-mini"
    )

    with trace("Content selection"):
        # Generate drafts
        results = await asyncio.gather(
            Runner.run(professional_writer, message),
            Runner.run(creative_writer, message),
            Runner.run(concise_writer, message),
        )

        outputs = [result.final_output for result in results]
        content_options = "\n\n---CONTENT OPTION---\n\n".join(outputs)

        # Select best
        best_result = await Runner.run(selector, f"Select the best content:\n\n{content_options}")

        print("✅ Selected Best Content:")
        print("-" * 30)
        print(best_result.final_output[:300] + "..." if len(best_result.final_output) > 300 else best_result.final_output)

async def demo_full_handoff():
    """Demo: Full handoff workflow with content manager"""
    print("\n🔄 Demo 3: Full Handoff Workflow")
    print("=" * 50)

    # Add the publish tool to the formatter's tools
    formatter_agent.tools = [publish_content]

    message = "Create and publish a 300-word article about artificial intelligence in healthcare"

    with trace("Full content creation handoff"):
        result = await Runner.run(content_manager, message)

    print("✅ Handoff workflow completed!")
    print("The Content Manager generated drafts, selected the best one,")
    print("then handed off to the Content Formatter for final publishing.")
    print(f"Result: {result.final_output}")

async def main():
    """Run all demos"""
    print("🤖 Simple Handoff Demo - Content Creation Workflow")
    print("This demonstrates agent handoffs using LLMs as tools")
    print("=" * 60)

    try:
        await demo_parallel_generation()
        await demo_with_selection()
        await demo_full_handoff()

        print("\n🎉 All demos completed successfully!")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
