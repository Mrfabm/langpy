"""
LangPy Post-Installation Script.

Runs after pip install to display helpful information.
"""


def post_install():
    """Post-installation hook for LangPy."""
    print()
    print("=" * 60)
    print("  LangPy installed successfully!")
    print("=" * 60)
    print()
    print("Quick Start:")
    print("-" * 60)
    print()
    print("  from langpy import Langpy")
    print()
    print("  lb = Langpy(api_key='your-api-key')")
    print()
    print("  # Use any of the 9 primitives:")
    print("  response = await lb.agent.run(")
    print("      model='openai:gpt-4',")
    print("      input='Hello!'")
    print("  )")
    print()
    print("Available Primitives:")
    print("-" * 60)
    print("  lb.agent    - Unified LLM API (100+ models)")
    print("  lb.pipe     - Single LLM call with templates")
    print("  lb.memory   - Vector storage and RAG")
    print("  lb.thread   - Conversation history")
    print("  lb.workflow - Multi-step orchestration")
    print("  lb.parser   - Document text extraction")
    print("  lb.chunker  - Text segmentation")
    print("  lb.embed    - Text to vectors")
    print("  lb.tools    - Pre-built and custom tools")
    print()
    print("Documentation:")
    print("-" * 60)
    print("  https://github.com/Mrfabm/langpy")
    print()
    print("Commands:")
    print("-" * 60)
    print("  langpy           - CLI help")
    print("  langpy-init      - Initialize a new project")
    print("  langpy-setup-docs - Set up MkDocs documentation")
    print()


if __name__ == "__main__":
    post_install()
