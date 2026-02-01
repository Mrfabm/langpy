"""
Thread Primitive
================
Manage conversation history for multi-turn interactions.

Threads enable:
    - Persistent conversation storage
    - Multi-turn context for LLMs
    - Conversation management (create, list, delete)
    - Tagging and metadata

Architecture:
    User → Thread.add_message() → Storage → Thread.get_messages() → LLM

    ┌───────────────────────────────┐
    │           Thread              │
    │  ┌─────────────────────────┐  │
    │  │    Message History      │  │
    │  │  ┌────┐ ┌────┐ ┌────┐   │  │
    │  │  │user│ │ ai │ │user│   │  │
    │  │  └────┘ └────┘ └────┘   │  │
    │  └─────────────────────────┘  │
    │            ↓                  │
    │     Context for LLM           │
    └───────────────────────────────┘
"""

import asyncio
import io
import os
import sys

from dotenv import load_dotenv
load_dotenv()

# Fix Windows console encoding for Unicode output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langpy_sdk import Thread, Pipe


# =============================================================================
# BASIC THREAD OPERATIONS
# =============================================================================

async def basic_thread_demo():
    """Demonstrate basic Thread create and message operations."""
    print("=" * 60)
    print("   BASIC THREAD - Create and Add Messages")
    print("=" * 60)
    print()

    thread = Thread()

    # Create a new thread
    print("1. Creating a thread:")
    print("-" * 40)
    thread_id = await thread.create(
        name="Support Chat",
        tags=["support", "demo"],
        metadata={"customer_id": "12345"}
    )
    print(f"   Thread ID: {thread_id}")
    print()

    # Add messages
    print("2. Adding messages:")
    print("-" * 40)

    await thread.add_message(thread_id, "user", "Hello, I need help with my order")
    print("   Added: [user] Hello, I need help with my order")

    await thread.add_message(thread_id, "assistant", "Hi! I'd be happy to help. What's your order number?")
    print("   Added: [assistant] Hi! I'd be happy to help...")

    await thread.add_message(thread_id, "user", "Order #12345")
    print("   Added: [user] Order #12345")

    print()

    # Get messages
    print("3. Retrieving messages:")
    print("-" * 40)

    messages = await thread.get_messages(thread_id)
    for msg in messages:
        print(f"   [{msg['role']}] {msg['content']}")

    print()

    # Clean up
    await thread.delete(thread_id)


# =============================================================================
# THREAD MANAGEMENT
# =============================================================================

async def thread_management_demo():
    """Demonstrate thread listing, updating, and deleting."""
    print("=" * 60)
    print("   THREAD MANAGEMENT - List, Update, Delete")
    print("=" * 60)
    print()

    thread = Thread()

    # Create multiple threads
    print("1. Creating multiple threads:")
    print("-" * 40)

    thread_ids = []
    for i, tag in enumerate(["sales", "support", "support"]):
        tid = await thread.create(
            name=f"Chat {i+1}",
            tags=[tag, "demo"]
        )
        thread_ids.append(tid)
        print(f"   Created: Chat {i+1} [{tag}]")

    print()

    # List all threads
    print("2. Listing all threads:")
    print("-" * 40)

    threads = await thread.list()
    for t in threads:
        print(f"   {t.name} - {t.message_count} messages [{', '.join(t.tags)}]")

    print()

    # Filter by tag
    print("3. Filtering by tag (support):")
    print("-" * 40)

    support_threads = await thread.list(tags=["support"])
    for t in support_threads:
        print(f"   {t.name} [{', '.join(t.tags)}]")

    print()

    # Update a thread
    print("4. Updating thread name and tags:")
    print("-" * 40)

    updated = await thread.update(
        thread_ids[0],
        name="VIP Sales Chat",
        tags=["sales", "vip", "demo"]
    )
    print(f"   Updated: {updated.name} [{', '.join(updated.tags)}]")

    print()

    # Clean up
    for tid in thread_ids:
        await thread.delete(tid)


# =============================================================================
# MULTI-TURN CONVERSATION
# =============================================================================

async def multi_turn_demo():
    """Demonstrate multi-turn conversation with an LLM."""
    print("=" * 60)
    print("   MULTI-TURN CONVERSATION - Thread + Pipe")
    print("=" * 60)
    print()

    thread = Thread()

    # Create conversation thread
    thread_id = await thread.create("Math Tutor Session")

    # Check if we can use Pipe (requires API key)
    import os
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))

    if has_api_key:
        pipe = Pipe(
            model="gpt-4o-mini",
            system="You are a helpful assistant. Keep responses concise."
        )

        async def chat(message: str) -> str:
            """Send message and get response, maintaining history."""
            await thread.add_message(thread_id, "user", message)
            history = await thread.get_messages(thread_id)

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                response = await pipe.run(history)

            await thread.add_message(thread_id, "assistant", response.content)
            return response.content

        print("Multi-turn math conversation:")
        print("-" * 40)

        conversation = [
            "What is 15 times 7?",
            "Now add 23 to that result",
        ]

        for message in conversation:
            print(f"\n   User: {message}")
            response = await chat(message)
            print(f"   Assistant: {response}")
    else:
        # Simulate conversation without API
        print("Simulated multi-turn conversation (no API key):")
        print("-" * 40)

        simulated = [
            ("user", "What is 15 times 7?"),
            ("assistant", "15 times 7 equals 105."),
            ("user", "Now add 23 to that result"),
            ("assistant", "105 + 23 = 128."),
        ]

        for role, content in simulated:
            await thread.add_message(thread_id, role, content)
            print(f"\n   {role.title()}: {content}")

    print()

    # Show final message count
    info = await thread.get(thread_id)
    print(f"Total messages in thread: {info.message_count}")
    print()

    # Clean up
    await thread.delete(thread_id)


# =============================================================================
# MESSAGE HISTORY LIMITING
# =============================================================================

async def history_limit_demo():
    """Demonstrate limiting message history to manage context."""
    print("=" * 60)
    print("   HISTORY LIMITS - Managing Context Window")
    print("=" * 60)
    print()

    thread = Thread()
    thread_id = await thread.create("Long Conversation")

    # Add many messages
    print("1. Adding 20 messages:")
    print("-" * 40)

    for i in range(10):
        await thread.add_message(thread_id, "user", f"User message {i+1}")
        await thread.add_message(thread_id, "assistant", f"Assistant response {i+1}")

    info = await thread.get(thread_id)
    print(f"   Total messages: {info.message_count}")
    print()

    # Get all messages
    print("2. Get ALL messages:")
    print("-" * 40)
    all_messages = await thread.get_messages(thread_id)
    print(f"   Retrieved: {len(all_messages)} messages")
    print()

    # Get only recent messages
    print("3. Get LAST 6 messages (for context window):")
    print("-" * 40)
    recent = await thread.get_messages(thread_id, limit=6)
    print(f"   Retrieved: {len(recent)} messages")
    for msg in recent:
        print(f"   [{msg['role']}] {msg['content']}")
    print()

    print("Tip: Use limit to fit within LLM context windows!")
    print()

    # Clean up
    await thread.delete(thread_id)


# =============================================================================
# THREAD WITH METADATA
# =============================================================================

async def metadata_demo():
    """Demonstrate thread and message metadata."""
    print("=" * 60)
    print("   METADATA - Enriching Conversations")
    print("=" * 60)
    print()

    thread = Thread()

    # Create thread with metadata
    print("1. Thread with metadata:")
    print("-" * 40)

    thread_id = await thread.create(
        name="Customer Support #42",
        tags=["support", "billing", "priority"],
        metadata={
            "customer_id": "cust_123",
            "subscription": "premium",
            "issue_type": "billing",
            "started_by": "web_chat"
        }
    )

    info = await thread.get(thread_id)
    print(f"   Name: {info.name}")
    print(f"   Tags: {info.tags}")
    print(f"   Metadata: {info.metadata}")
    print()

    # Add messages with metadata
    print("2. Messages with metadata:")
    print("-" * 40)

    msg = await thread.add_message(
        thread_id,
        "user",
        "I have a billing question",
        metadata={
            "source": "web_widget",
            "browser": "Chrome",
            "page": "/account/billing"
        }
    )
    print(f"   Message ID: {msg.id}")
    print(f"   Metadata: {msg.metadata}")
    print()

    # Get messages as Message objects (not dicts)
    print("3. Retrieving with full metadata:")
    print("-" * 40)

    messages = await thread.get_messages(thread_id, as_dicts=False)
    for msg in messages:
        print(f"   ID: {msg.id[:8]}...")
        print(f"   Role: {msg.role}")
        print(f"   Content: {msg.content}")
        print(f"   Created: {msg.created_at}")
        print(f"   Metadata: {msg.metadata}")
    print()

    # Clean up
    await thread.delete(thread_id)


# =============================================================================
# CLEAR MESSAGES
# =============================================================================

async def clear_messages_demo():
    """Demonstrate clearing messages while keeping the thread."""
    print("=" * 60)
    print("   CLEAR MESSAGES - Reset Conversation")
    print("=" * 60)
    print()

    thread = Thread()
    thread_id = await thread.create("Resettable Chat")

    # Add messages
    print("1. Initial conversation:")
    print("-" * 40)

    await thread.add_message(thread_id, "user", "Hello!")
    await thread.add_message(thread_id, "assistant", "Hi there!")
    await thread.add_message(thread_id, "user", "How are you?")

    info = await thread.get(thread_id)
    print(f"   Messages: {info.message_count}")
    print()

    # Clear messages
    print("2. Clearing messages:")
    print("-" * 40)

    await thread.clear_messages(thread_id)
    info = await thread.get(thread_id)
    print(f"   Messages after clear: {info.message_count}")
    print(f"   Thread still exists: {info.name}")
    print()

    # Thread can be reused
    print("3. Thread can be reused:")
    print("-" * 40)

    await thread.add_message(thread_id, "user", "Starting fresh!")
    info = await thread.get(thread_id)
    print(f"   New message count: {info.message_count}")
    print()

    # Clean up
    await thread.delete(thread_id)


# =============================================================================
# DEMO RUNNER
# =============================================================================

async def demo():
    """Run all Thread demonstrations."""
    print()
    print("*" * 60)
    print("*" + " " * 17 + "THREAD PRIMITIVE DEMO" + " " * 17 + "*")
    print("*" * 60)
    print()

    await basic_thread_demo()
    await thread_management_demo()
    await multi_turn_demo()
    await history_limit_demo()
    await metadata_demo()
    await clear_messages_demo()

    print("=" * 60)
    print("   Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
