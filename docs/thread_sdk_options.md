# Thread SDK Options - Complete Reference

This document provides a comprehensive reference for all available options when working with the Thread SDK in LangPy.

## Table of Contents

1. [Thread Overview](#thread-overview)
2. [Thread Creation](#thread-creation)
3. [Thread Configuration](#thread-configuration)
4. [Thread Operations](#thread-operations)
5. [Message Management](#message-management)
6. [Advanced Features](#advanced-features)
7. [Error Handling](#error-handling)
8. [Examples](#examples)

## Thread Overview

The Thread SDK provides conversation thread management similar to Langbase, enabling:

- **Conversation State Management**: Persistent conversation threads
- **Message Storage**: Automatic message persistence to disk
- **Thread Organization**: Tagging, archiving, and status management
- **Advanced Filtering**: Message and thread filtering capabilities
- **Search Functionality**: Search across messages and threads

## Thread Creation

### Factory Method (Recommended)

```python
from sdk import thread

# Create a thread interface
thread_interface = thread()

# Create a thread interface with custom storage path
thread_interface = thread(storage_path="/path/to/threads")
```

### Direct Creation Methods

```python
# Using ThreadInterface directly
from sdk.thread_interface import ThreadInterface

thread_interface = ThreadInterface(
    async_backend=None,
    sync_backend=None,
    storage_path="/path/to/threads"
)

# Using AsyncThread directly
from thread import AsyncThread

async_thread = AsyncThread(storage_path="/path/to/threads")

# Using SyncThread directly
from thread import SyncThread

sync_thread = SyncThread(storage_path="/path/to/threads")
```

## Thread Configuration

### ThreadInterface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `async_backend` | `callable` | `None` | Optional async backend callable |
| `sync_backend` | `callable` | `None` | Optional sync backend callable |
| `storage_path` | `str` | `None` | Storage path for threads (defaults to `~/.langpy/threads`) |

### Thread Model

```python
class Thread:
    id: str                          # Unique thread identifier
    name: Optional[str]              # Optional thread name
    messages: List[ThreadMessage]    # List of messages in the thread
    metadata: Dict[str, Any]         # Thread metadata
    created_at: int                  # Creation timestamp
    updated_at: int                  # Last update timestamp
    status: str                      # Thread status: "active", "archived", "deleted"
    tags: List[str]                  # Thread tags for organization
```

### ThreadMessage Model

```python
class ThreadMessage:
    id: str                          # Unique message identifier
    role: str                        # Message role: "user", "assistant", "system", "tool"
    content: Union[str, Dict]        # Message content
    name: Optional[str]              # Optional message name
    tool_call_id: Optional[str]      # Tool call ID for tool messages
    created_at: int                  # Creation timestamp
    metadata: Dict[str, Any]         # Message metadata
```

## Thread Operations

### Create Thread

```python
# Create a new thread
thread_obj = await thread_interface.create_thread(
    name="Customer Support Chat",           # Optional thread name
    metadata={"customer_id": "12345"}       # Optional metadata
)

# Create thread with tags
thread_obj = await thread_interface.create_thread(
    name="Technical Discussion",
    metadata={"priority": "high"},
    tags=["technical", "urgent"]
)
```

### Get Thread

```python
# Get thread by ID
thread_obj = await thread_interface.get_thread(thread_id="thread_123")

if thread_obj:
    print(f"Thread: {thread_obj.name}")
    print(f"Messages: {len(thread_obj.messages)}")
    print(f"Status: {thread_obj.status}")
```

### List Threads

```python
# List all threads
all_threads = await thread_interface.list_threads()

# List threads with filtering
active_threads = await thread_interface.list_threads(
    status="active",
    tags=["support"],
    limit=10
)

# List archived threads
archived_threads = await thread_interface.list_threads(status="archived")
```

### Delete Thread

```python
# Delete a thread (marks as deleted)
deleted = await thread_interface.delete_thread(thread_id="thread_123")

if deleted:
    print("Thread deleted successfully")
```

### Update Thread Metadata

```python
# Update thread metadata
updated_thread = await thread_interface.update_thread_metadata(
    thread_id="thread_123",
    metadata={"priority": "high", "category": "technical"}
)

if updated_thread:
    print(f"Updated metadata: {updated_thread.metadata}")
```

## Message Management

### Add Message

```python
# Add user message
user_message = await thread_interface.add_message(
    thread_id="thread_123",
    role="user",
    content="Hello, I need help with my account"
)

# Add assistant message
assistant_message = await thread_interface.add_message(
    thread_id="thread_123",
    role="assistant",
    content="I'd be happy to help you with your account. What specific issue are you experiencing?"
)

# Add system message
system_message = await thread_interface.add_message(
    thread_id="thread_123",
    role="system",
    content="Customer priority level: VIP"
)

# Add tool message
tool_message = await thread_interface.add_message(
    thread_id="thread_123",
    role="tool",
    content="Account status: Active, Last login: 2024-01-15",
    name="account_lookup",
    tool_call_id="call_123"
)

# Add message with metadata
message_with_meta = await thread_interface.add_message(
    thread_id="thread_123",
    role="user",
    content="This is important",
    metadata={"priority": "high", "source": "mobile_app"}
)
```

### Get Messages

```python
# Get all messages from a thread
all_messages = await thread_interface.get_messages(thread_id="thread_123")

# Get limited number of messages
recent_messages = await thread_interface.get_messages(
    thread_id="thread_123",
    limit=10
)

# Get messages before a specific message
messages_before = await thread_interface.get_messages(
    thread_id="thread_123",
    before="message_456"
)

# Get messages after a specific message
messages_after = await thread_interface.get_messages(
    thread_id="thread_123",
    after="message_123"
)

# Get messages by role
user_messages = await thread_interface.get_messages(
    thread_id="thread_123",
    role="user"
)
```

## Advanced Features

### Thread Archiving

```python
# Archive a thread
archived = await thread_interface.archive_thread(thread_id="thread_123")

if archived:
    print("Thread archived successfully")

# Restore an archived thread
restored = await thread_interface.restore_thread(thread_id="thread_123")

if restored:
    print("Thread restored successfully")
```

### Thread Tagging

```python
# Add tags to a thread
tagged_thread = await thread_interface.add_thread_tags(
    thread_id="thread_123",
    tags=["support", "billing", "urgent"]
)

# Remove tags from a thread
untagged_thread = await thread_interface.remove_thread_tags(
    thread_id="thread_123",
    tags=["urgent"]
)
```

### Conversation Summary

```python
# Get conversation summary
summary = await thread_interface.get_conversation_summary(thread_id="thread_123")

print(f"Total messages: {summary['total_messages']}")
print(f"User messages: {summary['user_messages']}")
print(f"Assistant messages: {summary['assistant_messages']}")
print(f"Status: {summary['status']}")
print(f"Tags: {summary['tags']}")
```

### Message Search

```python
# Search messages across all threads
results = await thread_interface.search_messages(
    query="password reset",
    limit=10
)

# Search messages in specific threads
results = await thread_interface.search_messages(
    query="billing issue",
    thread_ids=["thread_123", "thread_456"],
    limit=5
)

for result in results:
    print(f"Thread: {result['thread_name']}")
    print(f"Message: {result['content']}")
    print(f"Role: {result['role']}")
    print("---")
```

## Error Handling

### Exception Handling

```python
# Handle thread not found
try:
    message = await thread_interface.add_message(
        thread_id="nonexistent_thread",
        role="user",
        content="Hello"
    )
except ValueError as e:
    print(f"Error: {e}")  # Thread not found

# Handle missing thread
thread_obj = await thread_interface.get_thread("nonexistent_thread")
if thread_obj is None:
    print("Thread not found")
```

### Safe Operations

```python
# Safe thread operations
async def safe_add_message(thread_id: str, role: str, content: str):
    try:
        # Check if thread exists
        thread_obj = await thread_interface.get_thread(thread_id)
        if not thread_obj:
            print(f"Thread {thread_id} not found")
            return None
        
        # Add message
        message = await thread_interface.add_message(
            thread_id=thread_id,
            role=role,
            content=content
        )
        return message
    except Exception as e:
        print(f"Error adding message: {e}")
        return None
```

## Examples

### Basic Thread Usage

```python
from sdk import thread
import asyncio

async def basic_thread_example():
    # Create thread interface
    thread_interface = thread()
    
    # Create a new thread
    new_thread = await thread_interface.create_thread(
        name="Customer Support",
        metadata={"customer_id": "12345", "priority": "medium"}
    )
    
    print(f"Created thread: {new_thread.id}")
    
    # Add messages to the thread
    user_msg = await thread_interface.add_message(
        thread_id=new_thread.id,
        role="user",
        content="I'm having trouble logging into my account"
    )
    
    assistant_msg = await thread_interface.add_message(
        thread_id=new_thread.id,
        role="assistant",
        content="I can help you with that. Let me check your account status."
    )
    
    # Get all messages
    messages = await thread_interface.get_messages(thread_id=new_thread.id)
    
    print(f"Thread has {len(messages)} messages:")
    for msg in messages:
        print(f"  {msg.role}: {msg.content}")

asyncio.run(basic_thread_example())
```

### Multi-Turn Conversation

```python
from sdk import thread
import asyncio

async def conversation_example():
    thread_interface = thread()
    
    # Create conversation thread
    conversation = await thread_interface.create_thread(
        name="Technical Support",
        tags=["technical", "support"]
    )
    
    # Simulate a multi-turn conversation
    turns = [
        ("user", "My application keeps crashing when I try to save files"),
        ("assistant", "I'm sorry to hear that. Can you tell me which version of the application you're using?"),
        ("user", "I'm using version 2.1.3"),
        ("assistant", "Thank you. This is a known issue in version 2.1.3. Let me walk you through the solution."),
        ("user", "Great, I'm ready to follow the steps"),
        ("assistant", "First, please close the application completely and then restart it in safe mode.")
    ]
    
    # Add each turn to the conversation
    for role, content in turns:
        await thread_interface.add_message(
            thread_id=conversation.id,
            role=role,
            content=content
        )
    
    # Get conversation summary
    summary = await thread_interface.get_conversation_summary(conversation.id)
    
    print(f"Conversation Summary:")
    print(f"  Total messages: {summary['total_messages']}")
    print(f"  User messages: {summary['user_messages']}")
    print(f"  Assistant messages: {summary['assistant_messages']}")
    print(f"  Status: {summary['status']}")
    print(f"  Tags: {summary['tags']}")

asyncio.run(conversation_example())
```

### Thread Management System

```python
from sdk import thread
import asyncio

async def thread_management_example():
    thread_interface = thread()
    
    # Create multiple threads with different categories
    threads_data = [
        {"name": "Billing Inquiry", "tags": ["billing", "inquiry"], "metadata": {"priority": "high"}},
        {"name": "Technical Support", "tags": ["technical", "support"], "metadata": {"priority": "medium"}},
        {"name": "Feature Request", "tags": ["feature", "request"], "metadata": {"priority": "low"}},
        {"name": "Bug Report", "tags": ["bug", "report"], "metadata": {"priority": "high"}}
    ]
    
    created_threads = []
    
    # Create threads
    for thread_data in threads_data:
        thread_obj = await thread_interface.create_thread(
            name=thread_data["name"],
            tags=thread_data["tags"],
            metadata=thread_data["metadata"]
        )
        created_threads.append(thread_obj)
        print(f"Created thread: {thread_obj.name}")
    
    # Add some messages to each thread
    for thread_obj in created_threads:
        await thread_interface.add_message(
            thread_id=thread_obj.id,
            role="user",
            content=f"This is a message for {thread_obj.name}"
        )
        
        await thread_interface.add_message(
            thread_id=thread_obj.id,
            role="assistant",
            content="Thank you for contacting us. We'll help you with this issue."
        )
    
    # List threads by category
    print("\nHigh Priority Threads:")
    high_priority = await thread_interface.list_threads(
        tags=["billing", "bug"],
        limit=10
    )
    
    for thread_obj in high_priority:
        if thread_obj.metadata.get("priority") == "high":
            print(f"  {thread_obj.name} - {thread_obj.tags}")
    
    # Archive completed threads
    billing_threads = [t for t in created_threads if "billing" in t.tags]
    for thread_obj in billing_threads:
        await thread_interface.archive_thread(thread_obj.id)
        print(f"Archived thread: {thread_obj.name}")
    
    # List active threads
    print("\nActive Threads:")
    active_threads = await thread_interface.list_threads(status="active")
    for thread_obj in active_threads:
        print(f"  {thread_obj.name} - {thread_obj.status}")

asyncio.run(thread_management_example())
```

### Message Search and Filtering

```python
from sdk import thread
import asyncio

async def search_and_filter_example():
    thread_interface = thread()
    
    # Create a test thread
    test_thread = await thread_interface.create_thread(
        name="Support Chat",
        tags=["support", "chat"]
    )
    
    # Add various messages
    messages_data = [
        ("user", "I need help with password reset"),
        ("assistant", "I can help you reset your password. Please provide your email address."),
        ("user", "my email is user@example.com"),
        ("assistant", "Thank you. I've sent a password reset link to your email."),
        ("user", "I didn't receive the email"),
        ("assistant", "Let me check your account status and resend the email."),
        ("system", "Password reset email sent successfully"),
        ("user", "Got it! The password reset worked. Thank you!")
    ]
    
    for role, content in messages_data:
        await thread_interface.add_message(
            thread_id=test_thread.id,
            role=role,
            content=content
        )
    
    # Search for messages containing "password"
    password_messages = await thread_interface.search_messages(
        query="password",
        thread_ids=[test_thread.id]
    )
    
    print("Messages containing 'password':")
    for msg in password_messages:
        print(f"  {msg['role']}: {msg['content']}")
    
    # Get only user messages
    user_messages = await thread_interface.get_messages(
        thread_id=test_thread.id,
        role="user"
    )
    
    print(f"\nUser messages ({len(user_messages)}):")
    for msg in user_messages:
        print(f"  {msg.content}")
    
    # Get only assistant messages
    assistant_messages = await thread_interface.get_messages(
        thread_id=test_thread.id,
        role="assistant"
    )
    
    print(f"\nAssistant messages ({len(assistant_messages)}):")
    for msg in assistant_messages:
        print(f"  {msg.content}")
    
    # Get last 3 messages
    recent_messages = await thread_interface.get_messages(
        thread_id=test_thread.id,
        limit=3
    )
    
    print(f"\nLast 3 messages:")
    for msg in recent_messages:
        print(f"  {msg.role}: {msg.content}")

asyncio.run(search_and_filter_example())
```

### Advanced Thread Features

```python
from sdk import thread
import asyncio

async def advanced_features_example():
    thread_interface = thread()
    
    # Create a thread with rich metadata
    advanced_thread = await thread_interface.create_thread(
        name="Premium Support Case",
        metadata={
            "customer_id": "PREM_12345",
            "customer_tier": "premium",
            "issue_type": "technical",
            "severity": "high",
            "assigned_agent": "agent_007",
            "created_by": "system",
            "department": "technical_support"
        },
        tags=["premium", "technical", "high-severity"]
    )
    
    # Add messages with metadata
    await thread_interface.add_message(
        thread_id=advanced_thread.id,
        role="user",
        content="Critical issue: Database connection keeps failing",
        metadata={
            "source": "mobile_app",
            "user_agent": "iOS 15.0",
            "timestamp": "2024-01-15T10:30:00Z"
        }
    )
    
    await thread_interface.add_message(
        thread_id=advanced_thread.id,
        role="assistant",
        content="I understand this is critical. Let me escalate this to our database team immediately.",
        metadata={
            "agent_id": "agent_007",
            "escalated": True,
            "escalation_level": "L2"
        }
    )
    
    # Update thread metadata as case progresses
    await thread_interface.update_thread_metadata(
        thread_id=advanced_thread.id,
        metadata={
            "status": "in_progress",
            "escalation_level": "L2",
            "estimated_resolution": "2024-01-15T16:00:00Z"
        }
    )
    
    # Add additional tags
    await thread_interface.add_thread_tags(
        thread_id=advanced_thread.id,
        tags=["escalated", "database", "critical"]
    )
    
    # Get comprehensive thread summary
    summary = await thread_interface.get_conversation_summary(advanced_thread.id)
    
    print("Advanced Thread Summary:")
    print(f"  Thread ID: {summary['thread_id']}")
    print(f"  Name: {summary['thread_name']}")
    print(f"  Status: {summary['status']}")
    print(f"  Tags: {summary['tags']}")
    print(f"  Total Messages: {summary['total_messages']}")
    print(f"  Metadata: {summary['metadata']}")
    
    # Search for high-severity threads
    high_severity_messages = await thread_interface.search_messages(
        query="critical",
        limit=5
    )
    
    print(f"\nHigh-severity messages found: {len(high_severity_messages)}")
    for msg in high_severity_messages:
        print(f"  Thread: {msg['thread_name']}")
        print(f"  Content: {msg['content']}")
        print("  ---")

asyncio.run(advanced_features_example())
```

### Synchronous Thread Operations

```python
from sdk import thread

def sync_thread_example():
    # Create thread interface
    thread_interface = thread()
    
    # Create thread synchronously
    new_thread = thread_interface.create_thread_sync(
        name="Sync Thread",
        metadata={"created_by": "sync_api"}
    )
    
    print(f"Created sync thread: {new_thread.id}")
    
    # Add messages synchronously
    user_msg = thread_interface.add_message_sync(
        thread_id=new_thread.id,
        role="user",
        content="This is a synchronous message"
    )
    
    assistant_msg = thread_interface.add_message_sync(
        thread_id=new_thread.id,
        role="assistant",
        content="Received your message via synchronous API"
    )
    
    # Get messages synchronously
    messages = thread_interface.get_messages_sync(thread_id=new_thread.id)
    
    print(f"Thread has {len(messages)} messages:")
    for msg in messages:
        print(f"  {msg.role}: {msg.content}")
    
    # Update metadata synchronously
    updated_thread = thread_interface.update_thread_metadata_sync(
        thread_id=new_thread.id,
        metadata={"updated_by": "sync_api", "processed": True}
    )
    
    print(f"Updated metadata: {updated_thread.metadata}")

sync_thread_example()
```

### Thread Backup and Export

```python
from sdk import thread
import asyncio
import json

async def backup_and_export_example():
    thread_interface = thread()
    
    # Get all threads
    all_threads = await thread_interface.list_threads()
    
    # Create backup data structure
    backup_data = {
        "export_timestamp": int(time.time()),
        "total_threads": len(all_threads),
        "threads": []
    }
    
    # Export each thread with its messages
    for thread_obj in all_threads:
        messages = await thread_interface.get_messages(thread_id=thread_obj.id)
        
        thread_export = {
            "thread": thread_obj.dict(),
            "messages": [msg.dict() for msg in messages],
            "summary": await thread_interface.get_conversation_summary(thread_obj.id)
        }
        
        backup_data["threads"].append(thread_export)
    
    # Save backup to file
    with open("thread_backup.json", "w") as f:
        json.dump(backup_data, f, indent=2)
    
    print(f"Exported {len(all_threads)} threads to backup file")
    
    # Generate report
    active_count = len([t for t in all_threads if t.status == "active"])
    archived_count = len([t for t in all_threads if t.status == "archived"])
    
    print(f"Export Summary:")
    print(f"  Active threads: {active_count}")
    print(f"  Archived threads: {archived_count}")
    print(f"  Total threads: {len(all_threads)}")

asyncio.run(backup_and_export_example())
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LANGPY_HOME` | Home directory for thread storage | `~/.langpy` |

## Best Practices

1. **Use descriptive thread names**: Make thread names clear and meaningful
2. **Leverage metadata**: Store relevant context in thread and message metadata
3. **Use tags for organization**: Implement a consistent tagging system
4. **Archive completed threads**: Keep active threads list manageable
5. **Search effectively**: Use search functionality to find relevant conversations
6. **Handle errors gracefully**: Implement proper error handling for missing threads
7. **Monitor thread growth**: Regularly clean up old or irrelevant threads
8. **Use appropriate message roles**: Follow consistent role conventions
9. **Implement pagination**: Use limits and filtering for large thread lists
10. **Backup important conversations**: Export critical threads for backup purposes

## Troubleshooting

### Common Issues

1. **Thread not found**: Check thread ID validity and existence
2. **Storage permission errors**: Verify write permissions to storage directory
3. **Message ordering**: Ensure proper message sequencing in conversations
4. **Metadata conflicts**: Check for metadata key collisions
5. **Tag management**: Avoid duplicate tags and maintain consistency
6. **Search performance**: Optimize search queries for large thread counts
7. **Storage space**: Monitor disk usage for thread storage
8. **Concurrent access**: Handle concurrent thread modifications properly
9. **Export/import issues**: Verify JSON format and data integrity
10. **Sync/async mixing**: Use consistent API patterns (sync vs async)

### Performance Optimization

1. **Limit message history**: Use pagination for large conversations
2. **Optimize search queries**: Use specific keywords and filters
3. **Regular cleanup**: Archive or delete old threads
4. **Efficient filtering**: Use appropriate filters to reduce result sets
5. **Batch operations**: Process multiple threads efficiently
6. **Storage optimization**: Use appropriate storage paths and permissions
7. **Memory management**: Monitor memory usage with large thread lists
8. **Index optimization**: Consider indexing for frequently searched fields 