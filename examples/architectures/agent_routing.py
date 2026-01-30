"""
Pattern 3: Agent Routing
========================

Input classification directing queries to specialized handlers.

This example shows how to BUILD a router by composing primitives:
    - Pipe: Classifier to determine route
    - Pipe: Specialized handlers for each domain

Architecture:
    Query → Pipe.run(classify) → Route to appropriate Pipe → Response

NO wrapper classes - just primitives composed together like Lego blocks.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langpy_sdk import Pipe


# =============================================================================
# PATTERN 3: AGENT ROUTING WITH DIRECT PRIMITIVE COMPOSITION
# =============================================================================

async def agent_routing_pattern():
    """
    Build a routing system by composing Pipe primitives.

    This is NOT a class - it's a demonstration of how primitives combine.
    """
    print("=" * 60)
    print("   PATTERN 3: AGENT ROUTING")
    print("   Direct Primitive Composition")
    print("=" * 60)
    print()

    # =========================================================================
    # STEP 1: Create the primitives (our building blocks)
    # =========================================================================

    # Classifier Pipe - determines which route to take
    classifier = Pipe(model="gpt-4o-mini")

    # Specialized handler Pipes - one for each domain
    tech_handler = Pipe(model="gpt-4o-mini")
    sales_handler = Pipe(model="gpt-4o-mini")
    support_handler = Pipe(model="gpt-4o-mini")

    print("Primitives created:")
    print("  - Classifier Pipe: determines route")
    print("  - Tech Handler Pipe: technical questions")
    print("  - Sales Handler Pipe: sales inquiries")
    print("  - Support Handler Pipe: customer support")
    print()

    # =========================================================================
    # STEP 2: Define the routes
    # =========================================================================

    routes = {
        "technical": {
            "description": "Technical questions, bugs, API issues",
            "handler": tech_handler,
            "system": "You are a technical support specialist. Help with bugs, APIs, and technical issues."
        },
        "sales": {
            "description": "Pricing, plans, features, purchasing",
            "handler": sales_handler,
            "system": "You are a sales representative. Help with pricing, plans, and purchases."
        },
        "support": {
            "description": "Account issues, billing, general help",
            "handler": support_handler,
            "system": "You are a customer support specialist. Help with accounts, billing, and general questions."
        }
    }

    # =========================================================================
    # STEP 3: Route queries using primitive composition
    # =========================================================================

    async def route_query(query: str) -> tuple[str, str]:
        """
        Route a query using primitive composition:
        1. Pipe.run() to classify the query
        2. Pipe.run() with appropriate handler
        """
        # COMPOSE: Classifier determines route
        routes_desc = "\n".join(f"- {name}: {r['description']}" for name, r in routes.items())

        classification = await classifier.run(
            f"""Classify this query into one category. Respond with ONLY the category name.

Categories:
{routes_desc}

Query: {query}

Category:""",
            system="You are a query classifier. Respond with only the category name."
        )

        # Parse the route
        route_name = classification.content.strip().lower()
        selected_route = None
        for name in routes.keys():
            if name in route_name:
                selected_route = name
                break

        if not selected_route:
            selected_route = "support"  # Default

        # COMPOSE: Handler processes the query
        route = routes[selected_route]
        response = await route["handler"].run(query, system=route["system"])

        return selected_route, response.content

    # =========================================================================
    # STEP 4: Test the routing
    # =========================================================================

    queries = [
        "How much does the enterprise plan cost?",
        "I'm getting an error when installing the SDK",
        "I need to reset my password",
        "Can you explain the API rate limits?",
        "I want to cancel my subscription",
    ]

    for query in queries:
        print(f"Query: {query}")
        print("-" * 40)

        route_name, response = await route_query(query)

        print(f"Routed to: {route_name.upper()}")
        print(f"Response: {response[:150]}...")
        print()

    print("=" * 60)


# =============================================================================
# SIMPLE ROUTER FUNCTION - Minimal composition example
# =============================================================================

async def simple_router(query: str, routes: dict[str, str]) -> tuple[str, str]:
    """
    Minimal router - pure primitive composition.

    Args:
        query: User's query
        routes: Dict of {route_name: description}

    Returns:
        Tuple of (route_name, response)
    """
    pipe = Pipe(model="gpt-4o-mini")

    # Classify
    routes_text = "\n".join(f"- {name}: {desc}" for name, desc in routes.items())
    classification = await pipe.run(
        f"Classify into [{', '.join(routes.keys())}]:\n\n{query}\n\nCategory:",
        system="Respond with only the category name."
    )

    route = classification.content.strip().lower()
    for name in routes.keys():
        if name in route:
            route = name
            break

    # Handle
    response = await pipe.run(query, system=f"You are a {route} specialist.")

    return route, response.content


# =============================================================================
# CUSTOMER SERVICE ROUTING
# =============================================================================

async def customer_service_demo():
    """
    Customer service routing with specialized handlers.
    Direct primitive composition.
    """
    print("\n" + "=" * 60)
    print("   CUSTOMER SERVICE ROUTING")
    print("   Classifier + Specialized Handlers")
    print("=" * 60 + "\n")

    # Create primitives
    classifier = Pipe(model="gpt-4o-mini")
    handlers = {
        "billing": Pipe(model="gpt-4o-mini"),
        "technical": Pipe(model="gpt-4o-mini"),
        "general": Pipe(model="gpt-4o-mini"),
    }

    queries = [
        "Why was I charged twice this month?",
        "The app keeps crashing on startup",
        "What are your business hours?",
    ]

    for query in queries:
        # COMPOSE: Classify
        result = await classifier.run(
            f"Classify as billing, technical, or general:\n\n{query}\n\nCategory:",
            system="Respond with only: billing, technical, or general"
        )

        route = "general"
        for name in handlers.keys():
            if name in result.content.lower():
                route = name
                break

        # COMPOSE: Handle with specialized pipe
        systems = {
            "billing": "You handle billing inquiries. Be helpful about charges and payments.",
            "technical": "You handle technical issues. Provide troubleshooting steps.",
            "general": "You handle general inquiries. Be friendly and informative.",
        }

        response = await handlers[route].run(query, system=systems[route])

        print(f"Query: {query}")
        print(f"Route: {route}")
        print(f"Response: {response.content[:100]}...")
        print()


# =============================================================================
# MULTI-DOMAIN Q&A ROUTING
# =============================================================================

async def multi_domain_demo():
    """
    Multi-domain Q&A with topic-specific handlers.
    Direct primitive composition.
    """
    print("\n" + "=" * 60)
    print("   MULTI-DOMAIN Q&A ROUTING")
    print("   Topic Classification + Expert Handlers")
    print("=" * 60 + "\n")

    pipe = Pipe(model="gpt-4o-mini")

    # Define domains with their system prompts
    domains = {
        "python": "You are a Python expert. Provide accurate Python advice.",
        "javascript": "You are a JavaScript expert. Provide accurate JS advice.",
        "database": "You are a database expert. Help with SQL and DB design.",
    }

    questions = [
        "How do I create a list comprehension?",
        "What's the difference between let and const?",
        "How do I optimize a slow SQL query?",
    ]

    for question in questions:
        # COMPOSE: Classify domain
        classification = await pipe.run(
            f"Classify as python, javascript, or database:\n\n{question}\n\nDomain:",
            system="Respond with only: python, javascript, or database"
        )

        domain = "python"  # Default
        for d in domains.keys():
            if d in classification.content.lower():
                domain = d
                break

        # COMPOSE: Answer with domain expertise
        answer = await pipe.run(question, system=domains[domain])

        print(f"Q: {question}")
        print(f"Domain: {domain}")
        print(f"A: {answer.content[:150]}...")
        print()


# =============================================================================
# DEMO
# =============================================================================

async def demo():
    """Run all agent routing demonstrations."""
    await agent_routing_pattern()
    await customer_service_demo()
    await multi_domain_demo()


if __name__ == "__main__":
    asyncio.run(demo())
