"""
AI Agency with Workflow and Memory - Demo
==========================================
"""

import asyncio
import os
from typing import Dict
from dotenv import load_dotenv
from langpy import Langpy

load_dotenv()


class AIAgency:
    """AI Agency using Workflow + Memory + Multiple Agents."""

    def __init__(self, api_key: str):
        print("=" * 70)
        print("AI AGENCY - WORKFLOW + MEMORY SYSTEM")
        print("=" * 70)

        # Initialize unified Langpy client
        self.lb = Langpy(api_key=api_key)
        print("[OK] Langpy client initialized")
        print("     All primitives ready: agent, memory, workflow, thread")
        print()

        # Agent personas
        self.ceo_instructions = """You are the CEO of an AI consulting firm.
Break down projects into clear tasks for: researcher, writer, reviewer.
Be strategic and provide specific, actionable instructions."""

        self.researcher_instructions = """You are a research specialist.
Gather information, analyze data, provide detailed findings with sources."""

        self.writer_instructions = """You are a content writer.
Create clear, engaging, well-structured content based on research."""

        self.reviewer_instructions = """You are a QA specialist.
Evaluate quality objectively, provide ratings (1-10) and constructive feedback."""

        # Storage
        self.ceo_thread_id = None
        self.employee_threads = {}

    async def setup_memory(self):
        """Setup company knowledge base using Memory primitive."""
        print("[PRIMITIVE: Memory] Setting up knowledge base...")

        # Company knowledge documents
        company_knowledge = [
            {"content": "LangPy is a Python framework with 9 composable AI primitives."},
            {"content": "The primitives: Agent, Pipe, Memory, Thread, Workflow, Parser, Chunker, Embed, Tools."},
            {"content": "Agent provides unified LLM API for 100+ models (OpenAI, Anthropic, etc)."},
            {"content": "Memory enables vector storage and RAG (Retrieval-Augmented Generation)."},
            {"content": "Workflow orchestrates multi-step AI systems with dependency management."},
            {"content": "Thread manages conversation history with persistence."},
            {"content": "Pipe provides single LLM calls with templating."},
            {"content": "LangPy primitives are composable - combine them like Lego blocks."},
        ]

        # Add to memory
        response = await self.lb.memory.add(documents=company_knowledge)

        if response.success:
            print(f"      [OK] Added {response.count} documents to memory")
            print(f"      [OK] Chunker & Embed used internally")
        else:
            print(f"      [ERROR] {response.error}")
        print()

    async def setup_threads(self):
        """Setup conversation threads for each agent."""
        print("[PRIMITIVE: Thread] Setting up conversation tracking...")

        # Create thread for CEO
        ceo_thread = await self.lb.thread.create(metadata={"role": "ceo"})
        if ceo_thread.success:
            self.ceo_thread_id = ceo_thread.thread_id
            print(f"      [OK] CEO thread: {self.ceo_thread_id}")

        # Create threads for employees
        for role in ["researcher", "writer", "reviewer"]:
            thread = await self.lb.thread.create(metadata={"role": role})
            if thread.success:
                self.employee_threads[role] = thread.thread_id
                print(f"      [OK] {role.capitalize()} thread: {thread.thread_id}")
        print()

    async def execute_project(self, project_description: str):
        """Execute project using Workflow primitive."""
        print("=" * 70)
        print("[PRIMITIVE: Workflow] Multi-Agent Orchestration")
        print("=" * 70)
        print(f"[PROJECT] {project_description}\n")

        wf = self.lb.workflow(name="agency-project", debug=True)

        # STEP 1: CEO Planning
        async def ceo_planning(ctx):
            print("\n[STEP 1/5] CEO Planning...")
            print("-" * 70)

            project = ctx.get("project_description")

            planning_prompt = f"""As CEO, analyze this project and create a detailed plan:

PROJECT: {project}

Provide specific instructions for each team member:
RESEARCH TASK: [what the researcher should investigate]
WRITING TASK: [what the writer should create]
REVIEW TASK: [what the reviewer should evaluate]"""

            response = await self.lb.agent.run(
                model="openai:gpt-4o-mini",
                input=planning_prompt,
                instructions=self.ceo_instructions,
                temperature=0.7
            )

            if response.success:
                if self.ceo_thread_id:
                    try:
                        await self.lb.thread.append(
                            thread_id=self.ceo_thread_id,
                            messages=[
                                {"role": "user", "content": project},
                                {"role": "assistant", "content": response.output}
                            ]
                        )
                    except Exception as e:
                        print(f"      [Warning] Thread save failed: {e}")

                print(f"[CEO] [OK] Plan created: {response.output[:150]}...")

                tasks = self._parse_plan(response.output)
                return {"plan": response.output, "tasks": tasks}
            else:
                raise Exception(f"CEO planning failed: {response.error}")

        wf.step(id="ceo_plan", run=ceo_planning)

        # STEP 2: Research with Memory RAG
        async def research_phase(ctx):
            print("\n[STEP 2/5] Research Phase (with Memory RAG)...")
            print("-" * 70)

            tasks = ctx.get("tasks", {})
            research_task = tasks.get("research", "Research the topic")

            # RAG: Retrieve from memory
            print("      [RAG] Querying memory for relevant context...")

            memory_response = await self.lb.memory.retrieve(
                query=ctx.get("project_description"),
                top_k=3,
                min_score=0.5
            )

            context = ""
            if memory_response.success and memory_response.documents:
                context = "\n".join([
                    f"- {doc.get('content', '')}"
                    for doc in memory_response.documents
                ])
                print(f"      [RAG] [OK] Retrieved {len(memory_response.documents)} relevant docs")
            else:
                print(f"      [RAG] [X] No context found")

            research_prompt = f"""Task: {research_task}

Relevant context from knowledge base:
{context}

Provide detailed research findings with specific facts and data."""

            response = await self.lb.agent.run(
                model="openai:gpt-4o-mini",
                input=research_prompt,
                instructions=self.researcher_instructions,
                temperature=0.3
            )

            if response.success:
                if self.employee_threads.get("researcher"):
                    try:
                        await self.lb.thread.append(
                            thread_id=self.employee_threads["researcher"],
                            messages=[
                                {"role": "user", "content": research_task},
                                {"role": "assistant", "content": response.output}
                            ]
                        )
                    except Exception as e:
                        print(f"      [Warning] Thread save failed: {e}")

                print(f"[RESEARCHER] [OK] Research completed: {response.output[:150]}...")
                return {"research": response.output}
            else:
                raise Exception(f"Research failed: {response.error}")

        wf.step(id="research", run=research_phase, after=["ceo_plan"])

        # STEP 3: Writing
        async def writing_phase(ctx):
            print("\n[STEP 3/5] Writing Phase...")
            print("-" * 70)

            tasks = ctx.get("tasks", {})
            writing_task = tasks.get("writing", "Create content")
            research = ctx.get("research", "")

            writing_prompt = f"""Task: {writing_task}

Research findings to base content on:
{research[:800]}...

Create well-structured, engaging content."""

            response = await self.lb.pipe.run(
                input=writing_prompt,
                instructions=self.writer_instructions,
                model="openai:gpt-4o-mini",
                temperature=0.7
            )

            if response.success:
                if self.employee_threads.get("writer"):
                    try:
                        await self.lb.thread.append(
                            thread_id=self.employee_threads["writer"],
                            messages=[
                                {"role": "user", "content": writing_task},
                                {"role": "assistant", "content": response.output}
                            ]
                        )
                    except Exception as e:
                        print(f"      [Warning] Thread save failed: {e}")

                print(f"[WRITER] [OK] Content created: {response.output[:150]}...")
                return {"writing": response.output}
            else:
                raise Exception(f"Writing failed: {response.error}")

        wf.step(id="write", run=writing_phase, after=["research"])

        # STEP 4: Review
        async def review_phase(ctx):
            print("\n[STEP 4/5] Review Phase...")
            print("-" * 70)

            tasks = ctx.get("tasks", {})
            review_task = tasks.get("review", "Review content")
            writing = ctx.get("writing", "")

            review_prompt = f"""Task: {review_task}

Content to review:
{writing}

Provide:
1. Overall Quality Rating (1-10)
2. Strengths (3-5 points)
3. Areas for Improvement (2-3 points)
4. Recommendation: APPROVE / MINOR_CHANGES / MAJOR_REVISION"""

            response = await self.lb.agent.run(
                model="openai:gpt-4o-mini",
                input=review_prompt,
                instructions=self.reviewer_instructions,
                temperature=0.5
            )

            if response.success:
                if self.employee_threads.get("reviewer"):
                    try:
                        await self.lb.thread.append(
                            thread_id=self.employee_threads["reviewer"],
                            messages=[
                                {"role": "user", "content": review_task},
                                {"role": "assistant", "content": response.output}
                            ]
                        )
                    except Exception as e:
                        print(f"      [Warning] Thread save failed: {e}")

                print(f"[REVIEWER] [OK] Review completed: {response.output[:150]}...")
                return {"review": response.output}
            else:
                raise Exception(f"Review failed: {response.error}")

        wf.step(id="review", run=review_phase, after=["write"])

        # STEP 5: CEO Decision
        async def ceo_decision(ctx):
            print("\n[STEP 5/5] CEO Final Decision...")
            print("-" * 70)

            research = ctx.get("research", "")
            writing = ctx.get("writing", "")
            review = ctx.get("review", "")

            decision_prompt = f"""Review all work and make final decision:

RESEARCH SUMMARY: {research[:300]}...
CONTENT SUMMARY: {writing[:300]}...
REVIEWER FEEDBACK: {review}

Provide:
1. FINAL DECISION: APPROVED / NEEDS_REVISION / REJECTED
2. Overall Project Assessment
3. Key Learnings for Future Projects"""

            response = await self.lb.agent.run(
                model="openai:gpt-4o-mini",
                input=decision_prompt,
                instructions=self.ceo_instructions,
                temperature=0.6
            )

            if response.success:
                if self.ceo_thread_id:
                    try:
                        await self.lb.thread.append(
                            thread_id=self.ceo_thread_id,
                            messages=[
                                {"role": "user", "content": "Review and decide"},
                                {"role": "assistant", "content": response.output}
                            ]
                        )
                    except Exception as e:
                        print(f"      [Warning] Thread save failed: {e}")

                print(f"[CEO] [OK] Decision: {response.output[:150]}...")

                # Update memory with learnings
                learning_doc = {
                    "content": f"Project: {ctx.get('project_description')[:100]}. Learning: {response.output[:200]}"
                }
                await self.lb.memory.add(documents=[learning_doc])
                print("      [OK] Learnings saved to memory for future projects")

                return {"decision": response.output}
            else:
                raise Exception(f"CEO decision failed: {response.error}")

        wf.step(id="decide", run=ceo_decision, after=["review"])

        # Execute workflow
        print("\n[WORKFLOW] Executing 5-step workflow...")
        print("           Order: ceo_plan -> research -> write -> review -> decide")
        print()

        result = await wf.run(project_description=project_description)

        if result.success:
            print("\n" + "=" * 70)
            print("[SUCCESS] Workflow Completed!")
            print("=" * 70)
            print(f"Status: {result.status}")
            print(f"Steps executed: {len(result.steps)}")
            print()

            print("Execution Times:")
            for step in result.steps:
                status = "[OK]" if step["success"] else "[X]"
                print(f"  [{status}] {step['id']}: {step['duration_ms']:.0f}ms")
            print()

            return result.outputs
        else:
            print(f"\n[FAILED] {result.error}")
            return {}

    def _parse_plan(self, plan: str) -> Dict[str, str]:
        """Parse CEO plan into structured tasks."""
        tasks = {}

        if "RESEARCH TASK:" in plan:
            tasks['research'] = plan.split("RESEARCH TASK:")[1].split("WRITING")[0].strip()
        if "WRITING TASK:" in plan:
            tasks['writing'] = plan.split("WRITING TASK:")[1].split("REVIEW")[0].strip()
        if "REVIEW TASK:" in plan:
            tasks['review'] = plan.split("REVIEW TASK:")[1].strip()

        if not tasks:
            tasks = {
                'research': 'Research the topic thoroughly',
                'writing': 'Create comprehensive content',
                'review': 'Review for quality and accuracy'
            }

        return tasks

    async def generate_summary(self, results: Dict):
        """Generate project summary."""
        print("=" * 70)
        print("PROJECT SUMMARY")
        print("=" * 70)
        print()

        print("[CEO] Conversation History:")
        if self.ceo_thread_id:
            thread = await self.lb.thread.list(thread_id=self.ceo_thread_id)
            if thread.success and thread.messages:
                for msg in thread.messages[-4:]:
                    role = msg.get('role', '').upper()
                    content = msg.get('content', '')
                    print(f"  [{role}]: {content[:100]}...")
        print()

        print("[TEAM] Activity Summary:")
        for role, thread_id in self.employee_threads.items():
            thread = await self.lb.thread.list(thread_id=thread_id)
            if thread.success and thread.messages:
                print(f"  {role.capitalize()}: {len(thread.messages)} messages")
        print()


async def main():
    """Run the AI Agency demo."""

    print("\n" + "=" * 70)
    print("  AI AGENCY WITH WORKFLOW + MEMORY")
    print("  Multi-Agent System with RAG")
    print("=" * 70)
    print()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] Set OPENAI_API_KEY in .env file")
        return

    # Initialize agency
    agency = AIAgency(api_key=api_key)

    # Setup primitives
    print("[SETUP] Initializing AI Agency...")
    print("=" * 70)
    await agency.setup_memory()
    await agency.setup_threads()

    # Define project
    project = """
    Create a guide on "Getting Started with LangPy" covering:
    1. What LangPy is and its benefits
    2. The 9 core primitives overview
    3. A simple code example
    4. Best practices for developers

    Target: Python developers new to AI
    Length: 500-700 words
    """

    # Execute project
    results = await agency.execute_project(project)

    # Show summary
    await agency.generate_summary(results)

    # Display final deliverable
    print("=" * 70)
    print("FINAL DELIVERABLE")
    print("=" * 70)

    writing_output = results.get('write', {})
    content = writing_output.get('writing', 'No content generated')

    print(content)
    print()

    decision_output = results.get('decide', {})
    decision = decision_output.get('decision', '')
    print("-" * 70)
    print(f"CEO DECISION:\n{decision[:300]}...")
    print()

    print("=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
