"""
AI AGENCY - Properly Using Workflow Primitive
==============================================

Complete AI agency using ALL 9 LangPy primitives with ACTUAL Workflow orchestration.
This version properly uses the Workflow primitive to orchestrate the multi-agent system.
"""

import asyncio
import os
from typing import Dict
from dotenv import load_dotenv
from langpy import Langpy, Context

# Load .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class AIAgency:
    """AI Agency using Workflow primitive for proper orchestration."""

    def __init__(self, api_key: str):
        print("=" * 70)
        print("AI AGENCY - USING WORKFLOW PRIMITIVE")
        print("=" * 70)

        # Initialize unified Langpy client
        self.lb = Langpy(api_key=api_key)

        print("[OK] Langpy client initialized")
        print("     All 9 primitives ready")
        print()

        # Agent instructions
        self.ceo_instructions = """You are the CEO of LangPy AI Solutions.
Break down projects into clear tasks for: researcher, writer, reviewer.
Provide specific instructions for each role."""

        self.researcher_instructions = """You are Dr. Sarah Chen, research specialist.
Gather information, analyze data, provide clear findings."""

        self.writer_instructions = """You are Marcus Rodriguez, content writer.
Create clear, engaging, well-structured content."""

        self.reviewer_instructions = """You are Dr. Emily Watson, QA specialist.
Evaluate quality, provide constructive feedback with ratings."""

        self.ceo_thread_id = None
        self.employee_threads = {}

    async def setup_memory(self):
        """Setup memory system (Primitives 3, 7, 8)."""
        print("[PRIMITIVES 3,7,8] Memory + Chunker + Embed...")

        company_knowledge = [
            "LangPy is a Python framework with 9 composable AI primitives.",
            "The primitives: Agent, Pipe, Memory, Thread, Workflow, Parser, Chunker, Embed, Tools.",
            "Agent provides unified LLM API for 100+ models.",
            "Memory enables vector storage and RAG.",
            "Workflow orchestrates multi-step AI systems with dependencies.",
        ]

        formatted_docs = [{"content": text} for text in company_knowledge]
        response = await self.lb.memory.add(documents=formatted_docs)

        if response.success:
            print(f"      [OK] Memory initialized")
            print("      [OK] Chunker & Embed used internally")
        else:
            print(f"      [ERROR] Memory: {response.error}")
        print()

    async def setup_threads(self):
        """Setup conversation threads (Primitive 4)."""
        print("[PRIMITIVE 4] Thread - Conversation tracking...")

        # Create threads for each agent
        ceo_response = await self.lb.thread.create(metadata={"role": "ceo"})
        if ceo_response.success:
            self.ceo_thread_id = ceo_response.thread_id
            print(f"      [OK] CEO thread: {self.ceo_thread_id}")

        for role in ["researcher", "writer", "reviewer"]:
            thread_response = await self.lb.thread.create(metadata={"role": role})
            if thread_response.success:
                self.employee_threads[role] = thread_response.thread_id
                print(f"      [OK] {role.capitalize()} thread: {thread_response.thread_id}")

        print()

    async def execute_project_with_workflow(self, project_description: str):
        """
        Execute project using ACTUAL Workflow primitive orchestration.

        This is the correct way - let Workflow handle dependencies, parallel execution,
        and data flow between steps!
        """
        print("=" * 70)
        print("[PRIMITIVE 5] WORKFLOW - Proper Orchestration")
        print("=" * 70)
        print(f"[PROJECT] {project_description}\n")

        # =====================================================================
        # CREATE WORKFLOW with proper step dependencies
        # =====================================================================
        print("[WORKFLOW] Building workflow with 5 dependent steps...")

        wf = self.lb.workflow(name="agency-project", debug=True)

        # STEP 1: CEO Planning (no dependencies)
        async def ceo_planning_step(ctx):
            """CEO analyzes project and creates plan."""
            try:
                print("\n[STEP 1] CEO Planning...")
                print("-" * 70)

                project = ctx.get("project_description")

                planning_prompt = f"""As CEO, analyze this project and create a plan:

Project: {project}

Provide:
RESEARCH TASK: [specific instructions for researcher]
WRITING TASK: [specific instructions for writer]
REVIEW TASK: [specific instructions for reviewer]"""

                response = await self.lb.agent.run(
                    model="openai:gpt-4o-mini",
                    input=planning_prompt,
                    instructions=self.ceo_instructions,
                    temperature=0.7
                )

                print(f"[DEBUG] Response type: {type(response)}")
                print(f"[DEBUG] Response success: {response.success}")
                print(f"[DEBUG] Response error: {response.error}")
                if hasattr(response, 'output'):
                    print(f"[DEBUG] Response output: {response.output[:100] if response.output else 'None'}...")

                if response.success:
                    # Save to thread (if available)
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
                            print(f"[WARNING] Thread save failed: {e}")

                    print(f"[CEO] Plan created: {response.output[:200]}...")

                    # Parse tasks from plan
                    tasks = self._parse_plan(response.output)
                    return {"plan": response.output, "tasks": tasks}
                else:
                    print(f"[ERROR] CEO planning error: {response.error}")
                    raise Exception(f"CEO planning failed: {response.error}")
            except Exception as e:
                print(f"[ERROR] CEO step exception: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                raise

        wf.step(id="ceo_plan", run=ceo_planning_step)

        # STEP 2: Research (depends on ceo_plan)
        async def research_step(ctx):
            """Researcher gathers information."""
            print("\n[STEP 2] Research Phase...")
            print("-" * 70)

            # Get task from previous step
            tasks = ctx.get("tasks", {})
            research_task = tasks.get("research", "Research the topic thoroughly")

            # Query memory for context (Primitive 3 + 8: Memory + Embed)
            memory_response = await self.lb.memory.retrieve(
                query=ctx.get("project_description"),
                top_k=3
            )

            context = ""
            if memory_response.success and memory_response.documents:
                context = "\n".join([f"- {doc.get('content', '')}" for doc in memory_response.documents])

            research_prompt = f"""Task: {research_task}

Context from knowledge base:
{context}

Provide detailed findings."""

            response = await self.lb.agent.run(
                model="openai:gpt-4o-mini",
                input=research_prompt,
                instructions=self.researcher_instructions,
                temperature=0.3
            )

            if response.success:
                # Save to thread (if available)
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
                        print(f"[WARNING] Thread save failed: {e}")

                print(f"[RESEARCH] Completed: {response.output[:200]}...")
                return {"research": response.output}
            else:
                print(f"[ERROR] Research error: {response.error}")
                raise Exception(f"Research failed: {response.error}")

        wf.step(id="research", run=research_step, after=["ceo_plan"])

        # STEP 3: Writing (depends on research) - Uses Pipe primitive
        async def writing_step(ctx):
            """Writer creates content."""
            print("\n[STEP 3] Writing Phase...")
            print("-" * 70)

            tasks = ctx.get("tasks", {})
            writing_task = tasks.get("writing", "Create comprehensive content")
            research = ctx.get("research", "")

            writing_prompt = f"""Task: {writing_task}

Research findings:
{research[:500]}...

Create engaging, well-structured content."""

            # Primitive 2: Pipe for templated calls
            response = await self.lb.pipe.run(
                input=writing_prompt,
                instructions=self.writer_instructions,
                model="openai:gpt-4o-mini",
                temperature=0.7
            )

            if response.success:
                # Save to thread (if available)
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
                        print(f"[WARNING] Thread save failed: {e}")

                print(f"[WRITING] Completed: {response.output[:200]}...")
                return {"writing": response.output}
            else:
                raise Exception(f"Writing failed: {response.error}")

        wf.step(id="write", run=writing_step, after=["research"])

        # STEP 4: Review (depends on write)
        async def review_step(ctx):
            """Reviewer evaluates quality."""
            print("\n[STEP 4] Review Phase...")
            print("-" * 70)

            tasks = ctx.get("tasks", {})
            review_task = tasks.get("review", "Review the content for quality")
            writing = ctx.get("writing", "")

            review_prompt = f"""Task: {review_task}

Content to review:
{writing[:400]}...

Provide:
1. Quality rating (1-10)
2. Strengths
3. Improvements needed"""

            response = await self.lb.agent.run(
                model="openai:gpt-4o-mini",
                input=review_prompt,
                instructions=self.reviewer_instructions,
                temperature=0.5
            )

            if response.success:
                # Save to thread (if available)
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
                        print(f"[WARNING] Thread save failed: {e}")

                print(f"[REVIEW] Completed: {response.output[:200]}...")
                return {"review": response.output}
            else:
                raise Exception(f"Review failed: {response.error}")

        wf.step(id="review", run=review_step, after=["write"])

        # STEP 5: CEO Decision (depends on review)
        async def ceo_decision_step(ctx):
            """CEO makes final decision."""
            print("\n[STEP 5] CEO Final Decision...")
            print("-" * 70)

            research = ctx.get("research", "")
            writing = ctx.get("writing", "")
            review = ctx.get("review", "")

            decision_prompt = f"""Review all work and make final decision:

Research: {research[:200]}...
Content: {writing[:200]}...
Review: {review}

Provide:
1. Decision: APPROVE/REVISE/REJECT
2. Overall assessment
3. Key learnings"""

            response = await self.lb.agent.run(
                model="openai:gpt-4o-mini",
                input=decision_prompt,
                instructions=self.ceo_instructions,
                temperature=0.6
            )

            if response.success:
                # Save to thread (if available)
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
                        print(f"[WARNING] Thread save failed: {e}")

                print(f"[CEO] Decision: {response.output[:200]}...")

                # Update memory with learnings (Primitive 3)
                learnings = [
                    {"content": f"Project: {ctx.get('project_description')[:100]}"},
                    {"content": f"Learning: {response.output[:200]}"}
                ]
                await self.lb.memory.add(documents=learnings)

                return {"decision": response.output}
            else:
                raise Exception(f"CEO decision failed: {response.error}")

        wf.step(id="decide", run=ceo_decision_step, after=["review"])

        # =====================================================================
        # EXECUTE WORKFLOW - Let it handle orchestration!
        # =====================================================================
        print("\n[WORKFLOW] Executing 5 steps with dependency management...")
        print("           Step order: ceo_plan -> research -> write -> review -> decide")
        print()

        workflow_result = await wf.run(project_description=project_description)

        if workflow_result.success:
            print("\n" + "=" * 70)
            print("[WORKFLOW] Execution Complete!")
            print("=" * 70)
            print(f"Status: {workflow_result.status}")
            print(f"Steps executed: {len(workflow_result.steps)}")
            print()

            # Show step durations
            print("Step Execution Times:")
            for step in workflow_result.steps:
                status = "OK" if step["success"] else "FAIL"
                print(f"  [{status}] {step['id']}: {step['duration_ms']:.0f}ms")
            print()

            return workflow_result.outputs
        else:
            print(f"\n[ERROR] Workflow failed: {workflow_result.error}")
            return {}

    def _parse_plan(self, plan: str) -> Dict[str, str]:
        """Parse CEO plan into tasks."""
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

        print("[CEO] Thread Messages:")
        thread_response = await self.lb.thread.list(thread_id=self.ceo_thread_id)
        if thread_response.success and thread_response.messages:
            for msg in thread_response.messages[-4:]:
                role = msg.get('role', 'unknown').upper()
                content = msg.get('content', '')
                print(f"  [{role}]: {content[:100]}...")
        print()

        print("[TEAM] Contributions:")
        for role, thread_id in self.employee_threads.items():
            thread_response = await self.lb.thread.list(thread_id=thread_id)
            if thread_response.success and thread_response.messages:
                print(f"  {role.capitalize()}: {len(thread_response.messages)} messages")
        print()

        print("=" * 70)
        print("ALL 9 PRIMITIVES DEMONSTRATED WITH WORKFLOW")
        print("=" * 70)
        print("  [1] Agent    - CEO + 3 employees")
        print("  [2] Pipe     - Writer agent uses Pipe")
        print("  [3] Memory   - Knowledge storage + RAG")
        print("  [4] Thread   - Conversation tracking")
        print("  [5] Workflow - PROPER orchestration with dependencies!")
        print("  [6] Parser   - Document processing (available)")
        print("  [7] Chunker  - Text segmentation (used in Memory)")
        print("  [8] Embed    - Vector embeddings (used in Memory)")
        print("  [9] Tools    - External capabilities (available)")
        print()


async def main():
    """Run the AI Agency with proper Workflow orchestration."""

    print("\n" + "=" * 70)
    print("  AI AGENCY - PROPER WORKFLOW ORCHESTRATION")
    print("  Using Workflow Primitive for Multi-Agent System")
    print("=" * 70)
    print()

    if not OPENAI_API_KEY:
        print("[ERROR] Set OPENAI_API_KEY in .env file")
        return

    agency = AIAgency(api_key=OPENAI_API_KEY)

    # Setup primitives
    print("[SETUP] Initializing primitives...")
    print("=" * 70)
    await agency.setup_memory()
    await agency.setup_threads()

    # Execute project with WORKFLOW primitive
    project = """
    Create a guide on "Getting Started with LangPy" covering:
    1. What LangPy is and its benefits
    2. The 9 core primitives
    3. A simple example
    4. Best practices

    Target: Python developers new to AI.
    Length: 800-1000 words.
    """

    # This now uses the Workflow primitive properly!
    results = await agency.execute_project_with_workflow(project)
    await agency.generate_summary(results)

    # Display final deliverable
    print("=" * 70)
    print("FINAL DELIVERABLE")
    print("=" * 70)
    if results and 'writing' in results:
        print(results['writing'])
    else:
        print("No content generated")
    print()
    print("=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\nThis properly demonstrated the Workflow primitive!")
    print("  - 5 steps with dependencies")
    print("  - Automatic dependency resolution")
    print("  - Step timing and error handling")
    print("  - Data flow between steps")
    print()


if __name__ == "__main__":
    asyncio.run(main())
