# LangPy Gaps and Solutions

## Current State

LangPy has solid **data-flow primitives** (nouns):

| Primitive | Purpose |
|-----------|---------|
| Pipe | Single LLM call |
| Agent | LLM + tool execution loop |
| Memory | Vector storage & RAG |
| Thread | Conversation state |
| Workflow | Multi-step orchestration |
| Parser | Document parsing |
| Chunker | Text chunking |
| Embed | Embeddings generation |

These compose linearly: `Memory | Pipe | Agent`

---

## Identified Gaps

### 1. Control Flow (Missing "Connector" Blocks)

| Gap | Description | Current Workaround |
|-----|-------------|-------------------|
| **Branch** | Conditional path selection at runtime | Manual if/else in Workflow steps |
| **Loop** | Repeat until condition met | Manual while loops outside primitives |
| **Select** | Dynamically choose which primitive to use | Hardcoded paths |

### 2. Agent Coordination

| Gap | Description | Current Workaround |
|-----|-------------|-------------------|
| **Spawn** | Create agent instances dynamically | Pre-defined agents only |
| **Message** | Agent-to-agent communication | None - agents isolated |
| **Wait** | Pause for external event or other agent | Manual async coordination |
| **Discovery** | Agents find/register with each other | None |

### 3. State Management

| Gap | Description | Current Workaround |
|-----|-------------|-------------------|
| **Shared State** | Selective memory sharing between agents | Full Memory access or none |
| **Scoped Context** | Different context views per agent | Single Context flows through |

---

## Proposed Solutions

### Solution 1: Skills (Orchestration Layer)

**What it solves:** Branch, Loop, Select, Spawn, Wait

**Concept:** Skills are reusable orchestration patterns composed from primitives.

```
┌─────────────────────────────────────┐
│            SKILLS LAYER             │
│  (Branch, Loop, Select, Spawn, Wait)│
├─────────────────────────────────────┤
│          PRIMITIVES LAYER           │
│  (Pipe, Agent, Memory, Thread, ...) │
└─────────────────────────────────────┘
```

**Skill Definition Structure:**

```yaml
skill:
  name: "skill-name"
  description: "What it does"

  inputs:
    param1: type
    param2: type = default

  outputs:
    result: type

  agents:
    agent_name:
      primitive: Agent
      config: { ... }
      instances: static | dynamic

  state:
    shared_var: type

  flow:
    - id: step1
      type: action | branch | loop | parallel-spawn
      agent: agent_name
      action: "what to do"

    - id: step2
      type: branch
      condition: "expression"
      if_true: { ... }
      if_false: { ... }

    - id: step3
      type: loop
      agent: agent_name
      action: "iterate"
      until: "exit condition"
      max_iterations: N

    - id: step4
      type: parallel-spawn
      agent: agent_name
      for_each: items
      output: append to state.results
      until: all complete
```

**To Build:**
- [ ] Skill definition parser (YAML/Python)
- [ ] Skill executor engine
- [ ] Skill registry
- [ ] Integration with `|` and `&` operators
- [ ] Skill-to-skill composition

---

### Solution 2: MCP Integration (Communication Layer)

**What it solves:** Message, Discovery, Agent-to-Agent communication

**Concept:** Agents expose MCP interfaces; other agents connect as clients.

```
┌─────────────┐     MCP      ┌─────────────┐
│   Agent A   │◄────────────►│   Agent B   │
│ (MCP Server)│              │ (MCP Client)│
└─────────────┘              └─────────────┘
        │                            │
        └──────────┬─────────────────┘
                   │
            ┌──────┴──────┐
            │  Registry   │
            │(MCP Resource)│
            └─────────────┘
```

**Agent as MCP Server exposes:**
- Tools (agent capabilities)
- Resources (agent state, knowledge)
- Prompts (agent instructions)

**Agent as MCP Client can:**
- Call other agents' tools
- Read other agents' resources
- Discover available agents via registry

**To Build:**
- [ ] MCP server wrapper for Agent primitive
- [ ] MCP client integration in Agent
- [ ] Agent registry as MCP resource
- [ ] Message protocol over MCP

---

## Coverage Matrix

| Gap | Solution | Component |
|-----|----------|-----------|
| Branch | Skills | `type: branch` in flow |
| Loop | Skills | `type: loop` with `until` |
| Select | Skills | Dynamic primitive in flow |
| Spawn | Skills | `instances: dynamic` |
| Wait | Skills | `until: all complete` |
| Message | MCP | Agent MCP server/client |
| Discovery | MCP | Agent registry resource |
| Shared State | Skills | `state:` block in skill |

---

## Implementation Priority

### Phase 1: Skills (Core)
1. Skill definition format
2. Basic flow executor (sequential steps)
3. Branch support
4. Loop support
5. Skill registry

### Phase 2: Skills (Advanced)
1. Spawn support
2. Wait/sync support
3. Parallel execution
4. Skill composition (skills calling skills)
5. Integration with `|` operator

### Phase 3: MCP Integration
1. Agent as MCP server
2. Agent as MCP client
3. Agent registry
4. Message passing

---

## Example: Complete Skill

```yaml
skill:
  name: "research-team"
  description: "Multi-agent research with dynamic spawning"

  inputs:
    query: string
    depth: int = 3

  outputs:
    report: string
    sources: list

  agents:
    coordinator:
      primitive: Agent
      config:
        model: "gpt-4o"
        system: "You coordinate research tasks"

    researcher:
      primitive: Agent
      config:
        model: "gpt-4o-mini"
        system: "You research specific topics"
      instances: dynamic

  state:
    findings: list
    confidence: float = 0.0

  flow:
    - id: plan
      agent: coordinator
      action: "Decompose {{inputs.query}} into subtasks"
      output: subtasks

    - id: research
      type: parallel-spawn
      agent: researcher
      for_each: subtasks
      action: "Research: {{item}}"
      output: append to state.findings
      until: all complete

    - id: evaluate
      type: loop
      agent: coordinator
      action: "Evaluate findings, identify gaps"
      output: state.confidence
      until: state.confidence >= 0.8
      max_iterations: 3
      on_continue:
        - id: fill-gaps
          type: parallel-spawn
          agent: researcher
          for_each: gaps
          action: "Research gap: {{item}}"
          output: append to state.findings

    - id: synthesize
      type: branch
      condition: state.findings.length > 10
      if_true:
        agent: coordinator
        action: "Create detailed report from {{state.findings}}"
      if_false:
        agent: coordinator
        action: "Create brief summary from {{state.findings}}"
      output: outputs.report
```

---

## Notes

- Skills keep primitives simple (nouns only)
- Orchestration logic lives in Skills (verbs)
- MCP enables agent communication without custom protocol
- Both solutions are additive - no changes to existing primitives needed
