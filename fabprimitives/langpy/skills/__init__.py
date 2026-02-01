"""
LangPy Skills - Support for Claude Agent Skills format.

Claude Skills are folders containing instructions, scripts, and resources
that extend Claude's capabilities for specialized tasks.

Format specification: https://github.com/anthropics/skills

Skill Structure:
    my-skill/
    ├── SKILL.md          # Required - YAML frontmatter + markdown instructions
    ├── scripts/          # Optional - executable code
    ├── references/       # Optional - docs loaded as needed
    └── assets/           # Optional - templates, images, etc.

SKILL.md Format:
    ---
    name: my-skill
    description: What this skill does and when to use it
    ---

    # My Skill

    Instructions Claude follows when this skill is active...

    ## Examples
    - Example usage 1
    - Example usage 2

    ## Guidelines
    - Guideline 1
    - Guideline 2

Example usage:
    from langpy.skills import SkillManager, Skill

    # Create manager
    manager = SkillManager()

    # List available skills
    skills = manager.list()
    for s in skills:
        print(f"{s.name}: {s.description}")

    # Load a skill
    skill = manager.load("my-skill")
    print(skill.instructions)

    # Install from path
    manager.install("/path/to/skill-folder")

    # Create a new skill
    skill = manager.create(
        name="my-new-skill",
        description="Use this skill when working with XYZ",
        instructions="# Instructions\\n\\nDo this, then that..."
    )

    # Build context for prompts
    context = manager.build_context(skill_names=["my-skill"])
"""

# Models
from .models import (
    Skill,
    SkillMetadata,
    SkillResource,
    SkillIndex,
    ResourceType,
    get_default_skills_dir,
    get_langpy_skills_dir,
)

# Parser
from .parser import (
    SkillParseError,
    parse_skill_directory,
    parse_skill_file,
    parse_skill_md,
    create_skill_from_content,
)

# Manager
from .manager import (
    SkillManager,
    get_manager,
    list_skills,
    load_skill,
    install_skill,
    uninstall_skill,
    create_skill,
)

__all__ = [
    # Models
    "Skill",
    "SkillMetadata",
    "SkillResource",
    "SkillIndex",
    "ResourceType",
    "get_default_skills_dir",
    "get_langpy_skills_dir",

    # Parser
    "SkillParseError",
    "parse_skill_directory",
    "parse_skill_file",
    "parse_skill_md",
    "create_skill_from_content",

    # Manager
    "SkillManager",
    "get_manager",
    "list_skills",
    "load_skill",
    "install_skill",
    "uninstall_skill",
    "create_skill",
]
