"""
LangPy SDK - Skill interface for Claude Agent Skills.

High-level interface for loading and using Claude Skills.

Format specification: https://github.com/anthropics/skills

Example:
    from langpy_sdk import Skill, SkillManager

    # Load a skill
    skill = Skill.load("my-skill")
    print(skill.name)
    print(skill.description)
    print(skill.instructions)

    # Create a skill
    skill = Skill.create(
        name="my-skill",
        description="Use this when working with XYZ",
        instructions="# Instructions\\n\\nDo this..."
    )
    skill.save()

    # Use SkillManager for advanced operations
    manager = SkillManager()
    skills = manager.list()
    context = manager.build_context(skill_names=["skill1", "skill2"])
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import from langpy.skills
from langpy.skills import (
    Skill as SkillModel,
    SkillMetadata,
    SkillManager as BaseSkillManager,
    SkillIndex,
    SkillResource,
    ResourceType,
    SkillParseError,
    parse_skill_directory,
    parse_skill_file,
    parse_skill_md,
    create_skill_from_content,
    get_manager,
    get_default_skills_dir,
    get_langpy_skills_dir,
)


class Skill:
    """
    Claude Agent Skill.

    Skills are folders containing instructions, scripts, and resources
    that extend Claude's capabilities for specialized tasks.

    Skill Structure:
        my-skill/
        ├── SKILL.md          # Required
        ├── scripts/          # Optional
        ├── references/       # Optional
        └── assets/           # Optional

    Example:
        # Load existing skill
        skill = Skill.load("my-skill")

        # Create new skill
        skill = Skill.create(
            name="my-skill",
            description="Use when doing XYZ",
            instructions="# My Skill\\n\\nInstructions here..."
        )
        skill.save()

        # Load from path
        skill = Skill.from_path("/path/to/skill")
    """

    def __init__(self, model: SkillModel):
        """
        Initialize from a SkillModel.

        Use Skill.load(), Skill.create(), or Skill.from_path() instead.
        """
        self._model = model

    @property
    def name(self) -> str:
        """Get the skill name."""
        return self._model.name

    @property
    def description(self) -> str:
        """Get the skill description."""
        return self._model.description

    @property
    def instructions(self) -> str:
        """Get the skill instructions (markdown body)."""
        return self._model.instructions

    @property
    def metadata(self) -> SkillMetadata:
        """Get the skill metadata."""
        return self._model.metadata

    @property
    def path(self) -> Optional[Path]:
        """Get the skill path (if loaded from disk)."""
        return self._model.path

    @property
    def scripts(self) -> List[SkillResource]:
        """Get the skill scripts."""
        return self._model.scripts

    @property
    def references(self) -> List[SkillResource]:
        """Get the skill references."""
        return self._model.references

    @property
    def assets(self) -> List[SkillResource]:
        """Get the skill assets."""
        return self._model.assets

    def get_script(self, name: str) -> Optional[SkillResource]:
        """Get a script by name."""
        return self._model.get_script(name)

    def get_reference(self, name: str) -> Optional[SkillResource]:
        """Get a reference by name."""
        return self._model.get_reference(name)

    def get_asset(self, name: str) -> Optional[SkillResource]:
        """Get an asset by name."""
        return self._model.get_asset(name)

    def to_skill_md(self) -> str:
        """Generate SKILL.md content."""
        return self._model.to_skill_md()

    def to_context(self, include_references: bool = False) -> str:
        """
        Generate context string for injection into prompts.

        Args:
            include_references: Include reference content

        Returns:
            Formatted context string
        """
        return self._model.to_context(include_references)

    def save(self, path: Optional[Path] = None, overwrite: bool = True) -> Path:
        """
        Save the skill to disk.

        Args:
            path: Target path (default: ~/.langpy/skills/{name}/)
            overwrite: Overwrite if exists

        Returns:
            Path where skill was saved
        """
        if path is None:
            path = get_langpy_skills_dir() / self.name

        path = Path(path)

        if path.exists() and not overwrite:
            raise FileExistsError(f"Path exists: {path}")

        path.mkdir(parents=True, exist_ok=True)

        # Write SKILL.md
        (path / "SKILL.md").write_text(self.to_skill_md(), encoding="utf-8")

        # Create resource directories
        (path / "scripts").mkdir(exist_ok=True)
        (path / "references").mkdir(exist_ok=True)
        (path / "assets").mkdir(exist_ok=True)

        # Copy resources if they exist
        for script in self.scripts:
            if script.path and script.path.exists():
                import shutil
                shutil.copy2(script.path, path / "scripts" / script.name)

        for ref in self.references:
            if ref.path and ref.path.exists():
                import shutil
                shutil.copy2(ref.path, path / "references" / ref.name)

        for asset in self.assets:
            if asset.path and asset.path.exists():
                import shutil
                shutil.copy2(asset.path, path / "assets" / asset.name)

        return path

    @classmethod
    def load(cls, name: str) -> "Skill":
        """
        Load a skill by name.

        Searches:
        - ~/.langpy/skills/
        - ~/.claude/skills/

        Args:
            name: Skill name

        Returns:
            Loaded Skill

        Raises:
            FileNotFoundError: If skill not found
        """
        manager = get_manager()
        model = manager.load(name)
        return cls(model)

    @classmethod
    def from_path(cls, path: Path) -> "Skill":
        """
        Load a skill from a specific path.

        Args:
            path: Path to skill directory or SKILL.md file

        Returns:
            Loaded Skill
        """
        path = Path(path)

        if path.is_file() and path.name == "SKILL.md":
            model = parse_skill_file(path)
        elif path.is_dir():
            model = parse_skill_directory(path)
        else:
            raise ValueError(f"Invalid skill path: {path}")

        return cls(model)

    @classmethod
    def from_content(cls, content: str) -> "Skill":
        """
        Create a skill from SKILL.md content.

        Args:
            content: SKILL.md content string

        Returns:
            Skill object
        """
        metadata, instructions = parse_skill_md(content)
        model = SkillModel(
            metadata=metadata,
            instructions=instructions,
        )
        return cls(model)

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        instructions: str = "",
        version: str = "1.0.0",
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> "Skill":
        """
        Create a new skill programmatically.

        Args:
            name: Skill name
            description: What the skill does and when to use it
            instructions: Markdown instructions
            version: Skill version
            author: Author name
            tags: List of tags

        Returns:
            New Skill object (not saved to disk)

        Example:
            skill = Skill.create(
                name="my-skill",
                description="Use when working with XYZ format",
                instructions='''
# My Skill

Instructions for using this skill...

## Examples
- Do X
- Do Y

## Guidelines
- Always Z
'''
            )
            skill.save()
        """
        model = create_skill_from_content(
            name=name,
            description=description,
            instructions=instructions,
            version=version,
            author=author,
            tags=tags or [],
        )
        return cls(model)

    @classmethod
    def list(cls) -> List["SkillInfo"]:
        """
        List all available skills.

        Returns:
            List of SkillInfo objects
        """
        manager = get_manager()
        return [
            SkillInfo(
                name=idx.name,
                description=idx.description,
                path=idx.path,
                version=idx.version,
            )
            for idx in manager.list()
        ]

    @classmethod
    def exists(cls, name: str) -> bool:
        """Check if a skill exists."""
        manager = get_manager()
        return manager.exists(name)

    @classmethod
    def delete(cls, name: str) -> bool:
        """Delete a skill."""
        manager = get_manager()
        return manager.uninstall(name)

    def __repr__(self) -> str:
        return f"Skill(name={self.name!r})"

    def __str__(self) -> str:
        return f"Skill: {self.name}"


class SkillInfo:
    """Lightweight skill info (metadata only)."""

    def __init__(
        self,
        name: str,
        description: str,
        path: Path,
        version: str = "1.0.0"
    ):
        self.name = name
        self.description = description
        self.path = path
        self.version = version

    def load(self) -> Skill:
        """Load the full skill."""
        return Skill.load(self.name)

    def __repr__(self) -> str:
        return f"SkillInfo(name={self.name!r})"


# Re-export SkillManager with same interface
class SkillManager(BaseSkillManager):
    """
    Manager for Claude Agent Skills.

    Handles loading, installing, and managing skills.

    Example:
        manager = SkillManager()

        # List skills
        for skill in manager.list():
            print(f"{skill.name}: {skill.description}")

        # Load a skill
        skill = manager.load("my-skill")

        # Install from path
        manager.install("/path/to/skill")

        # Create new skill
        manager.create("new-skill", "Description")

        # Build context for prompts
        context = manager.build_context(skill_names=["skill1"])
    """
    pass


# Convenience exports
__all__ = [
    "Skill",
    "SkillInfo",
    "SkillManager",
    "SkillParseError",
]
