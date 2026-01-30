"""
LangPy Skills - Data models for Claude Agent Skills format.

Claude Skills are folders containing a SKILL.md file with YAML frontmatter
and optional bundled resources (scripts, references, assets).

Format specification: https://github.com/anthropics/skills
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class ResourceType(str, Enum):
    """Types of bundled resources in a skill."""
    SCRIPT = "script"       # Executable code (scripts/)
    REFERENCE = "reference"  # Documentation loaded as needed (references/)
    ASSET = "asset"         # Files used in output (assets/)


@dataclass
class SkillMetadata:
    """
    YAML frontmatter metadata from SKILL.md.

    Required fields:
        name: Unique identifier for the skill
        description: What the skill does and when to use it

    The description is critical - it's used by Claude to determine
    when to activate the skill, so it should include trigger conditions.
    """
    name: str
    description: str

    # Optional metadata
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "name": self.name,
            "description": self.description,
        }
        if self.version != "1.0.0":
            d["version"] = self.version
        if self.author:
            d["author"] = self.author
        if self.tags:
            d["tags"] = self.tags
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillMetadata":
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            author=data.get("author"),
            tags=data.get("tags", []),
        )


@dataclass
class SkillResource:
    """
    A bundled resource within a skill.

    Resources can be:
    - scripts/: Executable code for deterministic tasks
    - references/: Documentation loaded into context as needed
    - assets/: Files used in output (templates, images, etc.)
    """
    name: str
    path: Path
    resource_type: ResourceType
    content: Optional[str] = None  # Loaded on demand

    @property
    def relative_path(self) -> str:
        """Get the relative path within the skill folder."""
        if self.resource_type == ResourceType.SCRIPT:
            return f"scripts/{self.name}"
        elif self.resource_type == ResourceType.REFERENCE:
            return f"references/{self.name}"
        elif self.resource_type == ResourceType.ASSET:
            return f"assets/{self.name}"
        return self.name

    def load_content(self) -> str:
        """Load the resource content from disk."""
        if self.content is None:
            self.content = self.path.read_text(encoding="utf-8")
        return self.content

    @property
    def is_text(self) -> bool:
        """Check if the resource is a text file."""
        text_extensions = {
            ".md", ".txt", ".py", ".js", ".ts", ".sh", ".bash",
            ".json", ".yaml", ".yml", ".xml", ".html", ".css",
            ".sql", ".r", ".rb", ".go", ".rs", ".java", ".kt",
        }
        return self.path.suffix.lower() in text_extensions


@dataclass
class Skill:
    """
    A Claude Agent Skill.

    Skills are folders containing:
    - SKILL.md (required): YAML frontmatter + markdown instructions
    - scripts/ (optional): Executable code
    - references/ (optional): Documentation for context
    - assets/ (optional): Files used in output

    Example structure:
        my-skill/
        ├── SKILL.md
        ├── scripts/
        │   └── process.py
        ├── references/
        │   └── api_docs.md
        └── assets/
            └── template.docx
    """
    metadata: SkillMetadata
    instructions: str  # Markdown body from SKILL.md
    path: Optional[Path] = None  # Skill folder path

    # Bundled resources
    scripts: List[SkillResource] = field(default_factory=list)
    references: List[SkillResource] = field(default_factory=list)
    assets: List[SkillResource] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Get the skill name."""
        return self.metadata.name

    @property
    def description(self) -> str:
        """Get the skill description."""
        return self.metadata.description

    @property
    def all_resources(self) -> List[SkillResource]:
        """Get all bundled resources."""
        return self.scripts + self.references + self.assets

    def get_script(self, name: str) -> Optional[SkillResource]:
        """Get a script by name."""
        for script in self.scripts:
            if script.name == name:
                return script
        return None

    def get_reference(self, name: str) -> Optional[SkillResource]:
        """Get a reference by name."""
        for ref in self.references:
            if ref.name == name:
                return ref
        return None

    def get_asset(self, name: str) -> Optional[SkillResource]:
        """Get an asset by name."""
        for asset in self.assets:
            if asset.name == name:
                return asset
        return None

    def to_skill_md(self) -> str:
        """
        Generate SKILL.md content.

        Returns:
            Complete SKILL.md with YAML frontmatter and instructions
        """
        import yaml

        # Build frontmatter
        frontmatter = yaml.dump(
            self.metadata.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True
        ).strip()

        return f"---\n{frontmatter}\n---\n\n{self.instructions}"

    def to_context(self, include_references: bool = False) -> str:
        """
        Generate context string for injection into prompts.

        This is used when the skill is activated to provide
        Claude with the skill instructions.

        Args:
            include_references: Whether to include reference content

        Returns:
            Formatted skill context
        """
        parts = [f"# Skill: {self.name}\n"]
        parts.append(self.instructions)

        if include_references and self.references:
            parts.append("\n\n## References\n")
            for ref in self.references:
                if ref.is_text:
                    content = ref.load_content()
                    parts.append(f"\n### {ref.name}\n{content}")

        return "\n".join(parts)

    def __repr__(self) -> str:
        resources = len(self.all_resources)
        return f"Skill(name={self.name!r}, resources={resources})"

    def __str__(self) -> str:
        return f"Skill: {self.name}"


@dataclass
class SkillIndex:
    """
    Index entry for a skill (metadata only).

    Used for skill discovery - Claude reads these to determine
    which skills to activate based on the task.
    """
    name: str
    description: str
    path: Path
    version: str = "1.0.0"

    @classmethod
    def from_skill(cls, skill: Skill) -> "SkillIndex":
        """Create index entry from a skill."""
        return cls(
            name=skill.name,
            description=skill.description,
            path=skill.path or Path("."),
            version=skill.metadata.version,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "path": str(self.path),
            "version": self.version,
        }


# Default skill storage locations
def get_default_skills_dir() -> Path:
    """Get the default skills directory (~/.claude/skills/)."""
    return Path.home() / ".claude" / "skills"


def get_langpy_skills_dir() -> Path:
    """Get the langpy skills directory (~/.langpy/skills/)."""
    base = Path(os.getenv("LANGPY_HOME", Path.home() / ".langpy"))
    return base / "skills"
