"""
LangPy Skills - Parser for SKILL.md files.

Parses the Claude Agent Skills format:
- YAML frontmatter (between --- markers)
- Markdown instructions body

Format specification: https://github.com/anthropics/skills
"""

from __future__ import annotations
import re
import yaml
from pathlib import Path
from typing import Optional, Tuple, List

from .models import (
    Skill,
    SkillMetadata,
    SkillResource,
    ResourceType,
)


class SkillParseError(Exception):
    """Error raised when parsing a skill fails."""

    def __init__(
        self,
        message: str,
        path: Optional[Path] = None,
        line: Optional[int] = None
    ):
        self.message = message
        self.path = path
        self.line = line
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [self.message]
        if self.path:
            parts.append(f"in {self.path}")
        if self.line:
            parts.append(f"at line {self.line}")
        return " ".join(parts)


# Pattern to match YAML frontmatter
FRONTMATTER_PATTERN = re.compile(
    r'^---\s*\n(.*?)\n---\s*\n',
    re.DOTALL
)


def parse_frontmatter(content: str) -> Tuple[dict, str]:
    """
    Parse YAML frontmatter from SKILL.md content.

    Args:
        content: Raw SKILL.md content

    Returns:
        Tuple of (frontmatter_dict, body_content)

    Raises:
        SkillParseError: If frontmatter is missing or invalid
    """
    match = FRONTMATTER_PATTERN.match(content)

    if not match:
        raise SkillParseError(
            "SKILL.md must start with YAML frontmatter (between --- markers)"
        )

    frontmatter_yaml = match.group(1)
    body = content[match.end():].strip()

    try:
        frontmatter = yaml.safe_load(frontmatter_yaml)
    except yaml.YAMLError as e:
        raise SkillParseError(f"Invalid YAML frontmatter: {e}")

    if not isinstance(frontmatter, dict):
        raise SkillParseError("Frontmatter must be a YAML mapping")

    return frontmatter, body


def validate_metadata(metadata: dict, path: Optional[Path] = None) -> None:
    """
    Validate required metadata fields.

    Args:
        metadata: Parsed frontmatter dict
        path: Optional path for error messages

    Raises:
        SkillParseError: If required fields are missing
    """
    if "name" not in metadata:
        raise SkillParseError("Missing required field: name", path=path)

    if "description" not in metadata:
        raise SkillParseError("Missing required field: description", path=path)

    if not metadata["name"]:
        raise SkillParseError("Field 'name' cannot be empty", path=path)

    if not metadata["description"]:
        raise SkillParseError("Field 'description' cannot be empty", path=path)


def parse_skill_md(content: str, path: Optional[Path] = None) -> Tuple[SkillMetadata, str]:
    """
    Parse a SKILL.md file content.

    Args:
        content: Raw SKILL.md content
        path: Optional path for error messages

    Returns:
        Tuple of (SkillMetadata, instructions_body)

    Raises:
        SkillParseError: If parsing fails
    """
    frontmatter, body = parse_frontmatter(content)
    validate_metadata(frontmatter, path)

    metadata = SkillMetadata.from_dict(frontmatter)
    return metadata, body


def discover_resources(skill_dir: Path) -> Tuple[List[SkillResource], List[SkillResource], List[SkillResource]]:
    """
    Discover bundled resources in a skill directory.

    Args:
        skill_dir: Path to skill directory

    Returns:
        Tuple of (scripts, references, assets)
    """
    scripts = []
    references = []
    assets = []

    # Discover scripts
    scripts_dir = skill_dir / "scripts"
    if scripts_dir.exists() and scripts_dir.is_dir():
        for file_path in scripts_dir.iterdir():
            if file_path.is_file():
                scripts.append(SkillResource(
                    name=file_path.name,
                    path=file_path,
                    resource_type=ResourceType.SCRIPT,
                ))

    # Discover references
    refs_dir = skill_dir / "references"
    if refs_dir.exists() and refs_dir.is_dir():
        for file_path in refs_dir.iterdir():
            if file_path.is_file():
                references.append(SkillResource(
                    name=file_path.name,
                    path=file_path,
                    resource_type=ResourceType.REFERENCE,
                ))

    # Discover assets
    assets_dir = skill_dir / "assets"
    if assets_dir.exists() and assets_dir.is_dir():
        for file_path in assets_dir.iterdir():
            if file_path.is_file():
                assets.append(SkillResource(
                    name=file_path.name,
                    path=file_path,
                    resource_type=ResourceType.ASSET,
                ))

    return scripts, references, assets


def parse_skill_directory(skill_dir: Path) -> Skill:
    """
    Parse a complete skill from a directory.

    Args:
        skill_dir: Path to skill directory (must contain SKILL.md)

    Returns:
        Parsed Skill object

    Raises:
        SkillParseError: If parsing fails
        FileNotFoundError: If SKILL.md doesn't exist
    """
    skill_dir = Path(skill_dir)

    if not skill_dir.exists():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

    skill_md_path = skill_dir / "SKILL.md"

    if not skill_md_path.exists():
        raise FileNotFoundError(f"SKILL.md not found in {skill_dir}")

    # Parse SKILL.md
    content = skill_md_path.read_text(encoding="utf-8")
    metadata, instructions = parse_skill_md(content, skill_md_path)

    # Discover resources
    scripts, references, assets = discover_resources(skill_dir)

    return Skill(
        metadata=metadata,
        instructions=instructions,
        path=skill_dir,
        scripts=scripts,
        references=references,
        assets=assets,
    )


def parse_skill_file(skill_md_path: Path) -> Skill:
    """
    Parse a skill from a SKILL.md file (no bundled resources).

    Args:
        skill_md_path: Path to SKILL.md file

    Returns:
        Parsed Skill object (without bundled resources)

    Raises:
        SkillParseError: If parsing fails
        FileNotFoundError: If file doesn't exist
    """
    skill_md_path = Path(skill_md_path)

    if not skill_md_path.exists():
        raise FileNotFoundError(f"File not found: {skill_md_path}")

    content = skill_md_path.read_text(encoding="utf-8")
    metadata, instructions = parse_skill_md(content, skill_md_path)

    # Check if there's a parent directory with resources
    parent = skill_md_path.parent
    scripts, references, assets = [], [], []

    if parent.name != skill_md_path.stem:
        # SKILL.md is in a skill folder, check for resources
        scripts, references, assets = discover_resources(parent)

    return Skill(
        metadata=metadata,
        instructions=instructions,
        path=parent if scripts or references or assets else None,
        scripts=scripts,
        references=references,
        assets=assets,
    )


def create_skill_from_content(
    name: str,
    description: str,
    instructions: str,
    **extra_metadata
) -> Skill:
    """
    Create a Skill object programmatically.

    Args:
        name: Skill name
        description: Skill description (when to use it)
        instructions: Markdown instructions
        **extra_metadata: Additional metadata (version, author, tags)

    Returns:
        Skill object
    """
    metadata = SkillMetadata(
        name=name,
        description=description,
        version=extra_metadata.get("version", "1.0.0"),
        author=extra_metadata.get("author"),
        tags=extra_metadata.get("tags", []),
    )

    return Skill(
        metadata=metadata,
        instructions=instructions,
    )
