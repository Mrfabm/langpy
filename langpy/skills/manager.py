"""
LangPy Skills - Manager for loading and managing Claude Agent Skills.

Handles:
- Loading skills from directories
- Installing skills to ~/.claude/skills/
- Listing available skills
- Skill discovery and indexing

Format specification: https://github.com/anthropics/skills
"""

from __future__ import annotations
import os
import json
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional, Dict, Any

from .models import (
    Skill,
    SkillIndex,
    SkillMetadata,
    get_default_skills_dir,
    get_langpy_skills_dir,
)
from .parser import (
    parse_skill_directory,
    parse_skill_file,
    parse_skill_md,
    SkillParseError,
)


class SkillManager:
    """
    Manager for Claude Agent Skills.

    Handles loading, installing, and managing skills.

    Default locations:
    - ~/.claude/skills/ (Claude's default)
    - ~/.langpy/skills/ (LangPy's location)

    Example:
        manager = SkillManager()

        # List available skills
        skills = manager.list()

        # Load a skill
        skill = manager.load("my-skill")

        # Install from path
        manager.install("/path/to/skill-folder")

        # Create a new skill
        manager.create("my-new-skill", "Description of when to use it")
    """

    def __init__(
        self,
        skills_dir: Optional[Path] = None,
        use_claude_dir: bool = True,
        use_langpy_dir: bool = True,
    ):
        """
        Initialize the skill manager.

        Args:
            skills_dir: Custom skills directory (primary)
            use_claude_dir: Also search ~/.claude/skills/
            use_langpy_dir: Also search ~/.langpy/skills/
        """
        self._search_paths: List[Path] = []

        if skills_dir:
            self._search_paths.append(Path(skills_dir))

        if use_langpy_dir:
            self._search_paths.append(get_langpy_skills_dir())

        if use_claude_dir:
            self._search_paths.append(get_default_skills_dir())

        # Primary directory for installing new skills
        self._install_dir = skills_dir or get_langpy_skills_dir()

        # Ensure directories exist
        for path in self._search_paths:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def install_dir(self) -> Path:
        """Get the directory where new skills are installed."""
        return self._install_dir

    @property
    def search_paths(self) -> List[Path]:
        """Get all skill search paths."""
        return list(self._search_paths)

    def list(self) -> List[SkillIndex]:
        """
        List all available skills.

        Returns:
            List of SkillIndex entries with name, description, path
        """
        skills = []
        seen_names = set()

        for search_path in self._search_paths:
            if not search_path.exists():
                continue

            for item in search_path.iterdir():
                if not item.is_dir():
                    continue

                skill_md = item / "SKILL.md"
                if not skill_md.exists():
                    continue

                # Skip if we've already seen this skill name
                if item.name in seen_names:
                    continue
                seen_names.add(item.name)

                try:
                    content = skill_md.read_text(encoding="utf-8")
                    metadata, _ = parse_skill_md(content, skill_md)
                    skills.append(SkillIndex(
                        name=metadata.name,
                        description=metadata.description,
                        path=item,
                        version=metadata.version,
                    ))
                except (SkillParseError, Exception):
                    # Skip invalid skills
                    continue

        return skills

    def list_names(self) -> List[str]:
        """
        List all available skill names.

        Returns:
            List of skill names
        """
        return [s.name for s in self.list()]

    def exists(self, name: str) -> bool:
        """
        Check if a skill exists.

        Args:
            name: Skill name

        Returns:
            True if skill exists
        """
        return self._find_skill_path(name) is not None

    def _find_skill_path(self, name: str) -> Optional[Path]:
        """Find the path to a skill by name."""
        for search_path in self._search_paths:
            skill_path = search_path / name
            if skill_path.exists() and (skill_path / "SKILL.md").exists():
                return skill_path
        return None

    def load(self, name: str) -> Skill:
        """
        Load a skill by name.

        Args:
            name: Skill name

        Returns:
            Loaded Skill object

        Raises:
            FileNotFoundError: If skill not found
        """
        skill_path = self._find_skill_path(name)

        if skill_path is None:
            raise FileNotFoundError(f"Skill not found: {name}")

        return parse_skill_directory(skill_path)

    def load_from_path(self, path: Path) -> Skill:
        """
        Load a skill from a specific path.

        Args:
            path: Path to skill directory or SKILL.md file

        Returns:
            Loaded Skill object
        """
        path = Path(path)

        if path.is_file() and path.name == "SKILL.md":
            return parse_skill_file(path)
        elif path.is_dir():
            return parse_skill_directory(path)
        else:
            raise ValueError(f"Invalid skill path: {path}")

    def install(
        self,
        source: Path,
        name: Optional[str] = None,
        overwrite: bool = False
    ) -> Skill:
        """
        Install a skill from a path.

        Args:
            source: Path to skill directory or .skill file
            name: Override skill name (default: use source name)
            overwrite: Overwrite existing skill

        Returns:
            Installed Skill object

        Raises:
            FileExistsError: If skill exists and overwrite=False
        """
        source = Path(source)

        # Handle .skill file (zip archive)
        if source.suffix == ".skill":
            return self._install_from_archive(source, name, overwrite)

        # Load the skill first to validate
        skill = self.load_from_path(source)

        # Determine target name
        target_name = name or skill.name
        target_path = self._install_dir / target_name

        if target_path.exists() and not overwrite:
            raise FileExistsError(f"Skill already exists: {target_name}")

        # Copy skill directory
        if target_path.exists():
            shutil.rmtree(target_path)

        shutil.copytree(source, target_path)

        # Return the installed skill
        return parse_skill_directory(target_path)

    def _install_from_archive(
        self,
        archive_path: Path,
        name: Optional[str],
        overwrite: bool
    ) -> Skill:
        """Install a skill from a .skill archive."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract archive
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(tmpdir)

            # Find SKILL.md in extracted content
            tmpdir_path = Path(tmpdir)
            skill_dirs = list(tmpdir_path.glob("**/SKILL.md"))

            if not skill_dirs:
                raise SkillParseError(f"No SKILL.md found in {archive_path}")

            # Use the first found skill directory
            skill_dir = skill_dirs[0].parent

            return self.install(skill_dir, name, overwrite)

    def uninstall(self, name: str) -> bool:
        """
        Uninstall a skill.

        Args:
            name: Skill name to uninstall

        Returns:
            True if uninstalled, False if not found
        """
        # Only uninstall from install directory
        target_path = self._install_dir / name

        if target_path.exists():
            shutil.rmtree(target_path)
            return True

        return False

    def create(
        self,
        name: str,
        description: str,
        instructions: str = "",
        include_dirs: bool = True,
    ) -> Skill:
        """
        Create a new skill.

        Args:
            name: Skill name
            description: Description of what the skill does and when to use it
            instructions: Markdown instructions (optional, can be added later)
            include_dirs: Create scripts/, references/, assets/ directories

        Returns:
            Created Skill object
        """
        target_path = self._install_dir / name

        if target_path.exists():
            raise FileExistsError(f"Skill already exists: {name}")

        # Create skill directory
        target_path.mkdir(parents=True)

        # Create SKILL.md
        skill_md_content = f"""---
name: {name}
description: {description}
---

# {name}

{instructions if instructions else "[Add your instructions here]"}

## Examples
- Example usage 1
- Example usage 2

## Guidelines
- Guideline 1
- Guideline 2
"""

        (target_path / "SKILL.md").write_text(skill_md_content, encoding="utf-8")

        # Create optional directories
        if include_dirs:
            (target_path / "scripts").mkdir()
            (target_path / "references").mkdir()
            (target_path / "assets").mkdir()

        return parse_skill_directory(target_path)

    def get_index(self) -> List[Dict[str, Any]]:
        """
        Get the skill index for Claude's context.

        This returns minimal metadata that Claude uses to determine
        which skills to activate based on the task.

        Returns:
            List of dicts with name and description
        """
        return [
            {"name": s.name, "description": s.description}
            for s in self.list()
        ]

    def build_context(
        self,
        skill_names: Optional[List[str]] = None,
        include_index: bool = True,
    ) -> str:
        """
        Build context string for injection into prompts.

        Args:
            skill_names: Specific skills to load (None = just index)
            include_index: Include the skill index

        Returns:
            Formatted context string
        """
        parts = []

        if include_index:
            index = self.get_index()
            if index:
                parts.append("# Available Skills\n")
                for entry in index:
                    parts.append(f"- **{entry['name']}**: {entry['description']}")
                parts.append("")

        if skill_names:
            parts.append("# Active Skills\n")
            for name in skill_names:
                try:
                    skill = self.load(name)
                    parts.append(skill.to_context())
                    parts.append("")
                except FileNotFoundError:
                    parts.append(f"[Skill not found: {name}]\n")

        return "\n".join(parts)


# Module-level convenience instance
_default_manager: Optional[SkillManager] = None


def get_manager() -> SkillManager:
    """Get the default skill manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = SkillManager()
    return _default_manager


def list_skills() -> List[SkillIndex]:
    """List all available skills."""
    return get_manager().list()


def load_skill(name: str) -> Skill:
    """Load a skill by name."""
    return get_manager().load(name)


def install_skill(source: Path, name: Optional[str] = None, overwrite: bool = False) -> Skill:
    """Install a skill from a path."""
    return get_manager().install(source, name, overwrite)


def uninstall_skill(name: str) -> bool:
    """Uninstall a skill."""
    return get_manager().uninstall(name)


def create_skill(name: str, description: str, instructions: str = "") -> Skill:
    """Create a new skill."""
    return get_manager().create(name, description, instructions)
