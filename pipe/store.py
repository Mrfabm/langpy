import json, os
from pathlib import Path
from typing import Dict, List, Optional
from pipe.schema import PipePreset

BASE = Path(os.getenv("LANGPY_HOME", Path.home() / ".langpy")) / "pipes"
BASE.mkdir(parents=True, exist_ok=True)

def _path(name: str) -> Path:
    return BASE / f"{name}.json"

def create(preset: PipePreset):
    fp = _path(preset.name)
    if fp.exists() and not preset.upsert:
        raise FileExistsError(f"pipe '{preset.name}' exists")
    fp.write_text(preset.model_dump_json(indent=2))

def load(name: str) -> Dict:
    return json.loads(_path(name).read_text())

def update(name: str, **patch):
    data = load(name)
    data.update(patch)
    _path(name).write_text(json.dumps(data, indent=2))

def delete(name: str):
    _path(name).unlink(missing_ok=True)

def list_all() -> List[str]:
    return [p.stem for p in BASE.glob("*.json")]


class PipeStore:
    """
    Store for managing pipe presets.
    Provides async interface for the SDK.
    """
    
    async def add_pipe(self, preset: PipePreset) -> None:
        """Add a pipe preset to the store."""
        create(preset)
    
    async def get_pipe(self, name: str) -> Optional[PipePreset]:
        """Get a pipe preset by name."""
        try:
            data = load(name)
            return PipePreset(**data)
        except (FileNotFoundError, KeyError):
            return None
    
    async def delete_pipe(self, name: str) -> bool:
        """Delete a pipe preset by name."""
        try:
            delete(name)
            return True
        except FileNotFoundError:
            return False
    
    async def list_pipes(self) -> List[str]:
        """List all pipe names."""
        return list_all()
