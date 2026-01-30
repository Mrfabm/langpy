"""
Run registry for LangPy workflows.

This module provides persistent storage for workflow run history using SQLite,
matching Langbase's run tracking capabilities.
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorkflowRun:
    """Represents a workflow run with all metadata."""
    id: str
    workflow_name: str
    status: str  # 'running', 'completed', 'failed'
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    error: Optional[str]
    started_at: int  # Unix timestamp
    completed_at: Optional[int]  # Unix timestamp
    duration_ms: Optional[int]
    steps: List[Dict[str, Any]]
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    
    @classmethod
    def create(cls, workflow_name: str, inputs: Dict[str, Any]) -> 'WorkflowRun':
        """Create a new workflow run."""
        return cls(
            id=str(uuid.uuid4()),
            workflow_name=workflow_name,
            status='running',
            inputs=inputs,
            outputs={},
            error=None,
            started_at=int(time.time()),
            completed_at=None,
            duration_ms=None,
            steps=[],
            context={},
            metadata={}
        )
    
    def complete(self, outputs: Dict[str, Any]) -> None:
        """Mark the run as completed."""
        self.status = 'completed'
        self.outputs = outputs
        self.completed_at = int(time.time())
        self.duration_ms = (self.completed_at - self.started_at) * 1000
    
    def fail(self, error: str) -> None:
        """Mark the run as failed."""
        self.status = 'failed'
        self.error = error
        self.completed_at = int(time.time())
        self.duration_ms = (self.completed_at - self.started_at) * 1000
    
    def add_step(self, step_data: Dict[str, Any]) -> None:
        """Add a step to the run."""
        self.steps.append(step_data)
    
    def update_context(self, key: str, value: Any) -> None:
        """Update the run context."""
        self.context[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


@dataclass
class RunFilter:
    """Filter for querying workflow runs."""
    workflow_name: Optional[str] = None
    status: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: Optional[int] = None
    offset: Optional[int] = None


class RunRegistry:
    """Registry for persistent workflow run storage."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize run registry.
        
        Args:
            storage_path: Path to store runs database
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path.home() / ".langpy"
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "runs.db"
        
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the runs database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    id TEXT PRIMARY KEY,
                    workflow_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    inputs TEXT,
                    outputs TEXT,
                    error TEXT,
                    started_at INTEGER NOT NULL,
                    completed_at INTEGER,
                    duration_ms INTEGER,
                    steps TEXT,
                    context TEXT,
                    metadata TEXT,
                    created_at INTEGER DEFAULT (strftime('%s', 'now')),
                    updated_at INTEGER DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflow_runs_name 
                ON workflow_runs(workflow_name)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflow_runs_status 
                ON workflow_runs(status)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflow_runs_started 
                ON workflow_runs(started_at)
            """)
            
            conn.commit()
    
    def save_run(self, run: WorkflowRun) -> None:
        """Save a workflow run to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO workflow_runs (
                    id, workflow_name, status, inputs, outputs, error,
                    started_at, completed_at, duration_ms, steps, context, metadata,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.id,
                run.workflow_name,
                run.status,
                json.dumps(run.inputs),
                json.dumps(run.outputs),
                run.error,
                run.started_at,
                run.completed_at,
                run.duration_ms,
                json.dumps(run.steps),
                json.dumps(run.context),
                json.dumps(run.metadata),
                int(time.time())
            ))
            conn.commit()
    
    def get_run(self, run_id: str) -> Optional[WorkflowRun]:
        """Get a workflow run by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM workflow_runs WHERE id = ?",
                (run_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_run(row, cursor.description)
            return None
    
    def list_runs(self, filter_params: Optional[RunFilter] = None) -> List[WorkflowRun]:
        """List workflow runs with optional filtering."""
        query = "SELECT * FROM workflow_runs"
        params = []
        conditions = []
        
        if filter_params:
            if filter_params.workflow_name:
                conditions.append("workflow_name = ?")
                params.append(filter_params.workflow_name)
            
            if filter_params.status:
                conditions.append("status = ?")
                params.append(filter_params.status)
            
            if filter_params.start_date:
                conditions.append("started_at >= ?")
                params.append(int(filter_params.start_date.timestamp()))
            
            if filter_params.end_date:
                conditions.append("started_at <= ?")
                params.append(int(filter_params.end_date.timestamp()))
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY started_at DESC"
        
        if filter_params and filter_params.limit:
            query += " LIMIT ?"
            params.append(filter_params.limit)
            
            if filter_params.offset:
                query += " OFFSET ?"
                params.append(filter_params.offset)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_run(row, cursor.description) for row in cursor.fetchall()]
    
    def delete_run(self, run_id: str) -> bool:
        """Delete a workflow run."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM workflow_runs WHERE id = ?",
                (run_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_runs_by_workflow(self, workflow_name: str) -> int:
        """Delete all runs for a specific workflow."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM workflow_runs WHERE workflow_name = ?",
                (workflow_name,)
            )
            conn.commit()
            return cursor.rowcount
    
    def get_workflow_stats(self, workflow_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for workflow runs."""
        base_query = """
            SELECT 
                COUNT(*) as total_runs,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_runs,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
                SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running_runs,
                AVG(CASE WHEN duration_ms IS NOT NULL THEN duration_ms END) as avg_duration_ms,
                MIN(started_at) as first_run,
                MAX(started_at) as last_run
            FROM workflow_runs
        """
        
        params = []
        if workflow_name:
            base_query += " WHERE workflow_name = ?"
            params.append(workflow_name)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(base_query, params)
            row = cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                stats = dict(zip(columns, row))
                
                # Convert timestamps to readable dates
                if stats['first_run']:
                    stats['first_run_date'] = datetime.fromtimestamp(stats['first_run']).isoformat()
                if stats['last_run']:
                    stats['last_run_date'] = datetime.fromtimestamp(stats['last_run']).isoformat()
                
                # Calculate success rate
                if stats['total_runs'] > 0:
                    stats['success_rate'] = (stats['completed_runs'] / stats['total_runs']) * 100
                else:
                    stats['success_rate'] = 0.0
                
                return stats
            
            return {}
    
    def cleanup_old_runs(self, days_old: int = 30, keep_failed: bool = True) -> int:
        """Clean up old workflow runs."""
        cutoff_timestamp = int(time.time()) - (days_old * 24 * 60 * 60)
        
        query = "DELETE FROM workflow_runs WHERE started_at < ?"
        params = [cutoff_timestamp]
        
        if keep_failed:
            query += " AND status != 'failed'"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
    
    def export_runs(self, output_path: str, workflow_name: Optional[str] = None) -> int:
        """Export workflow runs to JSON file."""
        filter_params = RunFilter(workflow_name=workflow_name) if workflow_name else None
        runs = self.list_runs(filter_params)
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "workflow_name": workflow_name,
            "total_runs": len(runs),
            "runs": [run.to_dict() for run in runs]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return len(runs)
    
    def _row_to_run(self, row: tuple, description: List[tuple]) -> WorkflowRun:
        """Convert database row to WorkflowRun object."""
        columns = [desc[0] for desc in description]
        data = dict(zip(columns, row))
        
        # Parse JSON fields
        for field in ['inputs', 'outputs', 'steps', 'context', 'metadata']:
            if data.get(field):
                try:
                    data[field] = json.loads(data[field])
                except (json.JSONDecodeError, TypeError):
                    data[field] = {}
            else:
                data[field] = {} if field != 'steps' else []
        
        # Remove database-specific fields
        for field in ['created_at', 'updated_at']:
            data.pop(field, None)
        
        return WorkflowRun(**data)


# Global registry instance
_run_registry: Optional[RunRegistry] = None


def get_run_registry(storage_path: Optional[str] = None) -> RunRegistry:
    """Get the global run registry instance."""
    global _run_registry
    if _run_registry is None:
        _run_registry = RunRegistry(storage_path=storage_path)
    return _run_registry


def set_run_registry(registry: RunRegistry) -> None:
    """Set the global run registry instance."""
    global _run_registry
    _run_registry = registry


# Convenience functions
def save_run(run: WorkflowRun) -> None:
    """Save a workflow run."""
    registry = get_run_registry()
    registry.save_run(run)


def get_run(run_id: str) -> Optional[WorkflowRun]:
    """Get a workflow run by ID."""
    registry = get_run_registry()
    return registry.get_run(run_id)


def list_runs(workflow_name: Optional[str] = None, 
              status: Optional[str] = None,
              limit: Optional[int] = None) -> List[WorkflowRun]:
    """List workflow runs with optional filtering."""
    registry = get_run_registry()
    filter_params = RunFilter(
        workflow_name=workflow_name,
        status=status,
        limit=limit
    )
    return registry.list_runs(filter_params)


def get_workflow_stats(workflow_name: Optional[str] = None) -> Dict[str, Any]:
    """Get workflow statistics."""
    registry = get_run_registry()
    return registry.get_workflow_stats(workflow_name)


def cleanup_old_runs(days_old: int = 30, keep_failed: bool = True) -> int:
    """Clean up old workflow runs."""
    registry = get_run_registry()
    return registry.cleanup_old_runs(days_old, keep_failed) 