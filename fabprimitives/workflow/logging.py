"""
Rich console logging for LangPy workflows.

This module provides rich console output with debug telemetry, banners,
and formatted logging similar to Langbase's workflow execution.
"""

import logging
import sys
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

# Try to import rich for enhanced formatting
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, TaskID
    from rich.text import Text
    from rich.logging import RichHandler
    from rich.traceback import install as install_rich_traceback
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
    install_rich_traceback()
except ImportError:
    RICH_AVAILABLE = False

# Fallback console if rich is not available
console = Console() if RICH_AVAILABLE else None


class WorkflowLogger:
    """Enhanced logger for workflow execution with rich formatting."""
    
    def __init__(self, debug: bool = False, use_rich: bool = True):
        """
        Initialize WorkflowLogger.
        
        Args:
            debug: Enable debug mode
            use_rich: Use rich formatting if available
        """
        self.debug = debug
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console = Console() if self.use_rich else None
        self.start_times: Dict[str, float] = {}
        self.step_counts: Dict[str, int] = {}
        
        # Set up logger
        self.logger = logging.getLogger("langpy.workflow")
        if not self.logger.handlers:
            self._setup_logger()
    
    def _setup_logger(self):
        """Set up the logger with appropriate handlers."""
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        
        if self.use_rich:
            # Rich handler for enhanced formatting
            handler = RichHandler(
                console=self.console,
                show_time=True,
                show_level=True,
                show_path=False,
                rich_tracebacks=True
            )
            formatter = logging.Formatter(
                fmt="%(message)s",
                datefmt="[%X]"
            )
        else:
            # Standard handler for compatibility
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def print_workflow_banner(self, workflow_name: str, run_id: str, inputs: Dict[str, Any]):
        """Print workflow start banner."""
        if self.use_rich:
            # Rich banner
            banner_text = f"🚀 Starting Workflow: [bold blue]{workflow_name}[/bold blue]"
            banner_text += f"\n📋 Run ID: [dim]{run_id}[/dim]"
            banner_text += f"\n⏰ Started: [dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]"
            
            if inputs:
                banner_text += f"\n📥 Inputs: [dim]{len(inputs)} parameters[/dim]"
                if self.debug:
                    for key, value in inputs.items():
                        if key != '_run_id':  # Skip internal keys
                            banner_text += f"\n   • [cyan]{key}[/cyan]: {repr(value)[:100]}..."
            
            self.console.print(Panel(
                banner_text,
                title="[bold green]Workflow Execution[/bold green]",
                border_style="green",
                padding=(1, 2)
            ))
        else:
            # Plain text banner
            print(f"\n{'='*60}")
            print(f"🚀 Starting Workflow: {workflow_name}")
            print(f"📋 Run ID: {run_id}")
            print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if inputs:
                print(f"📥 Inputs: {len(inputs)} parameters")
                if self.debug:
                    for key, value in inputs.items():
                        if key != '_run_id':
                            print(f"   • {key}: {repr(value)[:100]}...")
            print(f"{'='*60}\n")
    
    def print_workflow_completion(self, workflow_name: str, run_id: str, 
                                 outputs: Dict[str, Any], elapsed_ms: int):
        """Print workflow completion banner."""
        if self.use_rich:
            # Rich completion banner
            banner_text = f"✅ Workflow Complete: [bold green]{workflow_name}[/bold green]"
            banner_text += f"\n📋 Run ID: [dim]{run_id}[/dim]"
            banner_text += f"\n⏱️  Duration: [bold]{elapsed_ms}ms[/bold]"
            banner_text += f"\n📤 Outputs: [dim]{len(outputs)} results[/dim]"
            
            if self.debug and outputs:
                for key, value in outputs.items():
                    banner_text += f"\n   • [cyan]{key}[/cyan]: {repr(value)[:100]}..."
            
            self.console.print(Panel(
                banner_text,
                title="[bold green]Workflow Success[/bold green]",
                border_style="green",
                padding=(1, 2)
            ))
        else:
            # Plain text completion banner
            print(f"\n{'='*60}")
            print(f"✅ Workflow Complete: {workflow_name}")
            print(f"📋 Run ID: {run_id}")
            print(f"⏱️  Duration: {elapsed_ms}ms")
            print(f"📤 Outputs: {len(outputs)} results")
            if self.debug and outputs:
                for key, value in outputs.items():
                    print(f"   • {key}: {repr(value)[:100]}...")
            print(f"{'='*60}\n")
    
    def print_workflow_error(self, workflow_name: str, run_id: str, 
                            error: Exception, elapsed_ms: int):
        """Print workflow error banner."""
        if self.use_rich:
            # Rich error banner
            banner_text = f"❌ Workflow Failed: [bold red]{workflow_name}[/bold red]"
            banner_text += f"\n📋 Run ID: [dim]{run_id}[/dim]"
            banner_text += f"\n⏱️  Duration: [bold]{elapsed_ms}ms[/bold]"
            banner_text += f"\n💥 Error: [red]{str(error)}[/red]"
            
            self.console.print(Panel(
                banner_text,
                title="[bold red]Workflow Failed[/bold red]",
                border_style="red",
                padding=(1, 2)
            ))
        else:
            # Plain text error banner
            print(f"\n{'='*60}")
            print(f"❌ Workflow Failed: {workflow_name}")
            print(f"📋 Run ID: {run_id}")
            print(f"⏱️  Duration: {elapsed_ms}ms")
            print(f"💥 Error: {str(error)}")
            print(f"{'='*60}\n")
    
    def log_step_start(self, step_id: str, step_type: str, config: Dict[str, Any] = None):
        """Log step start."""
        self.start_times[step_id] = time.time()
        
        if self.use_rich:
            step_text = f"[bold blue]Step:[/bold blue] {step_id} [dim]({step_type})[/dim]"
            if config and self.debug:
                step_text += f"\n   Config: {config}"
            self.console.print(f"🔄 {step_text}")
        else:
            self.logger.info(f"🔄 Step: {step_id} ({step_type})")
            if config and self.debug:
                self.logger.debug(f"   Config: {config}")
    
    def log_step_success(self, step_id: str, result: Any = None):
        """Log step success."""
        elapsed_ms = int((time.time() - self.start_times.get(step_id, 0)) * 1000)
        
        if self.use_rich:
            step_text = f"[bold green]Step:[/bold green] {step_id} [dim]({elapsed_ms}ms)[/dim]"
            if result is not None and self.debug:
                step_text += f"\n   Result: {repr(result)[:200]}..."
            self.console.print(f"✅ {step_text}")
        else:
            self.logger.info(f"✅ Step: {step_id} ({elapsed_ms}ms)")
            if result is not None and self.debug:
                self.logger.debug(f"   Result: {repr(result)[:200]}...")
    
    def log_step_failure(self, step_id: str, error: Exception):
        """Log step failure."""
        elapsed_ms = int((time.time() - self.start_times.get(step_id, 0)) * 1000)
        
        if self.use_rich:
            step_text = f"[bold red]Step:[/bold red] {step_id} [dim]({elapsed_ms}ms)[/dim]"
            step_text += f"\n   Error: [red]{str(error)}[/red]"
            self.console.print(f"❌ {step_text}")
        else:
            self.logger.error(f"❌ Step: {step_id} ({elapsed_ms}ms)")
            self.logger.error(f"   Error: {str(error)}")
    
    def log_step_retry(self, step_id: str, attempt: int, delay_ms: int, error: Exception):
        """Log step retry."""
        if self.use_rich:
            retry_text = f"[bold yellow]Retry:[/bold yellow] {step_id} "
            retry_text += f"[dim](attempt {attempt}, sleeping {delay_ms}ms)[/dim]"
            retry_text += f"\n   Reason: [yellow]{str(error)}[/yellow]"
            self.console.print(f"🔄 {retry_text}")
        else:
            self.logger.warning(f"🔄 Retry: {step_id} (attempt {attempt}, sleeping {delay_ms}ms)")
            self.logger.warning(f"   Reason: {str(error)}")
    
    def log_parallel_start(self, step_ids: List[str], group_name: str):
        """Log parallel execution start."""
        if self.use_rich:
            parallel_text = f"[bold magenta]Parallel Group:[/bold magenta] {group_name}"
            parallel_text += f"\n   Steps: {', '.join(step_ids)}"
            self.console.print(f"🔀 {parallel_text}")
        else:
            self.logger.info(f"🔀 Parallel Group: {group_name}")
            self.logger.info(f"   Steps: {', '.join(step_ids)}")
    
    def log_parallel_complete(self, group_name: str, elapsed_ms: int):
        """Log parallel execution completion."""
        if self.use_rich:
            parallel_text = f"[bold green]Parallel Group Complete:[/bold green] {group_name} "
            parallel_text += f"[dim]({elapsed_ms}ms)[/dim]"
            self.console.print(f"✅ {parallel_text}")
        else:
            self.logger.info(f"✅ Parallel Group Complete: {group_name} ({elapsed_ms}ms)")
    
    def log_context_update(self, key: str, value: Any):
        """Log context update."""
        if self.debug:
            if self.use_rich:
                self.console.print(f"📝 Context: [cyan]{key}[/cyan] = {repr(value)[:100]}...")
            else:
                self.logger.debug(f"📝 Context: {key} = {repr(value)[:100]}...")
    
    def log_secret_access(self, step_id: str, secret_names: List[str]):
        """Log secret access."""
        if self.debug:
            if self.use_rich:
                secrets_text = f"[bold cyan]Secrets:[/bold cyan] {', '.join(secret_names)}"
                secrets_text += f" [dim](step: {step_id})[/dim]"
                self.console.print(f"🔐 {secrets_text}")
            else:
                self.logger.debug(f"🔐 Secrets: {', '.join(secret_names)} (step: {step_id})")
    
    def log_thread_handoff(self, step_id: str, thread_id: str):
        """Log thread handoff."""
        if self.debug:
            if self.use_rich:
                thread_text = f"[bold cyan]Thread Handoff:[/bold cyan] {thread_id}"
                thread_text += f" [dim](step: {step_id})[/dim]"
                self.console.print(f"🧵 {thread_text}")
            else:
                self.logger.debug(f"🧵 Thread Handoff: {thread_id} (step: {step_id})")
    
    def print_run_summary(self, runs: List[Dict[str, Any]]):
        """Print run history summary."""
        if not runs:
            self.logger.info("No workflow runs found")
            return
        
        if self.use_rich:
            # Rich table for run history
            table = Table(
                title="Workflow Run History",
                show_header=True,
                header_style="bold magenta"
            )
            
            table.add_column("Run ID", style="cyan", width=12)
            table.add_column("Workflow", style="blue")
            table.add_column("Status", style="green")
            table.add_column("Duration", style="yellow")
            table.add_column("Started", style="dim")
            
            for run in runs:
                status_style = "green" if run['status'] == 'completed' else "red"
                duration = f"{run.get('duration', 0)}ms" if run.get('duration') else "N/A"
                started = datetime.fromtimestamp(run['started_at']).strftime('%m-%d %H:%M')
                
                table.add_row(
                    run['id'][:8],
                    run['workflow_name'],
                    f"[{status_style}]{run['status']}[/{status_style}]",
                    duration,
                    started
                )
            
            self.console.print(table)
        else:
            # Plain text table
            print("\nWorkflow Run History:")
            print("-" * 80)
            print(f"{'Run ID':<12} {'Workflow':<20} {'Status':<12} {'Duration':<12} {'Started':<15}")
            print("-" * 80)
            
            for run in runs:
                duration = f"{run.get('duration', 0)}ms" if run.get('duration') else "N/A"
                started = datetime.fromtimestamp(run['started_at']).strftime('%m-%d %H:%M')
                
                print(f"{run['id'][:8]:<12} {run['workflow_name']:<20} {run['status']:<12} {duration:<12} {started:<15}")


# Global logger instance
_workflow_logger: Optional[WorkflowLogger] = None


def get_workflow_logger(debug: bool = False, use_rich: bool = True) -> WorkflowLogger:
    """Get the global workflow logger instance."""
    global _workflow_logger
    if _workflow_logger is None:
        _workflow_logger = WorkflowLogger(debug=debug, use_rich=use_rich)
    return _workflow_logger


def set_workflow_logger(logger: WorkflowLogger):
    """Set the global workflow logger instance."""
    global _workflow_logger
    _workflow_logger = logger


# Convenience functions
def log_workflow_start(workflow_name: str, run_id: str, inputs: Dict[str, Any], debug: bool = False):
    """Log workflow start."""
    logger = get_workflow_logger(debug=debug)
    logger.print_workflow_banner(workflow_name, run_id, inputs)


def log_workflow_success(workflow_name: str, run_id: str, outputs: Dict[str, Any], elapsed_ms: int, debug: bool = False):
    """Log workflow success."""
    logger = get_workflow_logger(debug=debug)
    logger.print_workflow_completion(workflow_name, run_id, outputs, elapsed_ms)


def log_workflow_error(workflow_name: str, run_id: str, error: Exception, elapsed_ms: int, debug: bool = False):
    """Log workflow error."""
    logger = get_workflow_logger(debug=debug)
    logger.print_workflow_error(workflow_name, run_id, error, elapsed_ms)


def log_step_start(step_id: str, step_type: str, config: Dict[str, Any] = None, debug: bool = False):
    """Log step start."""
    logger = get_workflow_logger(debug=debug)
    logger.log_step_start(step_id, step_type, config)


def log_step_success(step_id: str, result: Any = None, debug: bool = False):
    """Log step success."""
    logger = get_workflow_logger(debug=debug)
    logger.log_step_success(step_id, result)


def log_step_failure(step_id: str, error: Exception, debug: bool = False):
    """Log step failure."""
    logger = get_workflow_logger(debug=debug)
    logger.log_step_failure(step_id, error)


def log_step_retry(step_id: str, attempt: int, delay_ms: int, error: Exception, debug: bool = False):
    """Log step retry."""
    logger = get_workflow_logger(debug=debug)
    logger.log_step_retry(step_id, attempt, delay_ms, error) 