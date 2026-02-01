"""
CLI module for LangPy workflow commands.

Provides command-line interface for workflow operations matching 
Langbase's workflow CLI: `python -m workflow run path/to/file.py --debug`
"""

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import logging

# Get project root
project_root = Path(__file__).parent.parent

try:
    from .core import WorkflowEngine, get_workflow_engine
    from .run_registry import get_run_registry
    from .logging import get_workflow_logger
    from .exceptions import WorkflowError
except ImportError as e:
    print(f"Error importing workflow modules: {e}")
    sys.exit(1)


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_workflow_file(file_path: str) -> Dict[str, Any]:
    """Load and execute a workflow file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Workflow file not found: {file_path}")
    
    if not file_path.suffix == '.py':
        raise ValueError(f"Workflow file must be a Python file: {file_path}")
    
    # Load the module
    spec = importlib.util.spec_from_file_location("workflow_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load workflow file: {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    
    # Add project root to path for imports
    sys.path.insert(0, str(project_root))
    
    # Execute the module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error executing workflow file: {e}")
    
    # Look for workflow function or steps
    workflow_function = None
    workflow_steps = []
    
    # Check for decorated workflow function
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and hasattr(obj, 'workflow_name'):
            workflow_function = obj
            break
    
    # Check for steps list
    if hasattr(module, 'steps'):
        workflow_steps = module.steps
    
    # Check for workflow configuration
    workflow_config = getattr(module, 'workflow_config', {})
    
    return {
        'function': workflow_function,
        'steps': workflow_steps,
        'config': workflow_config,
        'module': module
    }


async def run_workflow_file(file_path: str, 
                          inputs: Optional[Dict[str, Any]] = None,
                          debug: bool = False) -> Dict[str, Any]:
    """Run a workflow from a file."""
    # Load the workflow file
    workflow_data = load_workflow_file(file_path)
    
    # Create workflow engine
    engine = get_workflow_engine(debug=debug)
    
    # Get inputs
    inputs = inputs or {}
    
    # Execute workflow
    if workflow_data['function']:
        # Execute decorated workflow function
        result = await workflow_data['function'](**inputs)
    elif workflow_data['steps']:
        # Execute step list
        workflow_name = Path(file_path).stem
        result = await engine.run(
            name=workflow_name,
            inputs=inputs,
            steps=workflow_data['steps']
        )
    else:
        raise WorkflowError(f"No workflow function or steps found in {file_path}")
    
    return result


def run_command(args: argparse.Namespace) -> int:
    """Execute the run command."""
    try:
        # Setup logging
        setup_logging(args.debug)
        
        # Parse inputs
        inputs = {}
        if args.inputs:
            try:
                inputs = json.loads(args.inputs)
            except json.JSONDecodeError:
                # Try to parse as key=value pairs
                for pair in args.inputs.split(','):
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        inputs[key.strip()] = value.strip()
                    else:
                        inputs[pair.strip()] = True
        
        # Run the workflow
        result = asyncio.run(run_workflow_file(
            args.file,
            inputs=inputs,
            debug=args.debug
        ))
        
        # Output result
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"✅ Workflow result saved to: {output_path}")
        else:
            print(f"✅ Workflow completed successfully")
            if args.debug:
                print(f"Result: {json.dumps(result, indent=2, default=str)}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def list_command(args: argparse.Namespace) -> int:
    """Execute the list command."""
    try:
        # Setup logging
        setup_logging(args.debug)
        
        # Get run registry
        registry = get_run_registry()
        
        # Get workflow logger for formatting
        logger = get_workflow_logger(debug=args.debug)
        
        # List runs
        runs = registry.list_runs()
        
        if not runs:
            print("No workflow runs found")
            return 0
        
        # Filter by workflow name if specified
        if args.workflow:
            runs = [run for run in runs if run.workflow_name == args.workflow]
        
        # Filter by status if specified
        if args.status:
            runs = [run for run in runs if run.status == args.status]
        
        # Limit results
        if args.limit:
            runs = runs[:args.limit]
        
        # Convert to dict format for display
        runs_dict = [run.to_dict() for run in runs]
        
        # Display results
        logger.print_run_summary(runs_dict)
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to list runs: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def stats_command(args: argparse.Namespace) -> int:
    """Execute the stats command."""
    try:
        # Setup logging
        setup_logging(args.debug)
        
        # Get run registry
        registry = get_run_registry()
        
        # Get stats
        stats = registry.get_workflow_stats(args.workflow)
        
        if not stats:
            print("No workflow runs found")
            return 0
        
        # Display stats
        print(f"\n📊 Workflow Statistics")
        if args.workflow:
            print(f"Workflow: {args.workflow}")
        
        print(f"{'='*50}")
        print(f"Total Runs: {stats['total_runs']}")
        print(f"Completed: {stats['completed_runs']}")
        print(f"Failed: {stats['failed_runs']}")
        print(f"Running: {stats['running_runs']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        
        if stats['avg_duration_ms']:
            print(f"Average Duration: {stats['avg_duration_ms']:.0f}ms")
        
        if stats.get('first_run_date'):
            print(f"First Run: {stats['first_run_date']}")
        
        if stats.get('last_run_date'):
            print(f"Last Run: {stats['last_run_date']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to get stats: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def cleanup_command(args: argparse.Namespace) -> int:
    """Execute the cleanup command."""
    try:
        # Setup logging
        setup_logging(args.debug)
        
        # Get run registry
        registry = get_run_registry()
        
        # Cleanup old runs
        deleted_count = registry.cleanup_old_runs(
            days_old=args.days,
            keep_failed=not args.include_failed
        )
        
        print(f"✅ Cleaned up {deleted_count} old workflow runs")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to cleanup runs: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='workflow',
        description='LangPy Workflow CLI - Execute and manage workflows'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a workflow from a file')
    run_parser.add_argument(
        'file',
        help='Path to the workflow Python file'
    )
    run_parser.add_argument(
        '--inputs', '-i',
        help='Input data as JSON string or key=value pairs'
    )
    run_parser.add_argument(
        '--output', '-o',
        help='Output file to save results'
    )
    run_parser.set_defaults(func=run_command)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List workflow runs')
    list_parser.add_argument(
        '--workflow', '-w',
        help='Filter by workflow name'
    )
    list_parser.add_argument(
        '--status', '-s',
        choices=['running', 'completed', 'failed'],
        help='Filter by status'
    )
    list_parser.add_argument(
        '--limit', '-l',
        type=int,
        help='Limit number of results'
    )
    list_parser.set_defaults(func=list_command)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show workflow statistics')
    stats_parser.add_argument(
        'workflow',
        nargs='?',
        help='Workflow name (optional)'
    )
    stats_parser.set_defaults(func=stats_command)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old workflow runs')
    cleanup_parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Delete runs older than this many days (default: 30)'
    )
    cleanup_parser.add_argument(
        '--include-failed',
        action='store_true',
        help='Also delete failed runs'
    )
    cleanup_parser.set_defaults(func=cleanup_command)
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    return args.func(args)


# Alias for SDK compatibility
cli_main = main


if __name__ == '__main__':
    sys.exit(main()) 