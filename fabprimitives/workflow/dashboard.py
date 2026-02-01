"""
Streamlit Dashboard for LangPy Workflow Monitoring

Provides a web-based dashboard for visualizing workflow runs, history,
and performance metrics with interactive filtering and drill-down capabilities.
"""

import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    px = None
    go = None

try:
    from .run_registry import RunRegistry, RunFilter
    from .logging import WorkflowLogger
except ImportError:
    # Fallback imports for standalone execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from run_registry import RunRegistry, RunFilter
    
    # Import from our custom logging module (not standard library)
    import importlib.util
    import os
    
    # Load our custom logging module explicitly to avoid conflicts
    logging_path = os.path.join(Path(__file__).parent, 'logging.py')
    spec = importlib.util.spec_from_file_location("workflow_logging", logging_path)
    logging_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(logging_module)
    WorkflowLogger = logging_module.WorkflowLogger


class WorkflowDashboard:
    """Streamlit dashboard for workflow monitoring."""
    
    def __init__(self):
        self.registry = RunRegistry()
        self.logger = WorkflowLogger(debug=False)
        
        # Check if plotly is available
        if not HAS_PLOTLY:
            print("⚠️  Warning: plotly not available. Charts will be disabled.")
            print("   Install plotly with: pip install plotly")
        
        # Configure Streamlit
        st.set_page_config(
            page_title="LangPy Workflow Dashboard",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Main dashboard application."""
        st.title("🚀 LangPy Workflow Dashboard")
        st.markdown("Monitor and analyze your workflow executions with interactive visualizations.")
        
        # Sidebar filters
        self._render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "📋 Run History", 
            "📈 Analytics", 
            "🔍 Run Details"
        ])
        
        with tab1:
            self._render_overview()
        
        with tab2:
            self._render_run_history()
        
        with tab3:
            self._render_analytics()
        
        with tab4:
            self._render_run_details()
    
    def _render_sidebar(self):
        """Render sidebar with filters and controls."""
        st.sidebar.title("🔧 Filters & Controls")
        
        # Auto-refresh
        if st.sidebar.checkbox("Auto-refresh", value=False):
            st.sidebar.info("Refreshing every 10 seconds...")
            time.sleep(10)
            st.rerun()
        
        # Manual refresh
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()
        
        # Workflow filter
        workflows = self._get_unique_workflows()
        workflow_filter = st.sidebar.selectbox(
            "Workflow",
            ["All"] + workflows,
            index=0
        )
        st.session_state.workflow_filter = None if workflow_filter == "All" else workflow_filter
        
        # Status filter
        status_filter = st.sidebar.selectbox(
            "Status",
            ["All", "completed", "failed", "running"],
            index=0
        )
        st.session_state.status_filter = None if status_filter == "All" else status_filter
        
        # Time range filter
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["All", "Last 24 hours", "Last 7 days", "Last 30 days"],
            index=1
        )
        st.session_state.time_range = time_range
        
        # Limit
        limit = st.sidebar.number_input(
            "Max Results",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )
        st.session_state.limit = limit
        
        # Database info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Database Info")
        total_runs = len(self.registry.list_runs())
        st.sidebar.metric("Total Runs", total_runs)
        
        # Cleanup controls
        st.sidebar.markdown("### 🧹 Cleanup")
        if st.sidebar.button("Clean Old Runs (30+ days)"):
            deleted = self.registry.cleanup_old_runs(days_old=30)
            st.sidebar.success(f"Deleted {deleted} old runs")
    
    def _render_overview(self):
        """Render overview dashboard."""
        st.header("📊 Workflow Overview")
        
        # Get filtered runs
        runs = self._get_filtered_runs()
        
        if not runs:
            st.warning("No runs found for the selected filters.")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([run.to_dict() for run in runs])
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Runs", len(df))
        
        with col2:
            completed = len(df[df['status'] == 'completed'])
            success_rate = (completed / len(df)) * 100 if len(df) > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        with col3:
            if 'duration_ms' in df.columns:
                avg_duration = df['duration_ms'].mean()
                st.metric("Avg Duration", f"{avg_duration:.0f}ms" if pd.notna(avg_duration) else "N/A")
            else:
                st.metric("Avg Duration", "N/A")
        
        with col4:
            failed = len(df[df['status'] == 'failed'])
            st.metric("Failed Runs", failed)
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            # Status distribution
            st.subheader("Status Distribution")
            status_counts = df['status'].value_counts()
            
            if HAS_PLOTLY:
                fig = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Workflow Status Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("📊 Status Distribution (Chart disabled - plotly not installed)")
                st.dataframe(status_counts)
        
        with col2:
            # Workflow distribution
            st.subheader("Workflow Distribution")
            workflow_counts = df['workflow_name'].value_counts()
            
            if HAS_PLOTLY:
                fig = px.bar(
                    x=workflow_counts.index,
                    y=workflow_counts.values,
                    title="Runs by Workflow"
                )
                fig.update_layout(xaxis_title="Workflow", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("📊 Workflow Distribution (Chart disabled - plotly not installed)")
                st.dataframe(workflow_counts)
        
        # Timeline
        st.subheader("Execution Timeline")
        if 'started_at' in df.columns and len(df) > 0:
            df['started_datetime'] = pd.to_datetime(df['started_at'], unit='s')
            df['hour'] = df['started_datetime'].dt.floor('H')
            
            timeline = df.groupby(['hour', 'status']).size().reset_index(name='count')
            
            fig = px.bar(
                timeline,
                x='hour',
                y='count',
                color='status',
                title="Workflow Executions Over Time"
            )
            fig.update_layout(xaxis_title="Time", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No timing data available for timeline visualization.")
    
    def _render_run_history(self):
        """Render run history table."""
        st.header("📋 Run History")
        
        # Get filtered runs
        runs = self._get_filtered_runs()
        
        if not runs:
            st.warning("No runs found for the selected filters.")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([run.to_dict() for run in runs])
        
        # Add formatted columns
        if 'started_at' in df.columns:
            df['Started'] = pd.to_datetime(df['started_at'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        if 'duration_ms' in df.columns:
            df['Duration'] = df['duration_ms'].apply(lambda x: f"{x:.0f}ms" if pd.notna(x) else "N/A")
        
        # Select columns for display
        display_columns = ['id', 'workflow_name', 'status', 'Started', 'Duration']
        available_columns = [col for col in display_columns if col in df.columns]
        
        # Display table
        st.dataframe(
            df[available_columns].rename(columns={
                'id': 'Run ID',
                'workflow_name': 'Workflow',
                'status': 'Status'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Export functionality
        st.markdown("### 💾 Export")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export to CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"workflow_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export to JSON"):
                json_data = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"workflow_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    def _render_analytics(self):
        """Render analytics dashboard."""
        st.header("📈 Analytics")
        
        # Get filtered runs
        runs = self._get_filtered_runs()
        
        if not runs:
            st.warning("No runs found for the selected filters.")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([run.to_dict() for run in runs])
        
        # Performance metrics
        st.subheader("⚡ Performance Metrics")
        
        if 'duration_ms' in df.columns and len(df) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_duration = df['duration_ms'].mean()
                st.metric("Average Duration", f"{avg_duration:.0f}ms" if pd.notna(avg_duration) else "N/A")
            
            with col2:
                median_duration = df['duration_ms'].median()
                st.metric("Median Duration", f"{median_duration:.0f}ms" if pd.notna(median_duration) else "N/A")
            
            with col3:
                max_duration = df['duration_ms'].max()
                st.metric("Max Duration", f"{max_duration:.0f}ms" if pd.notna(max_duration) else "N/A")
            
            # Duration distribution
            st.subheader("Duration Distribution")
            fig = px.histogram(
                df,
                x='duration_ms',
                nbins=20,
                title="Duration Distribution"
            )
            fig.update_layout(xaxis_title="Duration (ms)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
            
            # Duration by workflow
            st.subheader("Duration by Workflow")
            fig = px.box(
                df,
                x='workflow_name',
                y='duration_ms',
                title="Duration Distribution by Workflow"
            )
            fig.update_layout(xaxis_title="Workflow", yaxis_title="Duration (ms)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Error analysis
        st.subheader("❌ Error Analysis")
        failed_runs = df[df['status'] == 'failed']
        
        if len(failed_runs) > 0:
            st.metric("Failed Runs", len(failed_runs))
            
            # Error details
            if 'error' in failed_runs.columns:
                error_summary = failed_runs['error'].value_counts().head(10)
                st.write("Top Errors:")
                st.dataframe(error_summary, use_container_width=True)
        else:
            st.success("No failed runs in the selected time range! 🎉")
        
        # Trend analysis
        st.subheader("📊 Trend Analysis")
        
        if 'started_at' in df.columns and len(df) > 0:
            df['started_datetime'] = pd.to_datetime(df['started_at'], unit='s')
            
            # Success rate over time
            df['date'] = df['started_datetime'].dt.date
            daily_stats = df.groupby('date').agg({
                'status': ['count', lambda x: (x == 'completed').sum()]
            }).reset_index()
            
            daily_stats.columns = ['date', 'total_runs', 'successful_runs']
            daily_stats['success_rate'] = (daily_stats['successful_runs'] / daily_stats['total_runs']) * 100
            
            fig = px.line(
                daily_stats,
                x='date',
                y='success_rate',
                title="Success Rate Trend",
                markers=True
            )
            fig.update_layout(xaxis_title="Date", yaxis_title="Success Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_run_details(self):
        """Render detailed run inspection."""
        st.header("🔍 Run Details")
        
        # Get filtered runs
        runs = self._get_filtered_runs()
        
        if not runs:
            st.warning("No runs found for the selected filters.")
            return
        
        # Select run for details
        run_options = {f"{run.workflow_name} - {run.id[:8]} ({run.status})": run for run in runs}
        selected_run_key = st.selectbox(
            "Select Run for Details",
            list(run_options.keys())
        )
        
        if selected_run_key:
            selected_run = run_options[selected_run_key]
            
            # Run metadata
            st.subheader("📋 Run Metadata")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Run ID:** {selected_run.id}")
                st.write(f"**Workflow:** {selected_run.workflow_name}")
                st.write(f"**Status:** {selected_run.status}")
                st.write(f"**Started:** {datetime.fromtimestamp(selected_run.started_at).strftime('%Y-%m-%d %H:%M:%S')}")
                
                if selected_run.completed_at:
                    st.write(f"**Completed:** {datetime.fromtimestamp(selected_run.completed_at).strftime('%Y-%m-%d %H:%M:%S')}")
                
                if selected_run.duration_ms:
                    st.write(f"**Duration:** {selected_run.duration_ms}ms")
            
            with col2:
                if selected_run.error:
                    st.error(f"**Error:** {selected_run.error}")
                else:
                    st.success("No errors recorded")
            
            # Inputs
            st.subheader("📥 Inputs")
            if selected_run.inputs:
                st.json(selected_run.inputs)
            else:
                st.info("No inputs recorded")
            
            # Outputs
            st.subheader("📤 Outputs")
            if selected_run.outputs:
                st.json(selected_run.outputs)
            else:
                st.info("No outputs recorded")
            
            # Context
            st.subheader("🔗 Context")
            if selected_run.context:
                st.json(selected_run.context)
            else:
                st.info("No context recorded")
            
            # Steps
            st.subheader("📋 Steps")
            if selected_run.steps:
                steps_df = pd.DataFrame(selected_run.steps)
                st.dataframe(steps_df, use_container_width=True)
            else:
                st.info("No step details recorded")
            
            # Raw data
            with st.expander("🔍 Raw Data"):
                st.json(selected_run.to_dict())
    
    def _get_filtered_runs(self) -> List:
        """Get runs based on current filters."""
        # Get filter values from session state
        workflow_filter = getattr(st.session_state, 'workflow_filter', None)
        status_filter = getattr(st.session_state, 'status_filter', None)
        time_range = getattr(st.session_state, 'time_range', 'All')
        limit = getattr(st.session_state, 'limit', 100)
        
        # Create filter object
        filter_obj = RunFilter(
            workflow_name=workflow_filter,
            status=status_filter,
            limit=limit
        )
        
        # Get runs
        runs = self.registry.list_runs(filter_obj)
        
        # Apply time range filter
        if time_range != 'All':
            cutoff_time = self._get_cutoff_time(time_range)
            runs = [run for run in runs if run.started_at >= cutoff_time]
        
        return runs
    
    def _get_cutoff_time(self, time_range: str) -> int:
        """Get cutoff timestamp for time range."""
        now = datetime.now()
        
        if time_range == "Last 24 hours":
            cutoff = now - timedelta(hours=24)
        elif time_range == "Last 7 days":
            cutoff = now - timedelta(days=7)
        elif time_range == "Last 30 days":
            cutoff = now - timedelta(days=30)
        else:
            cutoff = now - timedelta(days=365)  # Default to 1 year
        
        return int(cutoff.timestamp())
    
    def _get_unique_workflows(self) -> List[str]:
        """Get unique workflow names."""
        all_runs = self.registry.list_runs()
        return sorted(list(set(run.workflow_name for run in all_runs)))


def get_workflow_dashboard() -> WorkflowDashboard:
    """Get workflow dashboard instance."""
    return WorkflowDashboard()


def main():
    """Main application entry point."""
    dashboard = WorkflowDashboard()
    dashboard.run()


if __name__ == "__main__":
    main() 