"""
Analytics Dashboard for SMART-TRAIN platform.

This module provides comprehensive analytics and visualization capabilities
for monitoring model performance, user engagement, and system health.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

from ..core.base import BaseProcessor
from ..core.logging import get_logger
from ..compliance.audit_trail import AuditTrailManager, AuditEventType, AuditSeverity

logger=get_logger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""
    PERFORMANCE="performance"
    USAGE="usage"
    QUALITY="quality"
    COMPLIANCE="compliance"
    SYSTEM="system"
    BUSINESS="business"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    tags: Dict[str, str]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DashboardConfig:
    """Configuration for analytics dashboard."""
    refresh_interval_seconds: int=30
    data_retention_days: int=90
    enable_real_time_updates: bool=True
    enable_alerts: bool=True
    max_data_points: int=10000

    # Visualization settings
    chart_theme: str="plotly_white"
    color_palette: List[str] = None

    # Performance thresholds
    response_time_threshold_ms: float=100.0
    accuracy_threshold: float=0.85
    compliance_threshold: float=0.9

    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette=[
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
            ]


class MetricsCollector:
    """
    Collects and stores metrics from various system components.
    """

    def __init__(self, config: DashboardConfig):
        self.config=config
        self.metrics_storage: List[Metric] = []
        self.audit_manager=AuditTrailManager()

        # Prometheus metrics
        self.registry=CollectorRegistry()
        self._setup_prometheus_metrics()

        logger.info("Metrics collector initialized")

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics collectors."""
        self.request_counter=Counter(
            'smart_train_requests_total',
            'Total number of requests',
            ['endpoint', 'method', 'status'],
            registry=self.registry
        )

        self.response_time_histogram=Histogram(
            'smart_train_response_time_seconds',
            'Response time in seconds',
            ['endpoint'],
            registry=self.registry
        )

        self.model_accuracy_gauge=Gauge(
            'smart_train_model_accuracy',
            'Current model accuracy',
            ['model_name'],
            registry=self.registry
        )

        self.active_users_gauge=Gauge(
            'smart_train_active_users',
            'Number of active users',
            registry=self.registry
        )

    def collect_metric(self, name: str, value: float, metric_type: MetricType,
                      tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Collect a metric data point."""
        metric=Metric(
            name=name,
            value=value,
            _timestamp=datetime.now(),
            metric_type=metric_type,
            tags=tags or {},
            metadata=metadata
        )

        self.metrics_storage.append(metric)

        # Update Prometheus metrics
        self._update_prometheus_metrics(metric)

        # Cleanup old metrics
        self._cleanup_old_metrics()

    def _update_prometheus_metrics(self, metric: Metric):
        """Update Prometheus metrics based on collected metric."""
        if metric.name== "request_count":
            self.request_counter.labels(
                endpoint=metric.tags.get("endpoint", "unknown"),
                method=metric.tags.get("method", "unknown"),
                status=metric.tags.get("status", "unknown")
            ).inc()

        elif metric.name== "response_time":
            self.response_time_histogram.labels(
                endpoint=metric.tags.get("endpoint", "unknown")
            ).observe(metric.value / 1000)  # Convert ms to seconds

        elif metric.name== "model_accuracy":
            self.model_accuracy_gauge.labels(
                model_name=metric.tags.get("model_name", "unknown")
            ).set(metric.value)

        elif metric.name== "active_users":
            self.active_users_gauge.set(metric.value)

    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time=datetime.now() - timedelta(days=self.config.data_retention_days)
        self.metrics_storage=[
            m for m in self.metrics_storage
            if m.timestamp > cutoff_time
        ]

        # Limit total data points
        if len(self.metrics_storage) > self.config.max_data_points:
            self.metrics_storage=self.metrics_storage[-self.config.max_data_points:]

    def get_metrics(self, metric_type: Optional[MetricType] = None,
                   time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Metric]:
        """Retrieve metrics with optional filtering."""
        metrics=self.metrics_storage

        if metric_type:
            metrics=[m for m in metrics if m.metric_type== metric_type]

        if time_range:
            start_time, end_time=time_range
            metrics=[
                m for m in metrics
                if start_time <= m.timestamp <= end_time
            ]

        return metrics

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected metrics."""
        if not self.metrics_storage:
            return {"message": "No metrics available"}

        df=pd.DataFrame([asdict(m) for m in self.metrics_storage])

        summary={
            "total_metrics": len(self.metrics_storage),
            "metric_types": df['metric_type'].value_counts().to_dict(),
            "time_range": {
                "start": df['timestamp'].min(),
                "end": df['timestamp'].max()
            },
            "top_metrics": df['name'].value_counts().head(10).to_dict()
        }

        return summary


class AnalyticsDashboard:
    """
    Interactive analytics dashboard for SMART-TRAIN platform.

    Provides real-time visualization of system metrics, model performance,
    and business analytics.
    """

    def __init__(self, config: DashboardConfig, metrics_collector: MetricsCollector):
        self.config=config
        self.metrics_collector=metrics_collector
        self.audit_manager=AuditTrailManager()

        logger.info("Analytics dashboard initialized")

    def create_dashboard(self):
        """Create Streamlit dashboard interface."""
        st.set_page_config(
            page_title="SMART-TRAIN Analytics",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🏥 SMART-TRAIN Analytics Dashboard")
        st.markdown("Real-time monitoring and analytics for medical AI training platform")

        # Sidebar controls
        self._create_sidebar()

        # Main dashboard content
        self._create_main_dashboard()

    def _create_sidebar(self):
        """Create dashboard sidebar with controls."""
        st.sidebar.header("Dashboard Controls")

        # Time range selector
        time_range=st.sidebar.selectbox(
            "Time Range",
            ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom"]
        )

        # Metric type filter
        metric_types=st.sidebar.multiselect(
            "Metric Types",
            [mt.value for mt in MetricType],
            default=[MetricType.PERFORMANCE.value, MetricType.USAGE.value]
        )

        # Auto-refresh toggle
        auto_refresh=st.sidebar.checkbox(
            "Auto Refresh",
            value=self.config.enable_real_time_updates
        )

        if auto_refresh:
            st.sidebar.info(f"Refreshing every {self.config.refresh_interval_seconds}s")
            time.sleep(self.config.refresh_interval_seconds)
            st.experimental_rerun()

        # Manual refresh button
        if st.sidebar.button("Refresh Now"):
            st.experimental_rerun()

        return time_range, metric_types

    def _create_main_dashboard(self):
        """Create main dashboard content."""
        # Key metrics row
        self._create_key_metrics_row()

        # Charts row
        col1, col2=st.columns(2)

        with col1:
            self._create_performance_chart()
            self._create_usage_analytics_chart()

        with col2:
            self._create_model_performance_chart()
            self._create_compliance_chart()

        # Detailed tables
        self._create_detailed_metrics_table()

        # System health section
        self._create_system_health_section()

    def _create_key_metrics_row(self):
        """Create key metrics overview row."""
        st.subheader("📊 Key Metrics")

        col1, col2, col3, col4=st.columns(4)

        # Get recent metrics
        recent_metrics=self.metrics_collector.get_metrics(
            time_range=(datetime.now() - timedelta(hours=1), datetime.now())
        )

        with col1:
            # Active users
            active_users=len(set(m.tags.get("user_id", "") for m in recent_metrics if m.tags.get("user_id")))
            st.metric("Active Users", active_users, delta="+5")

        with col2:
            # Average response time
            response_times=[m.value for m in recent_metrics if m.name== "response_time"]
            avg_response_time=np.mean(response_times) if response_times else 0
            st.metric("Avg Response Time", f"{avg_response_time:.1f}ms", delta="-2.3ms")

        with col3:
            # Model accuracy
            accuracy_metrics=[m.value for m in recent_metrics if m.name== "model_accuracy"]
            current_accuracy=accuracy_metrics[-1] if accuracy_metrics else 0
            st.metric("Model Accuracy", f"{current_accuracy:.1%}", delta="+0.2%")

        with col4:
            # System uptime
            st.metric("System Uptime", "99.9%", delta="+0.1%")

    def _create_performance_chart(self):
        """Create performance metrics chart."""
        st.subheader("⚡ Performance Metrics")

        # Get performance metrics
        performance_metrics=self.metrics_collector.get_metrics(MetricType.PERFORMANCE)

        if performance_metrics:
            df=pd.DataFrame([
                {
                    "timestamp": m.timestamp,
                    "response_time": m.value if m.name== "response_time" else None,
                    "throughput": m.value if m.name== "throughput" else None
                }
                for m in performance_metrics
            ])

            # Create subplot
            fig=make_subplots(
                rows=2, cols=1,
                subplot_titles=["Response Time (ms)", "Throughput (req/s)"],
                vertical_spacing=0.1
            )

            # Response time chart
            if not df["response_time"].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["response_time"],
                        mode="lines+markers",
                        name="Response Time",
                        line=dict(color=self.config.color_palette[0])
                    ),
                    row=1, col=1
                )

            # Throughput chart
            if not df["throughput"].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["throughput"],
                        mode="lines+markers",
                        name="Throughput",
                        line=dict(color=self.config.color_palette[1])
                    ),
                    row=2, col=1
                )

            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance metrics available")

    def _create_usage_analytics_chart(self):
        """Create usage analytics chart."""
        st.subheader("📈 Usage Analytics")

        # Mock usage data for demonstration
        usage_data={
            "Training Sessions": 1250,
            "API Calls": 15600,
            "Model Inferences": 8900,
            "Data Processed (GB)": 45.2
        }

        fig=go.Figure(data=[
            go.Bar(
                x=list(usage_data.keys()),
                y=list(usage_data.values()),
                marker_color=self.config.color_palette[:len(usage_data)]
            )
        ])

        fig.update_layout(
            title="Usage Statistics (Last 24 Hours)",
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

    def _create_model_performance_chart(self):
        """Create model performance chart."""
        st.subheader("🤖 Model Performance")

        # Mock model performance data
        models=["CPR Quality", "Pose Analysis", "Real-time Feedback", "Quality Scoring"]
        accuracy_scores=[0.94, 0.91, 0.89, 0.92]

        fig=go.Figure(data=[
            go.Bar(
                x=models,
                y=accuracy_scores,
                marker_color=self.config.color_palette[2],
                text=[f"{score:.1%}" for score in accuracy_scores],
                textposition="auto"
            )
        ])

        fig.update_layout(
            title="Model Accuracy Scores",
            yaxis_title="Accuracy",
            height=300,
            yaxis=dict(range=[0.8, 1.0])
        )

        st.plotly_chart(fig, use_container_width=True)

    def _create_compliance_chart(self):
        """Create compliance metrics chart."""
        st.subheader("🏥 Medical Compliance")

        # Mock compliance data
        compliance_metrics={
            "ISO 13485": 0.98,
            "IEC 62304": 0.96,
            "HIPAA": 0.99,
            "GDPR": 0.97
        }

        fig=go.Figure(data=[
            go.Bar(
                x=list(compliance_metrics.keys()),
                y=list(compliance_metrics.values()),
                marker_color=self.config.color_palette[3],
                text=[f"{score:.1%}" for score in compliance_metrics.values()],
                textposition="auto"
            )
        ])

        fig.update_layout(
            title="Compliance Scores",
            yaxis_title="Compliance Rate",
            height=300,
            yaxis=dict(range=[0.9, 1.0])
        )

        st.plotly_chart(fig, use_container_width=True)

    def _create_detailed_metrics_table(self):
        """Create detailed metrics table."""
        st.subheader("📋 Detailed Metrics")

        # Get recent metrics
        recent_metrics=self.metrics_collector.get_metrics(
            time_range=(datetime.now() - timedelta(hours=24), datetime.now())
        )

        if recent_metrics:
            df=pd.DataFrame([
                {
                    "Timestamp": m.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "Metric": m.name,
                    "Value": m.value,
                    "Type": m.metric_type.value,
                    "Tags": ", ".join([f"{k}:{v}" for k, v in m.tags.items()])
                }
                for m in recent_metrics[-100:]  # Show last 100 metrics
            ])

            st.dataframe(df, use_container_width=True)
        else:
            st.info("No detailed metrics available")

    def _create_system_health_section(self):
        """Create system health monitoring section."""
        st.subheader("🔧 System Health")

        col1, col2, col3=st.columns(3)

        with col1:
            st.metric("CPU Usage", "45%", delta="-5%")
            st.metric("Memory Usage", "62%", delta="+3%")

        with col2:
            st.metric("Disk Usage", "78%", delta="+2%")
            st.metric("Network I/O", "125 MB/s", delta="+15 MB/s")

        with col3:
            st.metric("Error Rate", "0.1%", delta="-0.05%")
            st.metric("Availability", "99.95%", delta="+0.02%")

        # System alerts
        st.subheader("🚨 Recent Alerts")

        alerts=[
            {"time": "2 hours ago", "level": "INFO", "message": "Model retrained successfully"},
            {"time": "6 hours ago", "level": "WARNING", "message": "High memory usage detected"},
            {"time": "1 day ago", "level": "INFO", "message": "System maintenance completed"}
        ]

        for alert in alerts:
            level_color={
                "INFO": "blue",
                "WARNING": "orange",
                "ERROR": "red"
            }.get(alert["level"], "gray")

            st.markdown(
                f":{level_color}[{alert['level']}] {alert['time']}: {alert['message']}"
            )

    def generate_report(self, time_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        start_time, end_time=time_range
        metrics=self.metrics_collector.get_metrics(time_range=time_range)

        if not metrics:
            return {"message": "No data available for the specified time range"}

        df=pd.DataFrame([asdict(m) for m in metrics])

        report={
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "total_metrics": len(metrics)
            },
            "performance_summary": {
                "avg_response_time_ms": df[df['name'] == 'response_time']['value'].mean(),
                "max_response_time_ms": df[df['name'] == 'response_time']['value'].max(),
                "total_requests": len(df[df['name'] == 'request_count']),
                "error_rate": 0.001  # Mock error rate
            },
            "usage_summary": {
                "unique_users": len(df[df['name'] == 'user_session']['tags'].unique()),
                "total_training_sessions": len(df[df['name'] == 'training_session']),
                "data_processed_gb": df[df['name'] == 'data_processed']['value'].sum()
            },
            "model_performance": {
                "avg_accuracy": df[df['name'] == 'model_accuracy']['value'].mean(),
                "inference_count": len(df[df['name'] == 'model_inference']),
                "avg_inference_time_ms": df[df['name'] == 'inference_time']['value'].mean()
            },
            "compliance_status": {
                "audit_events": len(df[df['metric_type'] == 'compliance']),
                "compliance_violations": 0,
                "data_privacy_score": 0.99
            }
        }

        return report


def create_sample_dashboard():
    """Create a sample dashboard with mock data for demonstration."""
    config=DashboardConfig()
    metrics_collector=MetricsCollector(config)

    # Generate sample metrics
    for i in range(100):
        _timestamp=datetime.now() - timedelta(minutes=i)

        # Performance metrics
        metrics_collector.collect_metric(
            "response_time",
            np.random.normal(75, 15),
            MetricType.PERFORMANCE,
            tags={"endpoint": "/api/v1/analyze"}
        )

        # Usage metrics
        metrics_collector.collect_metric(
            "active_users",
            np.random.poisson(50),
            MetricType.USAGE
        )

        # Model performance
        metrics_collector.collect_metric(
            "model_accuracy",
            np.random.normal(0.92, 0.02),
            MetricType.QUALITY,
            tags={"model_name": "cpr_quality"}
        )

    dashboard=AnalyticsDashboard(config, metrics_collector)
    return dashboard


if __name__== "__main__":
    # Run sample dashboard
    dashboard=create_sample_dashboard()
    dashboard.create_dashboard()
