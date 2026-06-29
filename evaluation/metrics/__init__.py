"""Metrics subpackage for thesis results evaluation."""

from evaluation.metrics.detection import compute_detection_metrics
from evaluation.metrics.dynamics import compute_dynamics_metrics
from evaluation.metrics.efficiency import compute_capacity_frontier, compute_efficiency_metrics
from evaluation.metrics.pose2d import compute_pose2d_metrics
from evaluation.metrics.pose3d import compute_pose3d_metrics

__all__ = [
    "compute_detection_metrics",
    "compute_pose2d_metrics",
    "compute_pose3d_metrics",
    "compute_dynamics_metrics",
    "compute_efficiency_metrics",
    "compute_capacity_frontier",
]
