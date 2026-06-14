"""Squat depth checker MVP package."""

from .depth import DepthResult, analyze_depth
from .pose import LandmarkFrame

__all__ = ["DepthResult", "LandmarkFrame", "analyze_depth"]
