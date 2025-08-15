"""
Configuration management for analysis methods and parameters.
"""

from .analysis_methods import AnalysisMethodSelector, AnalysisConfig
from .analysis_methods import create_default_config, create_quick_config, create_comprehensive_config

__all__ = [
    "AnalysisMethodSelector", 
    "AnalysisConfig",
    "create_default_config",
    "create_quick_config", 
    "create_comprehensive_config"
]