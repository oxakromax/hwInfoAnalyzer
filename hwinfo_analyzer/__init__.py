"""
HWiNFO Analyzer
A comprehensive tool for analyzing HWiNFO CSV logs with thermal monitoring,
anomaly detection, and hardware diagnostics.
"""

__version__ = "1.0.0"
__author__ = "HWiNFO Analyzer Contributors"
__email__ = "contributors@hwinfo-analyzer.org"
__license__ = "MIT"

from .core.analyzer import HWInfoAnalyzer
from .core.data_processor import HWInfoDataProcessor
from .analysis.thermal_analyzer import ThermalAnalyzer
from .analysis.anomaly_detector import AnomalyDetector
from .visualization.visualizer import HWiNFOVisualizer
from .config.analysis_methods import AnalysisMethodSelector

__all__ = [
    "HWInfoAnalyzer",
    "HWInfoDataProcessor", 
    "ThermalAnalyzer",
    "AnomalyDetector",
    "HWiNFOVisualizer",
    "AnalysisMethodSelector"
]