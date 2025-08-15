"""
Analysis Method Selector for HWiNFO
Configurable selection of analysis methods and visualization options
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import json
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

@dataclass
class AnalysisConfig:
    """Configuration for analysis methods"""
    # Anomaly detection methods
    enable_isolation_forest: bool = True
    enable_zscore: bool = True
    enable_iqr: bool = True
    
    # Thermal analysis
    enable_thermal_analysis: bool = True
    enable_peak_detection: bool = True
    enable_trend_analysis: bool = True
    
    # Voltage analysis
    enable_voltage_analysis: bool = True
    enable_voltage_stability: bool = True
    
    # Visualizations
    enable_all_plots: bool = True
    enable_temperature_trends: bool = True
    enable_distributions: bool = True
    enable_heatmaps: bool = True
    enable_voltage_plots: bool = True
    enable_anomaly_plots: bool = True
    enable_dashboard: bool = True
    enable_correlations: bool = True
    
    # Advanced configurations
    isolation_forest_contamination: float = 0.1
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    
    # Visualization configurations
    plot_resolution_dpi: int = 300
    max_sensors_per_plot: int = 6
    sample_rate_for_heatmap: int = 100

class AnalysisMethodSelector:
    """Analysis method selector and configurator"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.config = self._load_or_create_config()
    
    def _load_or_create_config(self) -> AnalysisConfig:
        """Load configuration from file or create default"""
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                return AnalysisConfig(**config_dict)
            except Exception as e:
                print(f"Error loading config: {e}. Using default configuration.")
        
        # Default configuration (ALL METHODS ENABLED)
        return AnalysisConfig()
    
    def save_config(self, filename: str = None):
        """Save current configuration to file"""
        config_file = filename or self.config_file or "analysis_config.json"
        
        config_dict = {
            # Anomaly methods
            'enable_isolation_forest': self.config.enable_isolation_forest,
            'enable_zscore': self.config.enable_zscore,
            'enable_iqr': self.config.enable_iqr,
            
            # Thermal analysis
            'enable_thermal_analysis': self.config.enable_thermal_analysis,
            'enable_peak_detection': self.config.enable_peak_detection,
            'enable_trend_analysis': self.config.enable_trend_analysis,
            
            # Voltage analysis
            'enable_voltage_analysis': self.config.enable_voltage_analysis,
            'enable_voltage_stability': self.config.enable_voltage_stability,
            
            # Visualizations
            'enable_all_plots': self.config.enable_all_plots,
            'enable_temperature_trends': self.config.enable_temperature_trends,
            'enable_distributions': self.config.enable_distributions,
            'enable_heatmaps': self.config.enable_heatmaps,
            'enable_voltage_plots': self.config.enable_voltage_plots,
            'enable_anomaly_plots': self.config.enable_anomaly_plots,
            'enable_dashboard': self.config.enable_dashboard,
            'enable_correlations': self.config.enable_correlations,
            
            # Advanced configurations
            'isolation_forest_contamination': self.config.isolation_forest_contamination,
            'zscore_threshold': self.config.zscore_threshold,
            'iqr_multiplier': self.config.iqr_multiplier,
            
            # Visualization configurations
            'plot_resolution_dpi': self.config.plot_resolution_dpi,
            'max_sensors_per_plot': self.config.max_sensors_per_plot,
            'sample_rate_for_heatmap': self.config.sample_rate_for_heatmap
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration saved to: {config_file}")
    
    def get_enabled_anomaly_methods(self) -> List[str]:
        """Returns list of enabled anomaly detection methods"""
        methods = []
        if self.config.enable_isolation_forest:
            methods.append('isolation_forest')
        if self.config.enable_zscore:
            methods.append('zscore')
        if self.config.enable_iqr:
            methods.append('iqr')
        return methods
    
    def get_enabled_analysis_methods(self) -> Dict[str, bool]:
        """Returns dictionary of enabled analysis methods"""
        return {
            'thermal_analysis': self.config.enable_thermal_analysis,
            'peak_detection': self.config.enable_peak_detection,
            'trend_analysis': self.config.enable_trend_analysis,
            'voltage_analysis': self.config.enable_voltage_analysis,
            'voltage_stability': self.config.enable_voltage_stability
        }
    
    def get_enabled_visualization_methods(self) -> Dict[str, bool]:
        """Returns dictionary of enabled visualization methods"""
        if self.config.enable_all_plots:
            return {
                'temperature_trends': True,
                'distributions': True,
                'heatmaps': True,
                'voltage_plots': True,
                'anomaly_plots': True,
                'dashboard': True,
                'correlations': True
            }
        
        return {
            'temperature_trends': self.config.enable_temperature_trends,
            'distributions': self.config.enable_distributions,
            'heatmaps': self.config.enable_heatmaps,
            'voltage_plots': self.config.enable_voltage_plots,
            'anomaly_plots': self.config.enable_anomaly_plots,
            'dashboard': self.config.enable_dashboard,
            'correlations': self.config.enable_correlations
        }
    
    def get_anomaly_parameters(self) -> Dict[str, Any]:
        """Returns parameters for anomaly detection algorithms"""
        return {
            'isolation_forest_contamination': self.config.isolation_forest_contamination,
            'zscore_threshold': self.config.zscore_threshold,
            'iqr_multiplier': self.config.iqr_multiplier
        }
    
    def get_visualization_parameters(self) -> Dict[str, Any]:
        """Returns parameters for visualizations"""
        return {
            'plot_resolution_dpi': self.config.plot_resolution_dpi,
            'max_sensors_per_plot': self.config.max_sensors_per_plot,
            'sample_rate_for_heatmap': self.config.sample_rate_for_heatmap
        }
    
    def enable_all_methods(self):
        """Enable all analysis methods"""
        self.config.enable_isolation_forest = True
        self.config.enable_zscore = True
        self.config.enable_iqr = True
        self.config.enable_thermal_analysis = True
        self.config.enable_peak_detection = True
        self.config.enable_trend_analysis = True
        self.config.enable_voltage_analysis = True
        self.config.enable_voltage_stability = True
        self.config.enable_all_plots = True
    
    def enable_minimal_analysis(self):
        """Enable only essential analysis (fast)"""
        self.config.enable_isolation_forest = True
        self.config.enable_zscore = False
        self.config.enable_iqr = False
        self.config.enable_thermal_analysis = True
        self.config.enable_peak_detection = False
        self.config.enable_trend_analysis = False
        self.config.enable_voltage_analysis = True
        self.config.enable_voltage_stability = False
        self.config.enable_all_plots = False
        self.config.enable_temperature_trends = True
        self.config.enable_dashboard = True
    
    def enable_comprehensive_analysis(self):
        """Enable complete and exhaustive analysis"""
        self.enable_all_methods()
        # More sensitive configurations for exhaustive analysis
        self.config.isolation_forest_contamination = 0.05
        self.config.zscore_threshold = 2.5
        self.config.iqr_multiplier = 1.2
    
    def create_preset_configs(self):
        """Create predefined configuration files"""
        presets = {
            'comprehensive': {
                'description': 'Complete analysis with all methods enabled',
                'config': AnalysisConfig()
            },
            'minimal': {
                'description': 'Fast analysis with essential methods',
                'config': AnalysisConfig(
                    enable_zscore=False,
                    enable_iqr=False,
                    enable_peak_detection=False,
                    enable_trend_analysis=False,
                    enable_voltage_stability=False,
                    enable_all_plots=False,
                    enable_temperature_trends=True,
                    enable_dashboard=True,
                    enable_distributions=False,
                    enable_heatmaps=False,
                    enable_voltage_plots=False,
                    enable_anomaly_plots=False,
                    enable_correlations=False
                )
            },
            'thermal_focus': {
                'description': 'Focus on thermal analysis',
                'config': AnalysisConfig(
                    enable_voltage_analysis=False,
                    enable_voltage_stability=False,
                    enable_voltage_plots=False,
                    zscore_threshold=2.0,
                    iqr_multiplier=1.2
                )
            },
            'voltage_focus': {
                'description': 'Focus on voltage analysis',
                'config': AnalysisConfig(
                    enable_thermal_analysis=True,
                    enable_peak_detection=False,
                    enable_trend_analysis=False,
                    enable_temperature_trends=False,
                    enable_distributions=False,
                    enable_heatmaps=False,
                    enable_anomaly_plots=False,
                    enable_correlations=True
                )
            }
        }
        
        # Save presets
        for preset_name, preset_data in presets.items():
            filename = f"preset_{preset_name}.json"
            temp_config = self.config
            self.config = preset_data['config']
            self.save_config(filename)
            self.config = temp_config
            
            # Create description file
            desc_filename = f"preset_{preset_name}_description.txt"
            with open(desc_filename, 'w', encoding='utf-8') as f:
                f.write(f"PRESET: {preset_name.upper()}\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Description: {preset_data['description']}\n\n")
                f.write("Usage:\n")
                f.write(f"python improved_analyzer.py test.CSV --config {filename}\n\n")
                f.write("Enabled methods:\n")
                f.write("-" * 20 + "\n")
                
                config = preset_data['config']
                if config.enable_isolation_forest:
                    f.write("[SI] Isolation Forest\n")
                if config.enable_zscore:
                    f.write("[SI] Z-Score\n")
                if config.enable_iqr:
                    f.write("[SI] IQR\n")
                if config.enable_thermal_analysis:
                    f.write("[YES] Thermal Analysis\n")
                if config.enable_voltage_analysis:
                    f.write("[YES] Voltage Analysis\n")
                if config.enable_all_plots or any([
                    config.enable_temperature_trends,
                    config.enable_distributions,
                    config.enable_heatmaps,
                    config.enable_dashboard
                ]):
                    f.write("[YES] Visualizations\n")
        
        print(f"Created {len(presets)} predefined configuration files")
        return list(presets.keys())
    
    def print_current_config(self):
        """Print current configuration"""
        print("\nANALYSIS CONFIGURATION")
        print("=" * 40)
        
        print("\nAnomaly Detection Methods:")
        print(f"  Isolation Forest: {'[YES]' if self.config.enable_isolation_forest else '[NO]'}")
        print(f"  Z-Score: {'[YES]' if self.config.enable_zscore else '[NO]'}")
        print(f"  IQR: {'[YES]' if self.config.enable_iqr else '[NO]'}")
        
        print("\nThermal and Voltage Analysis:")
        print(f"  Thermal Analysis: {'[YES]' if self.config.enable_thermal_analysis else '[NO]'}")
        print(f"  Peak Detection: {'[YES]' if self.config.enable_peak_detection else '[NO]'}")
        print(f"  Trend Analysis: {'[YES]' if self.config.enable_trend_analysis else '[NO]'}")
        print(f"  Voltage Analysis: {'[YES]' if self.config.enable_voltage_analysis else '[NO]'}")
        print(f"  Voltage Stability: {'[YES]' if self.config.enable_voltage_stability else '[NO]'}")
        
        print("\nVisualizations:")
        if self.config.enable_all_plots:
            print("  All plots: [YES]")
        else:
            print(f"  Temperature Trends: {'[YES]' if self.config.enable_temperature_trends else '[NO]'}")
            print(f"  Distributions: {'[YES]' if self.config.enable_distributions else '[NO]'}")
            print(f"  Heatmaps: {'[YES]' if self.config.enable_heatmaps else '[NO]'}")
            print(f"  Voltage Plots: {'[YES]' if self.config.enable_voltage_plots else '[NO]'}")
            print(f"  Anomaly Plots: {'[YES]' if self.config.enable_anomaly_plots else '[NO]'}")
            print(f"  Dashboard: {'[YES]' if self.config.enable_dashboard else '[NO]'}")
            print(f"  Correlations: {'[YES]' if self.config.enable_correlations else '[NO]'}")
        
        print("\nParameters:")
        print(f"  IF Contamination: {self.config.isolation_forest_contamination}")
        print(f"  Z-Score Threshold: {self.config.zscore_threshold}")
        print(f"  IQR Multiplier: {self.config.iqr_multiplier}")
        print(f"  Plot Resolution: {self.config.plot_resolution_dpi} DPI")

# Utility functions for creating quick configurations
def create_default_config() -> AnalysisMethodSelector:
    """Create default configuration (all methods)"""
    selector = AnalysisMethodSelector()
    selector.enable_all_methods()
    return selector

def create_quick_config() -> AnalysisMethodSelector:
    """Create configuration for quick analysis"""
    selector = AnalysisMethodSelector()
    selector.enable_minimal_analysis()
    return selector

def create_comprehensive_config() -> AnalysisMethodSelector:
    """Create configuration for exhaustive analysis"""
    selector = AnalysisMethodSelector()
    selector.enable_comprehensive_analysis()
    return selector

if __name__ == "__main__":
    # Create predefined configurations
    selector = AnalysisMethodSelector()
    
    print("Creating predefined configurations...")
    presets = selector.create_preset_configs()
    
    print(f"\nCreated configurations: {', '.join(presets)}")
    print("\nDefault configuration (comprehensive):")
    selector.print_current_config()
    
    # Save default configuration
    selector.save_config("default_config.json")