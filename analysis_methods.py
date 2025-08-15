"""
Selector y configurador de métodos de análisis para HWiNFO
Permite seleccionar qué métodos ejecutar por defecto o específicos
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
    """Configuración para métodos de análisis"""
    # Métodos de anomalías
    enable_isolation_forest: bool = True
    enable_zscore: bool = True
    enable_iqr: bool = True
    
    # Análisis térmico
    enable_thermal_analysis: bool = True
    enable_peak_detection: bool = True
    enable_trend_analysis: bool = True
    
    # Análisis de voltajes
    enable_voltage_analysis: bool = True
    enable_voltage_stability: bool = True
    
    # Visualizaciones
    enable_all_plots: bool = True
    enable_temperature_trends: bool = True
    enable_distributions: bool = True
    enable_heatmaps: bool = True
    enable_voltage_plots: bool = True
    enable_anomaly_plots: bool = True
    enable_dashboard: bool = True
    enable_correlations: bool = True
    
    # Configuraciones avanzadas
    isolation_forest_contamination: float = 0.1
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    
    # Configuraciones de visualización
    plot_resolution_dpi: int = 300
    max_sensors_per_plot: int = 6
    sample_rate_for_heatmap: int = 100

class AnalysisMethodSelector:
    """Selector y configurador de métodos de análisis"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.config = self._load_or_create_config()
    
    def _load_or_create_config(self) -> AnalysisConfig:
        """Carga configuración desde archivo o crea una por defecto"""
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                return AnalysisConfig(**config_dict)
            except Exception as e:
                print(f"Error loading config: {e}. Using default configuration.")
        
        # Configuración por defecto (TODOS LOS MÉTODOS ACTIVADOS)
        return AnalysisConfig()
    
    def save_config(self, filename: str = None):
        """Guarda la configuración actual a archivo"""
        config_file = filename or self.config_file or "analysis_config.json"
        
        config_dict = {
            # Métodos de anomalías
            'enable_isolation_forest': self.config.enable_isolation_forest,
            'enable_zscore': self.config.enable_zscore,
            'enable_iqr': self.config.enable_iqr,
            
            # Análisis térmico
            'enable_thermal_analysis': self.config.enable_thermal_analysis,
            'enable_peak_detection': self.config.enable_peak_detection,
            'enable_trend_analysis': self.config.enable_trend_analysis,
            
            # Análisis de voltajes
            'enable_voltage_analysis': self.config.enable_voltage_analysis,
            'enable_voltage_stability': self.config.enable_voltage_stability,
            
            # Visualizaciones
            'enable_all_plots': self.config.enable_all_plots,
            'enable_temperature_trends': self.config.enable_temperature_trends,
            'enable_distributions': self.config.enable_distributions,
            'enable_heatmaps': self.config.enable_heatmaps,
            'enable_voltage_plots': self.config.enable_voltage_plots,
            'enable_anomaly_plots': self.config.enable_anomaly_plots,
            'enable_dashboard': self.config.enable_dashboard,
            'enable_correlations': self.config.enable_correlations,
            
            # Configuraciones avanzadas
            'isolation_forest_contamination': self.config.isolation_forest_contamination,
            'zscore_threshold': self.config.zscore_threshold,
            'iqr_multiplier': self.config.iqr_multiplier,
            
            # Configuraciones de visualización
            'plot_resolution_dpi': self.config.plot_resolution_dpi,
            'max_sensors_per_plot': self.config.max_sensors_per_plot,
            'sample_rate_for_heatmap': self.config.sample_rate_for_heatmap
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Configuración guardada en: {config_file}")
    
    def get_enabled_anomaly_methods(self) -> List[str]:
        """Retorna lista de métodos de anomalías habilitados"""
        methods = []
        if self.config.enable_isolation_forest:
            methods.append('isolation_forest')
        if self.config.enable_zscore:
            methods.append('zscore')
        if self.config.enable_iqr:
            methods.append('iqr')
        return methods
    
    def get_enabled_analysis_methods(self) -> Dict[str, bool]:
        """Retorna diccionario de métodos de análisis habilitados"""
        return {
            'thermal_analysis': self.config.enable_thermal_analysis,
            'peak_detection': self.config.enable_peak_detection,
            'trend_analysis': self.config.enable_trend_analysis,
            'voltage_analysis': self.config.enable_voltage_analysis,
            'voltage_stability': self.config.enable_voltage_stability
        }
    
    def get_enabled_visualization_methods(self) -> Dict[str, bool]:
        """Retorna diccionario de visualizaciones habilitadas"""
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
        """Retorna parámetros para algoritmos de anomalías"""
        return {
            'isolation_forest_contamination': self.config.isolation_forest_contamination,
            'zscore_threshold': self.config.zscore_threshold,
            'iqr_multiplier': self.config.iqr_multiplier
        }
    
    def get_visualization_parameters(self) -> Dict[str, Any]:
        """Retorna parámetros para visualizaciones"""
        return {
            'plot_resolution_dpi': self.config.plot_resolution_dpi,
            'max_sensors_per_plot': self.config.max_sensors_per_plot,
            'sample_rate_for_heatmap': self.config.sample_rate_for_heatmap
        }
    
    def enable_all_methods(self):
        """Habilita todos los métodos de análisis"""
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
        """Habilita solo análisis esencial (rápido)"""
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
        """Habilita análisis completo y exhaustivo"""
        self.enable_all_methods()
        # Configuraciones más sensibles para análisis exhaustivo
        self.config.isolation_forest_contamination = 0.05
        self.config.zscore_threshold = 2.5
        self.config.iqr_multiplier = 1.2
    
    def create_preset_configs(self):
        """Crea archivos de configuración predefinidos"""
        presets = {
            'comprehensive': {
                'description': 'Análisis completo con todos los métodos habilitados',
                'config': AnalysisConfig()
            },
            'minimal': {
                'description': 'Análisis rápido con métodos esenciales',
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
                'description': 'Enfoque en análisis térmico',
                'config': AnalysisConfig(
                    enable_voltage_analysis=False,
                    enable_voltage_stability=False,
                    enable_voltage_plots=False,
                    zscore_threshold=2.0,
                    iqr_multiplier=1.2
                )
            },
            'voltage_focus': {
                'description': 'Enfoque en análisis de voltajes',
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
        
        # Guardar presets
        for preset_name, preset_data in presets.items():
            filename = f"preset_{preset_name}.json"
            temp_config = self.config
            self.config = preset_data['config']
            self.save_config(filename)
            self.config = temp_config
            
            # Crear archivo de descripción
            desc_filename = f"preset_{preset_name}_description.txt"
            with open(desc_filename, 'w', encoding='utf-8') as f:
                f.write(f"PRESET: {preset_name.upper()}\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Descripción: {preset_data['description']}\n\n")
                f.write("Uso:\n")
                f.write(f"python improved_analyzer.py test.CSV --config {filename}\n\n")
                f.write("Métodos habilitados:\n")
                f.write("-" * 20 + "\n")
                
                config = preset_data['config']
                if config.enable_isolation_forest:
                    f.write("[SI] Isolation Forest\n")
                if config.enable_zscore:
                    f.write("[SI] Z-Score\n")
                if config.enable_iqr:
                    f.write("[SI] IQR\n")
                if config.enable_thermal_analysis:
                    f.write("[SI] Análisis Térmico\n")
                if config.enable_voltage_analysis:
                    f.write("[SI] Análisis de Voltajes\n")
                if config.enable_all_plots or any([
                    config.enable_temperature_trends,
                    config.enable_distributions,
                    config.enable_heatmaps,
                    config.enable_dashboard
                ]):
                    f.write("[SI] Visualizaciones\n")
        
        print(f"Creados {len(presets)} archivos de configuración predefinidos")
        return list(presets.keys())
    
    def print_current_config(self):
        """Imprime la configuración actual"""
        print("\nCONFIGURACIÓN ACTUAL DE ANÁLISIS")
        print("=" * 40)
        
        print("\nMétodos de Anomalías:")
        print(f"  Isolation Forest: {'[SI]' if self.config.enable_isolation_forest else '[NO]'}")
        print(f"  Z-Score: {'[SI]' if self.config.enable_zscore else '[NO]'}")
        print(f"  IQR: {'[SI]' if self.config.enable_iqr else '[NO]'}")
        
        print("\nAnálisis Térmico y Voltajes:")
        print(f"  Análisis Térmico: {'[SI]' if self.config.enable_thermal_analysis else '[NO]'}")
        print(f"  Detección de Picos: {'[SI]' if self.config.enable_peak_detection else '[NO]'}")
        print(f"  Análisis de Tendencias: {'[SI]' if self.config.enable_trend_analysis else '[NO]'}")
        print(f"  Análisis de Voltajes: {'[SI]' if self.config.enable_voltage_analysis else '[NO]'}")
        print(f"  Estabilidad de Voltajes: {'[SI]' if self.config.enable_voltage_stability else '[NO]'}")
        
        print("\nVisualizaciones:")
        if self.config.enable_all_plots:
            print("  Todos los gráficos: [SI]")
        else:
            print(f"  Tendencias de Temperatura: {'[SI]' if self.config.enable_temperature_trends else '[NO]'}")
            print(f"  Distribuciones: {'[SI]' if self.config.enable_distributions else '[NO]'}")
            print(f"  Heatmaps: {'[SI]' if self.config.enable_heatmaps else '[NO]'}")
            print(f"  Gráficos de Voltaje: {'[SI]' if self.config.enable_voltage_plots else '[NO]'}")
            print(f"  Gráficos de Anomalías: {'[SI]' if self.config.enable_anomaly_plots else '[NO]'}")
            print(f"  Dashboard: {'[SI]' if self.config.enable_dashboard else '[NO]'}")
            print(f"  Correlaciones: {'[SI]' if self.config.enable_correlations else '[NO]'}")
        
        print("\nParámetros:")
        print(f"  Contaminación IF: {self.config.isolation_forest_contamination}")
        print(f"  Umbral Z-Score: {self.config.zscore_threshold}")
        print(f"  Multiplicador IQR: {self.config.iqr_multiplier}")
        print(f"  Resolución gráficos: {self.config.plot_resolution_dpi} DPI")

# Funciones de utilidad para crear configuraciones rápidas
def create_default_config() -> AnalysisMethodSelector:
    """Crea configuración por defecto (todos los métodos)"""
    selector = AnalysisMethodSelector()
    selector.enable_all_methods()
    return selector

def create_quick_config() -> AnalysisMethodSelector:
    """Crea configuración para análisis rápido"""
    selector = AnalysisMethodSelector()
    selector.enable_minimal_analysis()
    return selector

def create_comprehensive_config() -> AnalysisMethodSelector:
    """Crea configuración para análisis exhaustivo"""
    selector = AnalysisMethodSelector()
    selector.enable_comprehensive_analysis()
    return selector

if __name__ == "__main__":
    # Crear configuraciones predefinidas
    selector = AnalysisMethodSelector()
    
    print("Creando configuraciones predefinidas...")
    presets = selector.create_preset_configs()
    
    print(f"\nCreadas configuraciones: {', '.join(presets)}")
    print("\nConfiguración por defecto (comprehensive):")
    selector.print_current_config()
    
    # Guardar configuración por defecto
    selector.save_config("default_config.json")