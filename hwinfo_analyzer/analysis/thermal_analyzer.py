"""
Thermal Analysis Module
Analyzes temperature data and provides hardware-specific diagnostics.
"""

import numpy as np
import pandas as pd
from ..core.thermal_thresholds import ThermalThresholds

class ThermalAnalyzer:
    """Analyzes thermal behavior of hardware components."""
    
    def __init__(self):
        self.thresholds = ThermalThresholds()
    
    def analyze_cpu_temperatures(self, data_processor, anomaly_detector):
        """Analyze CPU temperature behavior with support for hybrid architectures."""
        cpu_analysis = {
            'overall_status': 'unknown',
            'individual_cores': {},
            'thermal_events': [],
            'recommendations': [],
            'health_score': 100,
            'cpu_architecture': 'unknown'
        }
        
        if not data_processor.cpu_temp_columns:
            cpu_analysis['overall_status'] = 'no_data'
            return cpu_analysis
        
        # Detect CPU architecture
        cpu_vendor = ThermalThresholds.detect_cpu_vendor(data_processor.cpu_temp_columns)
        cpu_analysis['cpu_architecture'] = cpu_vendor
        
        all_cpu_temps = []
        core_analyses = {}
        
        # Analyze each CPU temperature sensor
        for col in data_processor.cpu_temp_columns:
            data = data_processor.get_column_data(col)
            if len(data) < 5:
                continue
            
            # Get anomalies
            anomalies = anomaly_detector.detect_anomalies(data)
            patterns = anomaly_detector.detect_patterns(data)
            
            # Classify temperatures
            mean_temp = data.mean()
            max_temp = data.max()
            min_temp = data.min()
            
            # Determine core type for hybrid CPUs and classify temperatures
            core_type = 'standard'
            if cpu_vendor == 'intel_hybrid':
                if 'P-CORE' in col.upper():
                    core_type = 'p_core'
                elif 'E-CORE' in col.upper():
                    core_type = 'e_core'
                mean_classification = ThermalThresholds.classify_hybrid_cpu_temperature(mean_temp, core_type)
                max_classification = ThermalThresholds.classify_hybrid_cpu_temperature(max_temp, core_type)
                cpu_type = 'intel_hybrid'
            else:
                # Use traditional classification
                cpu_type = cpu_vendor if cpu_vendor in ['intel', 'amd'] else 'generic'
                mean_classification = self.thresholds.classify_cpu_temperature(mean_temp, cpu_type)
                max_classification = self.thresholds.classify_cpu_temperature(max_temp, cpu_type)
            
            core_analysis = {
                'sensor_name': col,
                'core_type': core_type,
                'mean_temp': mean_temp,
                'max_temp': max_temp,
                'min_temp': min_temp,
                'mean_classification': mean_classification,
                'max_classification': max_classification,
                'anomaly_percentage': anomalies['anomaly_percentage'],
                'stability': patterns['stability'],
                'trend': patterns['trend'],
                'issues': []
            }
            
            # Identify issues using appropriate thresholds
            if cpu_vendor == 'intel_hybrid':
                thresholds = ThermalThresholds.CPU_THRESHOLDS['intel_hybrid']
                if core_type == 'p_core':
                    critical_thresh = thresholds['p_core_critical']
                    warning_thresh = thresholds['p_core_warning']
                    normal_thresh = thresholds['p_core_normal_max']
                else:  # e_core
                    critical_thresh = thresholds['e_core_critical']
                    warning_thresh = thresholds['e_core_warning']
                    normal_thresh = thresholds['e_core_normal_max']
            else:
                thresholds = self.thresholds.get_cpu_thresholds(cpu_type)
                critical_thresh = thresholds['critical']
                warning_thresh = thresholds['warning']
                normal_thresh = thresholds['normal_load_max']
            
            if max_temp > critical_thresh:
                core_analysis['issues'].append(f"Critical temperature reached: {max_temp:.1f}°C ({core_type})")
                cpu_analysis['health_score'] -= 30
            elif max_temp > warning_thresh:
                core_analysis['issues'].append(f"Warning temperature reached: {max_temp:.1f}°C ({core_type})")
                cpu_analysis['health_score'] -= 15
            elif mean_temp > normal_thresh:
                core_analysis['issues'].append(f"Elevated average temperature: {mean_temp:.1f}°C ({core_type})")
                cpu_analysis['health_score'] -= 10
            
            if anomalies['anomaly_percentage'] > 10:
                core_analysis['issues'].append(f"High temperature instability: {anomalies['anomaly_percentage']:.1f}% anomalies")
                cpu_analysis['health_score'] -= 15
            elif anomalies['anomaly_percentage'] > 5:
                core_analysis['issues'].append(f"Moderate temperature spikes: {anomalies['anomaly_percentage']:.1f}% anomalies")
                cpu_analysis['health_score'] -= 10
            
            core_analyses[col] = core_analysis
            all_cpu_temps.extend(data.tolist())
        
        cpu_analysis['individual_cores'] = core_analyses
        
        # Overall CPU assessment
        if all_cpu_temps:
            overall_max = max(all_cpu_temps)
            overall_mean = np.mean(all_cpu_temps)
            
            if overall_max > 95:
                cpu_analysis['overall_status'] = 'critical'
                cpu_analysis['recommendations'].extend([
                    "IMMEDIATE ACTION REQUIRED: CPU temperatures are critically high",
                    "Check CPU cooler mounting and thermal paste",
                    "Verify all cooling fans are working",
                    "Consider reducing CPU load or improving cooling"
                ])
            elif overall_max > 85:
                cpu_analysis['overall_status'] = 'warning'
                cpu_analysis['recommendations'].extend([
                    "CPU temperatures are elevated and need attention",
                    "Check case airflow and clean dust filters",
                    "Monitor temperatures under load"
                ])
            elif overall_mean > 80:
                cpu_analysis['overall_status'] = 'elevated'
                cpu_analysis['recommendations'].append("CPU is running warm but within acceptable limits")
            else:
                cpu_analysis['overall_status'] = 'good'
        
        cpu_analysis['health_score'] = max(0, cpu_analysis['health_score'])
        return cpu_analysis
    
    def analyze_gpu_temperatures(self, data_processor, anomaly_detector):
        """Analyze GPU temperature behavior."""
        gpu_analysis = {
            'overall_status': 'unknown',
            'individual_gpus': {},
            'thermal_events': [],
            'recommendations': [],
            'health_score': 100
        }
        
        if not data_processor.gpu_temp_columns:
            gpu_analysis['overall_status'] = 'no_data'
            return gpu_analysis
        
        all_gpu_temps = []
        gpu_analyses = {}
        
        # Analyze each GPU temperature sensor
        for col in data_processor.gpu_temp_columns:
            data = data_processor.get_column_data(col)
            if len(data) < 5:
                continue
            
            # Get anomalies and patterns
            anomalies = anomaly_detector.detect_anomalies(data)
            patterns = anomaly_detector.detect_patterns(data)
            
            mean_temp = data.mean()
            max_temp = data.max()
            min_temp = data.min()
            
            # Determine GPU type (basic heuristic)
            gpu_type = 'generic'
            if any(nvidia_term in col.upper() for nvidia_term in ['NVIDIA', 'GTX', 'RTX']):
                gpu_type = 'nvidia'
            elif any(amd_term in col.upper() for amd_term in ['AMD', 'RADEON', 'RX']):
                gpu_type = 'amd'
            
            mean_classification = self.thresholds.classify_gpu_temperature(mean_temp, gpu_type)
            max_classification = self.thresholds.classify_gpu_temperature(max_temp, gpu_type)
            
            gpu_sensor_analysis = {
                'sensor_name': col,
                'mean_temp': mean_temp,
                'max_temp': max_temp,
                'min_temp': min_temp,
                'mean_classification': mean_classification,
                'max_classification': max_classification,
                'anomaly_percentage': anomalies['anomaly_percentage'],
                'stability': patterns['stability'],
                'trend': patterns['trend'],
                'issues': []
            }
            
            # Identify GPU-specific issues
            thresholds = self.thresholds.get_gpu_thresholds(gpu_type)
            
            if max_temp > thresholds['critical']:
                gpu_sensor_analysis['issues'].append(f"Critical GPU temperature: {max_temp:.1f}°C")
                gpu_analysis['health_score'] -= 25
            elif max_temp > thresholds['warning']:
                gpu_sensor_analysis['issues'].append(f"High GPU temperature: {max_temp:.1f}°C")
                gpu_analysis['health_score'] -= 15
            elif mean_temp > thresholds['normal_load_max']:
                gpu_sensor_analysis['issues'].append(f"Elevated average GPU temperature: {mean_temp:.1f}°C")
                gpu_analysis['health_score'] -= 10
            
            if anomalies['anomaly_percentage'] > 8:
                gpu_sensor_analysis['issues'].append(f"GPU temperature instability: {anomalies['anomaly_percentage']:.1f}% anomalies")
                gpu_analysis['health_score'] -= 10
            
            gpu_analyses[col] = gpu_sensor_analysis
            all_gpu_temps.extend(data.tolist())
        
        gpu_analysis['individual_gpus'] = gpu_analyses
        
        # Overall GPU assessment
        if all_gpu_temps:
            overall_max = max(all_gpu_temps)
            overall_mean = np.mean(all_gpu_temps)
            
            if overall_max > 90:
                gpu_analysis['overall_status'] = 'critical'
                gpu_analysis['recommendations'].extend([
                    "GPU temperatures are critically high",
                    "Check GPU fan curves and thermal paste",
                    "Ensure adequate case ventilation",
                    "Consider undervolting or reducing GPU clocks"
                ])
            elif overall_max > 80:
                gpu_analysis['overall_status'] = 'warning'
                gpu_analysis['recommendations'].extend([
                    "GPU is running hot",
                    "Monitor GPU temperatures during gaming",
                    "Check GPU fan operation"
                ])
            elif overall_mean > 75:
                gpu_analysis['overall_status'] = 'elevated'
                gpu_analysis['recommendations'].append("GPU temperatures are acceptable but monitor under load")
            else:
                gpu_analysis['overall_status'] = 'good'
        
        gpu_analysis['health_score'] = max(0, gpu_analysis['health_score'])
        return gpu_analysis
    
    def analyze_system_thermals(self, data_processor, anomaly_detector):
        """Analyze overall system thermal behavior."""
        system_analysis = {
            'thermal_zones': {},
            'correlations': {},
            'system_health_score': 100,
            'recommendations': []
        }
        
        # Analyze motherboard and other temperature sensors
        for col in data_processor.motherboard_temp_columns:
            data = data_processor.get_column_data(col)
            if len(data) < 5:
                continue
            
            anomalies = anomaly_detector.detect_anomalies(data)
            mean_temp = data.mean()
            max_temp = data.max()
            
            zone_analysis = {
                'sensor_name': col,
                'mean_temp': mean_temp,
                'max_temp': max_temp,
                'anomaly_percentage': anomalies['anomaly_percentage'],
                'status': 'unknown'
            }
            
            # Classify based on sensor type
            if any(term in col.upper() for term in ['MOTHERBOARD', 'SYSTEM', 'AMBIENT']):
                if max_temp > 60:
                    zone_analysis['status'] = 'hot'
                    system_analysis['system_health_score'] -= 10
                elif mean_temp > 50:
                    zone_analysis['status'] = 'warm'
                    system_analysis['system_health_score'] -= 5
                else:
                    zone_analysis['status'] = 'good'
            elif any(term in col.upper() for term in ['VRM', 'MOSFET', 'CHIPSET']):
                if max_temp > 85:
                    zone_analysis['status'] = 'critical'
                    system_analysis['system_health_score'] -= 20
                elif max_temp > 75:
                    zone_analysis['status'] = 'hot'
                    system_analysis['system_health_score'] -= 10
                else:
                    zone_analysis['status'] = 'good'
            
            system_analysis['thermal_zones'][col] = zone_analysis
        
        # Generate system-level recommendations
        if system_analysis['system_health_score'] < 80:
            system_analysis['recommendations'].extend([
                "System thermal management needs improvement",
                "Check case fan configuration and airflow",
                "Clean dust from all components"
            ])
        
        system_analysis['system_health_score'] = max(0, system_analysis['system_health_score'])
        return system_analysis