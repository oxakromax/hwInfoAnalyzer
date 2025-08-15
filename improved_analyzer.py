#!/usr/bin/env python3
"""
Improved HWiNFO Analyzer
Modular and scientifically accurate analysis of HWiNFO logs.
"""

import argparse
import os
from datetime import datetime

from data_processor import HWInfoDataProcessor
from anomaly_detector import AnomalyDetector
from thermal_analyzer import ThermalAnalyzer
from thermal_thresholds import ThermalThresholds

class ImprovedHWInfoAnalyzer:
    """Main analyzer class with improved accuracy and modularity."""
    
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data_processor = HWInfoDataProcessor()
        self.anomaly_detector = AnomalyDetector()
        self.thermal_analyzer = ThermalAnalyzer()
        self.analysis_results = {}
    
    def run_analysis(self):
        """Run complete analysis pipeline."""
        print("Starting Improved HWiNFO Analysis...")
        print("=" * 60)
        
        # Load and process data
        self.data_processor.load_csv(self.csv_file)
        
        # Get data summary
        summary = self.data_processor.get_summary_statistics()
        self.analysis_results['summary'] = summary
        
        # Thermal analysis
        print("\nAnalyzing CPU thermal behavior...")
        cpu_analysis = self.thermal_analyzer.analyze_cpu_temperatures(
            self.data_processor, self.anomaly_detector
        )
        self.analysis_results['cpu_thermal'] = cpu_analysis
        
        print("Analyzing GPU thermal behavior...")
        gpu_analysis = self.thermal_analyzer.analyze_gpu_temperatures(
            self.data_processor, self.anomaly_detector
        )
        self.analysis_results['gpu_thermal'] = gpu_analysis
        
        print("Analyzing system thermal zones...")
        system_analysis = self.thermal_analyzer.analyze_system_thermals(
            self.data_processor, self.anomaly_detector
        )
        self.analysis_results['system_thermal'] = system_analysis
        
        # Voltage analysis
        print("Analyzing voltage stability...")
        voltage_analysis = self._analyze_voltages()
        self.analysis_results['voltage'] = voltage_analysis
        
        # Generate final diagnosis
        diagnosis = self._generate_diagnosis()
        self.analysis_results['diagnosis'] = diagnosis
        
        return self.analysis_results
    
    def _analyze_voltages(self):
        """Analyze voltage stability and anomalies."""
        voltage_analysis = {
            'rail_analyses': {},
            'overall_stability': 'unknown',
            'health_score': 100,
            'recommendations': []
        }
        
        if not self.data_processor.voltage_columns:
            voltage_analysis['overall_stability'] = 'no_data'
            return voltage_analysis
        
        instability_count = 0
        critical_rails = []
        
        # Analyze each voltage rail
        for col in self.data_processor.voltage_columns:
            data = self.data_processor.get_column_data(col)
            if len(data) < 10:
                continue
            
            anomalies = self.anomaly_detector.detect_anomalies(data, method='statistical')
            patterns = self.anomaly_detector.detect_patterns(data)
            
            mean_voltage = data.mean()
            std_voltage = data.std()
            cv = std_voltage / abs(mean_voltage) if mean_voltage != 0 else float('inf')
            
            rail_analysis = {
                'rail_name': col,
                'mean_voltage': mean_voltage,
                'std_voltage': std_voltage,
                'coefficient_of_variation': cv,
                'anomaly_percentage': anomalies['anomaly_percentage'],
                'stability_rating': patterns['stability'],
                'issues': []
            }
            
            # Determine rail type and thresholds
            rail_type = 'system_rails'  # Default
            if any(term in col.upper() for term in ['CPU', 'CORE', 'VCORE']):
                rail_type = 'cpu_core'
            elif any(term in col.upper() for term in ['GPU', 'GRAPHICS']):
                rail_type = 'gpu_core'
            
            thresholds = ThermalThresholds.VOLTAGE_THRESHOLDS[rail_type]
            
            # Assess voltage stability
            if cv > thresholds['critical_variation']:
                rail_analysis['issues'].append(f"Critical voltage instability: {cv*100:.2f}% variation")
                voltage_analysis['health_score'] -= 20
                critical_rails.append(col)
                instability_count += 1
            elif cv > thresholds['warning_variation']:
                rail_analysis['issues'].append(f"High voltage variation: {cv*100:.2f}%")
                voltage_analysis['health_score'] -= 10
                instability_count += 1
            elif cv > thresholds['normal_variation']:
                rail_analysis['issues'].append(f"Elevated voltage variation: {cv*100:.2f}%")
                voltage_analysis['health_score'] -= 5
            
            # Check for voltage spikes
            if anomalies['anomaly_percentage'] > 5:
                rail_analysis['issues'].append(f"Voltage spikes detected: {anomalies['anomaly_percentage']:.1f}% anomalies")
                voltage_analysis['health_score'] -= 10
            
            voltage_analysis['rail_analyses'][col] = rail_analysis
        
        # Overall voltage assessment
        total_rails = len(voltage_analysis['rail_analyses'])
        if total_rails == 0:
            voltage_analysis['overall_stability'] = 'no_data'
        elif instability_count == 0:
            voltage_analysis['overall_stability'] = 'stable'
        elif instability_count / total_rails < 0.3:
            voltage_analysis['overall_stability'] = 'mostly_stable'
        else:
            voltage_analysis['overall_stability'] = 'unstable'
            voltage_analysis['recommendations'].extend([
                "Multiple voltage rails show instability",
                "Check PSU quality and capacity",
                "Verify all power connections are secure"
            ])
        
        if critical_rails:
            voltage_analysis['recommendations'].extend([
                f"Critical voltage instability detected in: {', '.join(critical_rails[:3])}",
                "This may indicate PSU problems or inadequate power delivery",
                "Consider PSU replacement or load reduction"
            ])
        
        voltage_analysis['health_score'] = max(0, voltage_analysis['health_score'])
        return voltage_analysis
    
    def _generate_diagnosis(self):
        """Generate comprehensive system diagnosis."""
        diagnosis = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file_analyzed': self.csv_file,
            'system_health_score': 100,
            'overall_status': 'unknown',
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'component_status': {}
        }
        
        # Component health scores
        cpu_score = self.analysis_results.get('cpu_thermal', {}).get('health_score', 100)
        gpu_score = self.analysis_results.get('gpu_thermal', {}).get('health_score', 100)
        system_score = self.analysis_results.get('system_thermal', {}).get('health_score', 100)
        voltage_score = self.analysis_results.get('voltage', {}).get('health_score', 100)
        
        # Calculate weighted system health score
        weights = {'cpu': 0.4, 'gpu': 0.3, 'system': 0.15, 'voltage': 0.15}
        diagnosis['system_health_score'] = (
            cpu_score * weights['cpu'] +
            gpu_score * weights['gpu'] +
            system_score * weights['system'] +
            voltage_score * weights['voltage']
        )
        
        # Component status
        diagnosis['component_status'] = {
            'cpu': self._score_to_status(cpu_score),
            'gpu': self._score_to_status(gpu_score),
            'system_thermal': self._score_to_status(system_score),
            'voltage': self._score_to_status(voltage_score)
        }
        
        # Overall system status
        diagnosis['overall_status'] = self._score_to_status(diagnosis['system_health_score'])
        
        # Collect issues and recommendations
        for component in ['cpu_thermal', 'gpu_thermal', 'system_thermal', 'voltage']:
            component_data = self.analysis_results.get(component, {})
            
            # Collect recommendations
            if 'recommendations' in component_data:
                diagnosis['recommendations'].extend(component_data['recommendations'])
            
            # Collect critical issues
            if component == 'cpu_thermal':
                cpu_status = component_data.get('overall_status', 'unknown')
                if cpu_status == 'critical':
                    diagnosis['critical_issues'].append("CPU temperatures are critically high")
                elif cpu_status == 'warning':
                    diagnosis['warnings'].append("CPU temperatures are elevated")
            
            elif component == 'gpu_thermal':
                gpu_status = component_data.get('overall_status', 'unknown')
                if gpu_status == 'critical':
                    diagnosis['critical_issues'].append("GPU temperatures are critically high")
                elif gpu_status == 'warning':
                    diagnosis['warnings'].append("GPU temperatures are elevated")
            
            elif component == 'voltage':
                voltage_status = component_data.get('overall_stability', 'unknown')
                if voltage_status == 'unstable':
                    diagnosis['critical_issues'].append("Multiple voltage rails are unstable")
                elif voltage_status == 'mostly_stable':
                    diagnosis['warnings'].append("Some voltage instability detected")
        
        # Add general recommendations based on overall health
        if diagnosis['system_health_score'] < 50:
            diagnosis['recommendations'].insert(0, "IMMEDIATE ACTION REQUIRED: System has multiple critical issues")
        elif diagnosis['system_health_score'] < 75:
            diagnosis['recommendations'].insert(0, "System needs attention to prevent potential hardware damage")
        
        return diagnosis
    
    def _score_to_status(self, score):
        """Convert numeric score to status string."""
        if score >= 90:
            return 'excellent'
        elif score >= 75:
            return 'good'
        elif score >= 60:
            return 'fair'
        elif score >= 40:
            return 'poor'
        else:
            return 'critical'
    
    def print_summary(self):
        """Print a concise summary of the analysis."""
        if 'diagnosis' not in self.analysis_results:
            print("No diagnosis available")
            return
        
        diagnosis = self.analysis_results['diagnosis']
        
        print(f"\n" + "=" * 60)
        print("HWINFO ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Overall health
        status_symbols = {
            'excellent': '[EXCELLENT]',
            'good': '[GOOD]',
            'fair': '[FAIR]',
            'poor': '[POOR]',
            'critical': '[CRITICAL]'
        }
        
        status = diagnosis['overall_status']
        print(f"System Health: {status_symbols.get(status, '[UNKNOWN]')} {status.upper()}")
        print(f"Health Score: {diagnosis['system_health_score']:.1f}/100")
        
        # Component status
        print(f"\nComponent Status:")
        for component, comp_status in diagnosis['component_status'].items():
            print(f"  {component.upper()}: {comp_status}")
        
        # Critical issues
        if diagnosis['critical_issues']:
            print(f"\nCRITICAL ISSUES:")
            for issue in diagnosis['critical_issues']:
                print(f"  - {issue}")
        
        # Warnings
        if diagnosis['warnings']:
            print(f"\nWARNINGS:")
            for warning in diagnosis['warnings']:
                print(f"  - {warning}")
        
        # Top recommendations
        if diagnosis['recommendations']:
            print(f"\nRECOMMENDATIONS:")
            for rec in diagnosis['recommendations'][:5]:  # Show top 5
                print(f"  - {rec}")
        
        # Data summary
        summary = self.analysis_results.get('summary', {})
        print(f"\nData Summary:")
        print(f"  Total samples: {summary.get('total_samples', 'unknown')}")
        if summary.get('sampling_frequency'):
            print(f"  Sampling frequency: {summary['sampling_frequency']}")
        
        print(f"\nAnalysis completed: {diagnosis['timestamp']}")

def main():
    parser = argparse.ArgumentParser(description='Improved HWiNFO CSV Log Analyzer')
    parser.add_argument('csv_file', help='Path to HWiNFO CSV log file')
    parser.add_argument('--output', '-o', help='Output directory for detailed reports')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found!")
        return
    
    # Run analysis
    analyzer = ImprovedHWInfoAnalyzer(args.csv_file)
    results = analyzer.run_analysis()
    
    # Print summary
    analyzer.print_summary()
    
    # Save detailed report if output directory specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        report_file = os.path.join(args.output, 'detailed_analysis.txt')
        save_detailed_report(results, report_file)
        print(f"\nDetailed report saved to: {report_file}")

def save_detailed_report(results, output_file):
    """Save detailed analysis report."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("DETAILED HWINFO ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        diagnosis = results.get('diagnosis', {})
        f.write(f"Analysis Date: {diagnosis.get('timestamp', 'unknown')}\n")
        f.write(f"File: {diagnosis.get('file_analyzed', 'unknown')}\n")
        f.write(f"System Health Score: {diagnosis.get('system_health_score', 0):.1f}/100\n\n")
        
        # CPU Analysis
        cpu_data = results.get('cpu_thermal', {})
        f.write("CPU THERMAL ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Overall Status: {cpu_data.get('overall_status', 'unknown')}\n")
        f.write(f"Health Score: {cpu_data.get('health_score', 0)}/100\n\n")
        
        for core_name, core_data in cpu_data.get('individual_cores', {}).items():
            f.write(f"{core_name}:\n")
            f.write(f"  Mean: {core_data['mean_temp']:.1f}°C ({core_data['mean_classification']})\n")
            f.write(f"  Max: {core_data['max_temp']:.1f}°C ({core_data['max_classification']})\n")
            f.write(f"  Stability: {core_data['stability']}\n")
            if core_data['issues']:
                for issue in core_data['issues']:
                    f.write(f"  Issue: {issue}\n")
            f.write("\n")
        
        # GPU Analysis
        gpu_data = results.get('gpu_thermal', {})
        f.write("GPU THERMAL ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Overall Status: {gpu_data.get('overall_status', 'unknown')}\n")
        f.write(f"Health Score: {gpu_data.get('health_score', 0)}/100\n\n")
        
        for gpu_name, gpu_sensor_data in gpu_data.get('individual_gpus', {}).items():
            f.write(f"{gpu_name}:\n")
            f.write(f"  Mean: {gpu_sensor_data['mean_temp']:.1f}°C ({gpu_sensor_data['mean_classification']})\n")
            f.write(f"  Max: {gpu_sensor_data['max_temp']:.1f}°C ({gpu_sensor_data['max_classification']})\n")
            f.write(f"  Stability: {gpu_sensor_data['stability']}\n")
            if gpu_sensor_data['issues']:
                for issue in gpu_sensor_data['issues']:
                    f.write(f"  Issue: {issue}\n")
            f.write("\n")
        
        # Voltage Analysis
        voltage_data = results.get('voltage', {})
        f.write("VOLTAGE ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Overall Stability: {voltage_data.get('overall_stability', 'unknown')}\n")
        f.write(f"Health Score: {voltage_data.get('health_score', 0)}/100\n\n")
        
        # Show most problematic voltage rails
        rail_issues = []
        for rail_name, rail_data in voltage_data.get('rail_analyses', {}).items():
            if rail_data['issues']:
                rail_issues.append((rail_name, rail_data['issues']))
        
        if rail_issues:
            f.write("Problematic Voltage Rails:\n")
            for rail_name, issues in rail_issues[:5]:  # Show top 5
                f.write(f"  {rail_name}:\n")
                for issue in issues:
                    f.write(f"    - {issue}\n")
        
        # Recommendations
        f.write("\nRECOMMENDATIONS\n")
        f.write("-" * 15 + "\n")
        for rec in diagnosis.get('recommendations', []):
            f.write(f"• {rec}\n")

if __name__ == "__main__":
    main()