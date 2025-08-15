#!/usr/bin/env python3
"""
Command Line Interface for HWiNFO Analyzer
Entry point for the application
"""

import argparse
import os
import sys
from datetime import datetime

from .core.analyzer import HWInfoAnalyzer
from .config.analysis_methods import AnalysisMethodSelector


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='HWiNFO CSV Log Analyzer with Thermal Monitoring and Visualizations',
        prog='hwinfo-analyzer'
    )
    parser.add_argument('csv_file', help='Path to HWiNFO CSV log file')
    parser.add_argument('--output', '-o', help='Output directory for detailed reports and plots')
    parser.add_argument('--config', '-c', help='Configuration file for analysis methods')
    parser.add_argument('--no-plots', action='store_true', help='Disable visualization generation')
    parser.add_argument('--preset', choices=['comprehensive', 'minimal', 'thermal_focus', 'voltage_focus'], 
                       help='Use predefined analysis preset')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found!")
        sys.exit(1)
    
    # Setup configuration
    config_file = args.config
    if args.preset:
        config_file = f"preset_{args.preset}.json"
        if not os.path.exists(config_file):
            print(f"Creating preset configuration: {config_file}")
            selector = AnalysisMethodSelector()
            selector.create_preset_configs()
    
    # Setup output directory
    output_dir = args.output or "analysis_output"
    
    try:
        # Run analysis
        analyzer = HWInfoAnalyzer(
            args.csv_file, 
            config_file=config_file,
            enable_plots=not args.no_plots,
            output_dir=output_dir
        )
        results = analyzer.run_analysis()
        
        # Print summary
        analyzer.print_summary()
        
        # Save detailed report
        os.makedirs(output_dir, exist_ok=True)
        report_file = os.path.join(output_dir, 'detailed_analysis.txt')
        save_detailed_report(results, report_file)
        print(f"\nDetailed report saved to: {report_file}")
        
        # Print summary of generated files
        print(f"\nAnalysis complete! Files generated in: {output_dir}")
        if results.get('plots'):
            print(f"Visualization files: {len(results['plots'])}")
            print(f"View plots directory: {output_dir}/plots/")
    
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


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
        f.write(f"Health Score: {cpu_data.get('health_score', 0)}/100\n")
        f.write(f"Architecture: {cpu_data.get('cpu_architecture', 'unknown')}\n\n")
        
        # Show top 10 cores by issues
        cores_with_issues = []
        for core_name, core_data in cpu_data.get('individual_cores', {}).items():
            if core_data.get('issues'):
                cores_with_issues.append((core_name, core_data))
        
        cores_with_issues.sort(key=lambda x: len(x[1]['issues']), reverse=True)
        
        for core_name, core_data in cores_with_issues[:10]:
            f.write(f"{core_name}:\n")
            f.write(f"  Core Type: {core_data.get('core_type', 'standard')}\n")
            f.write(f"  Mean: {core_data['mean_temp']:.1f}°C ({core_data['mean_classification']})\n")
            f.write(f"  Max: {core_data['max_temp']:.1f}°C ({core_data['max_classification']})\n")
            f.write(f"  Stability: {core_data['stability']}\n")
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