#!/usr/bin/env python3
"""
HWiNFO Log Analyzer
Detects anomalies, patterns, and peaks in temperature/voltage data from HWiNFO CSV logs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import warnings
import argparse
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class HWInfoAnalyzer:
    def __init__(self, csv_file):
        """Initialize the analyzer with a CSV file."""
        self.csv_file = csv_file
        self.df = None
        self.temp_columns = []
        self.voltage_columns = []
        self.analysis_results = {}
        
    def load_data(self):
        """Load and preprocess the CSV data."""
        print(f"Loading data from {self.csv_file}...")
        
        # Load CSV with error handling for malformed lines
        try:
            self.df = pd.read_csv(self.csv_file, on_bad_lines='skip', low_memory=False)
        except Exception as e:
            print(f"Error reading CSV with standard method: {e}")
            print("Trying alternative reading method...")
            # Try with different encoding and error handling
            try:
                self.df = pd.read_csv(self.csv_file, encoding='utf-8-sig', on_bad_lines='skip', 
                                    low_memory=False, skipinitialspace=True)
            except Exception as e2:
                print(f"Error with UTF-8-sig encoding: {e2}")
                print("Trying with latin1 encoding...")
                self.df = pd.read_csv(self.csv_file, encoding='latin1', on_bad_lines='skip', 
                                    low_memory=False, skipinitialspace=True)
        
        # Clean up the dataframe
        print(f"Initial data shape: {self.df.shape}")
        
        # Remove completely empty rows
        self.df = self.df.dropna(how='all')
        print(f"After removing empty rows: {self.df.shape}")
        
        # Combine Date and Time columns into datetime
        if 'Date' in self.df.columns and 'Time' in self.df.columns:
            # Try different datetime formats
            try:
                self.df['DateTime'] = pd.to_datetime(
                    self.df['Date'].astype(str) + ' ' + self.df['Time'].astype(str), 
                    format='%d.%m.%Y %H:%M:%S.%f',
                    errors='coerce'
                )
            except:
                try:
                    # Alternative format without microseconds
                    self.df['DateTime'] = pd.to_datetime(
                        self.df['Date'].astype(str) + ' ' + self.df['Time'].astype(str), 
                        format='%d.%m.%Y %H:%M:%S',
                        errors='coerce'
                    )
                except:
                    print("Warning: Could not parse datetime. Using sequential index.")
                    self.df['DateTime'] = pd.date_range(start='2024-01-01', periods=len(self.df), freq='1S')
            
            # Remove rows with invalid datetime
            self.df = self.df.dropna(subset=['DateTime'])
            self.df = self.df.set_index('DateTime')
        else:
            print("Warning: Date/Time columns not found. Using sequential index.")
            self.df.index = pd.date_range(start='2024-01-01', periods=len(self.df), freq='1S')
        
        # Identify temperature and voltage columns
        # Handle different encodings for degree symbol
        self.temp_columns = [col for col in self.df.columns if 
                           '[°C]' in col or '[�C]' in col or 'Temperature' in col or 'Temp' in col]
        self.voltage_columns = [col for col in self.df.columns if 
                              '[V]' in col and 'VID' not in col and 'RPM' not in col and 'MHz' not in col]
        
        # Filter out non-numeric columns
        numeric_temp_columns = []
        for col in self.temp_columns:
            try:
                pd.to_numeric(self.df[col], errors='coerce')
                numeric_temp_columns.append(col)
            except:
                continue
        self.temp_columns = numeric_temp_columns
        
        numeric_voltage_columns = []
        for col in self.voltage_columns:
            try:
                pd.to_numeric(self.df[col], errors='coerce')
                numeric_voltage_columns.append(col)
            except:
                continue
        self.voltage_columns = numeric_voltage_columns
        
        print(f"Found {len(self.temp_columns)} temperature columns")
        print(f"Found {len(self.voltage_columns)} voltage columns")
        print(f"Loaded {len(self.df)} data points")
        
    def detect_anomalies(self, method='isolation_forest'):
        """Detect anomalies in temperature and voltage data."""
        print("\nDetecting anomalies...")
        
        anomalies = {}
        
        # Analyze temperature columns
        for col in self.temp_columns:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 10:  # Need sufficient data points
                    anomalies[col] = self._detect_column_anomalies(data, method)
        
        # Analyze voltage columns
        for col in self.voltage_columns:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 10:
                    anomalies[col] = self._detect_column_anomalies(data, method)
        
        self.analysis_results['anomalies'] = anomalies
        return anomalies
    
    def _detect_column_anomalies(self, data, method='isolation_forest'):
        """Detect anomalies in a single column."""
        # Convert to numeric and remove non-finite values
        data = pd.to_numeric(data, errors='coerce').dropna()
        data = data[np.isfinite(data)]
        
        if len(data) < 10:
            return {
                'total_points': len(data),
                'anomalies': [],
                'anomaly_indices': [],
                'anomaly_percentage': 0,
                'statistics': {
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'q1': np.nan,
                    'q3': np.nan
                }
            }
        
        anomaly_info = {
            'total_points': len(data),
            'anomalies': [],
            'anomaly_indices': [],
            'statistics': {
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'q1': data.quantile(0.25),
                'q3': data.quantile(0.75)
            }
        }
        
        if method == 'isolation_forest':
            # Isolation Forest
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            anomaly_labels = iso_forest.fit_predict(data.values.reshape(-1, 1))
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
        elif method == 'statistical':
            # Statistical method (Z-score > 3)
            z_scores = np.abs(stats.zscore(data))
            anomaly_indices = np.where(z_scores > 3)[0]
            
        elif method == 'iqr':
            # Interquartile Range method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomaly_indices = data[(data < lower_bound) | (data > upper_bound)].index
        
        # Store anomaly information
        anomaly_info['anomaly_indices'] = anomaly_indices.tolist() if hasattr(anomaly_indices, 'tolist') else anomaly_indices
        anomaly_info['anomalies'] = data.iloc[anomaly_indices].tolist() if len(anomaly_indices) > 0 else []
        anomaly_info['anomaly_percentage'] = len(anomaly_indices) / len(data) * 100
        
        return anomaly_info
    
    def detect_patterns(self):
        """Detect patterns in the data."""
        print("\nDetecting patterns...")
        
        patterns = {}
        
        # Analyze temperature patterns
        for col in self.temp_columns[:5]:  # Limit to first 5 for performance
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 50:
                    patterns[col] = self._analyze_column_patterns(data)
        
        # Analyze voltage patterns
        for col in self.voltage_columns[:5]:  # Limit to first 5 for performance
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 50:
                    patterns[col] = self._analyze_column_patterns(data)
        
        self.analysis_results['patterns'] = patterns
        return patterns
    
    def _analyze_column_patterns(self, data):
        """Analyze patterns in a single column."""
        # Convert to numeric and clean data
        data = pd.to_numeric(data, errors='coerce').dropna()
        data = data[np.isfinite(data)]
        
        if len(data) < 10:
            return {
                'trend': 'insufficient_data',
                'cyclical': 'insufficient_data',
                'stability': 'insufficient_data',
                'change_points': []
            }
        
        pattern_info = {
            'trend': self._detect_trend(data),
            'cyclical': self._detect_cyclical(data),
            'stability': self._assess_stability(data),
            'change_points': self._detect_change_points(data)
        }
        return pattern_info
    
    def _detect_trend(self, data):
        """Detect overall trend in the data."""
        if len(data) < 10:
            return "insufficient_data"
        
        # Linear regression to detect trend
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        if p_value < 0.05:  # Significant trend
            if slope > 0:
                return f"increasing (slope: {slope:.4f})"
            else:
                return f"decreasing (slope: {slope:.4f})"
        else:
            return "stable"
    
    def _detect_cyclical(self, data):
        """Detect cyclical patterns using autocorrelation."""
        if len(data) < 20:
            return "insufficient_data"
        
        # Calculate autocorrelation
        autocorr = np.correlate(data - data.mean(), data - data.mean(), mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find peaks in autocorrelation (potential periods)
        peaks, _ = find_peaks(autocorr[1:20], height=0.3)  # Skip lag 0, check first 20 lags
        
        if len(peaks) > 0:
            return f"cyclical pattern detected (period: ~{peaks[0]+1} samples)"
        else:
            return "no clear cyclical pattern"
    
    def _assess_stability(self, data):
        """Assess the stability of the data."""
        cv = data.std() / data.mean() if data.mean() != 0 else float('inf')
        
        if cv < 0.05:
            return "very stable"
        elif cv < 0.1:
            return "stable"
        elif cv < 0.2:
            return "moderately stable"
        else:
            return "unstable"
    
    def _detect_change_points(self, data):
        """Detect significant change points in the data."""
        if len(data) < 20:
            return []
        
        # Simple change point detection using rolling statistics
        window = min(10, len(data) // 5)
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        
        # Look for significant changes in mean
        mean_changes = np.abs(rolling_mean.diff()) > (2 * rolling_std.shift(1))
        change_points = data.index[mean_changes].tolist()
        
        return change_points[:5]  # Return up to 5 change points
    
    def detect_peaks(self):
        """Detect temperature and voltage peaks."""
        print("\nDetecting peaks...")
        
        peaks_info = {}
        
        # Analyze temperature peaks
        for col in self.temp_columns:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 20:
                    peaks_info[col] = self._find_column_peaks(data, col)
        
        # Analyze voltage peaks
        for col in self.voltage_columns:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 20:
                    peaks_info[col] = self._find_column_peaks(data, col)
        
        self.analysis_results['peaks'] = peaks_info
        return peaks_info
    
    def _find_column_peaks(self, data, column_name):
        """Find peaks in a single column."""
        # Convert to numeric and clean data
        data = pd.to_numeric(data, errors='coerce').dropna()
        data = data[np.isfinite(data)]
        
        if len(data) < 20:
            return {
                'peak_count': 0,
                'peak_indices': [],
                'peak_values': [],
                'peak_times': [],
                'max_peak': None,
                'min_peak': None
            }
        
        try:
            # For temperatures, find high peaks
            if '[°C]' in column_name or '[�C]' in column_name or 'Temperature' in column_name:
                height_threshold = data.quantile(0.9)  # 90th percentile
                peaks, properties = find_peaks(data.values, height=height_threshold, distance=5)
            
            # For voltages, find both high and low peaks
            else:
                # High peaks
                height_threshold_high = data.quantile(0.95)
                peaks_high, props_high = find_peaks(data.values, height=height_threshold_high, distance=5)
                
                # Low peaks (inverted signal)
                height_threshold_low = data.quantile(0.05)
                peaks_low, props_low = find_peaks(-data.values, height=-height_threshold_low, distance=5)
                
                peaks = np.concatenate([peaks_high, peaks_low]) if len(peaks_high) > 0 or len(peaks_low) > 0 else np.array([])
                
        except Exception as e:
            print(f"Warning: Error finding peaks in {column_name}: {e}")
            peaks = np.array([])
        
        peak_info = {
            'peak_count': len(peaks),
            'peak_indices': peaks.tolist(),
            'peak_values': data.iloc[peaks].tolist() if len(peaks) > 0 else [],
            'peak_times': data.index[peaks].tolist() if len(peaks) > 0 else [],
            'max_peak': data.iloc[peaks].max() if len(peaks) > 0 else None,
            'min_peak': data.iloc[peaks].min() if len(peaks) > 0 else None
        }
        
        return peak_info
    
    def generate_diagnosis(self):
        """Generate a comprehensive diagnosis based on all analyses."""
        print("\nGenerating diagnosis...")
        
        diagnosis = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file_analyzed': self.csv_file,
            'summary': {},
            'warnings': [],
            'recommendations': []
        }
        
        # Temperature analysis
        temp_issues = self._diagnose_temperatures()
        diagnosis['temperature_analysis'] = temp_issues
        
        # Voltage analysis
        voltage_issues = self._diagnose_voltages()
        diagnosis['voltage_analysis'] = voltage_issues
        
        # Overall system health
        diagnosis['overall_health'] = self._assess_overall_health(temp_issues, voltage_issues)
        
        self.analysis_results['diagnosis'] = diagnosis
        return diagnosis
    
    def _diagnose_temperatures(self):
        """Diagnose temperature-related issues."""
        temp_diagnosis = {
            'critical_temps': [],
            'overheating_detected': False,
            'thermal_throttling_risk': False,
            'temperature_spikes': []
        }
        
        if 'anomalies' not in self.analysis_results:
            return temp_diagnosis
        
        for col in self.temp_columns:
            if col in self.analysis_results['anomalies']:
                anomaly_data = self.analysis_results['anomalies'][col]
                max_temp = anomaly_data['statistics']['max']
                
                # Skip if max_temp is NaN
                if pd.isna(max_temp):
                    continue
                
                # Critical temperature thresholds
                if 'CPU' in col.upper() and max_temp > 85:
                    temp_diagnosis['critical_temps'].append(f"{col}: {max_temp:.1f}degC")
                    temp_diagnosis['overheating_detected'] = True
                elif 'GPU' in col.upper() and max_temp > 83:
                    temp_diagnosis['critical_temps'].append(f"{col}: {max_temp:.1f}degC")
                    temp_diagnosis['overheating_detected'] = True
                elif max_temp > 90:  # General high temp threshold
                    temp_diagnosis['critical_temps'].append(f"{col}: {max_temp:.1f}degC")
                    temp_diagnosis['overheating_detected'] = True
                
                # Check for thermal throttling risk
                if max_temp > 80:
                    temp_diagnosis['thermal_throttling_risk'] = True
                
                # Check for temperature spikes
                if anomaly_data['anomaly_percentage'] > 5:
                    temp_diagnosis['temperature_spikes'].append({
                        'component': col,
                        'spike_percentage': anomaly_data['anomaly_percentage'],
                        'max_anomaly': max(anomaly_data['anomalies']) if anomaly_data['anomalies'] else 0
                    })
        
        return temp_diagnosis
    
    def _diagnose_voltages(self):
        """Diagnose voltage-related issues."""
        voltage_diagnosis = {
            'voltage_instability': [],
            'overvoltage_detected': False,
            'undervoltage_detected': False,
            'voltage_spikes': []
        }
        
        if 'anomalies' not in self.analysis_results:
            return voltage_diagnosis
        
        for col in self.voltage_columns:
            if col in self.analysis_results['anomalies']:
                anomaly_data = self.analysis_results['anomalies'][col]
                
                # Check voltage stability (skip if mean or std is NaN)
                mean_val = anomaly_data['statistics']['mean']
                std_val = anomaly_data['statistics']['std']
                
                if not pd.isna(mean_val) and not pd.isna(std_val) and mean_val != 0:
                    cv = std_val / mean_val
                    if cv > 0.05:  # More than 5% coefficient of variation
                        voltage_diagnosis['voltage_instability'].append({
                            'rail': col,
                            'instability': f"{cv*100:.2f}%",
                            'std_dev': std_val
                        })
                
                # Check for voltage anomalies
                if anomaly_data['anomaly_percentage'] > 3:
                    voltage_diagnosis['voltage_spikes'].append({
                        'rail': col,
                        'spike_percentage': anomaly_data['anomaly_percentage'],
                        'anomaly_count': len(anomaly_data['anomalies'])
                    })
        
        return voltage_diagnosis
    
    def _assess_overall_health(self, temp_issues, voltage_issues):
        """Assess overall system health."""
        health_score = 100
        issues = []
        
        # Temperature-based deductions
        if temp_issues['overheating_detected']:
            health_score -= 30
            issues.append("Critical overheating detected")
        
        if temp_issues['thermal_throttling_risk']:
            health_score -= 15
            issues.append("Thermal throttling risk")
        
        if len(temp_issues['temperature_spikes']) > 0:
            health_score -= 10 * len(temp_issues['temperature_spikes'])
            issues.append(f"Temperature spikes in {len(temp_issues['temperature_spikes'])} components")
        
        # Voltage-based deductions
        if len(voltage_issues['voltage_instability']) > 0:
            health_score -= 15 * len(voltage_issues['voltage_instability'])
            issues.append(f"Voltage instability in {len(voltage_issues['voltage_instability'])} rails")
        
        if len(voltage_issues['voltage_spikes']) > 0:
            health_score -= 10 * len(voltage_issues['voltage_spikes'])
            issues.append(f"Voltage spikes in {len(voltage_issues['voltage_spikes'])} rails")
        
        health_score = max(0, health_score)  # Don't go below 0
        
        if health_score >= 90:
            status = "Excellent"
        elif health_score >= 75:
            status = "Good"
        elif health_score >= 50:
            status = "Fair"
        elif health_score >= 25:
            status = "Poor"
        else:
            status = "Critical"
        
        return {
            'score': health_score,
            'status': status,
            'issues': issues
        }
    
    def create_visualizations(self, output_dir='plots'):
        """Create visualization plots."""
        print(f"\nCreating visualizations in {output_dir}...")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot temperature trends
        self._plot_temperature_trends(output_dir)
        
        # Plot voltage trends
        self._plot_voltage_trends(output_dir)
        
        # Plot anomalies
        self._plot_anomalies(output_dir)
        
    def _plot_temperature_trends(self, output_dir):
        """Plot temperature trends."""
        if not self.temp_columns:
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot up to 8 temperature sensors
        for i, col in enumerate(self.temp_columns[:8]):
            if col in self.df.columns:
                plt.subplot(2, 4, i+1)
                data = self.df[col].dropna()
                plt.plot(data.index, data.values)
                plt.title(col.replace('[°C]', '').replace('Temperature', 'Temp'), fontsize=10)
                plt.xticks(rotation=45)
                plt.ylabel('Temperature (°C)')
                
                # Mark anomalies if available
                if 'anomalies' in self.analysis_results and col in self.analysis_results['anomalies']:
                    anomaly_indices = self.analysis_results['anomalies'][col]['anomaly_indices']
                    if anomaly_indices:
                        anomaly_values = data.iloc[anomaly_indices]
                        plt.scatter(anomaly_values.index, anomaly_values.values, 
                                   color='red', s=20, alpha=0.7, label='Anomalies')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temperature_trends.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_voltage_trends(self, output_dir):
        """Plot voltage trends."""
        if not self.voltage_columns:
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot up to 8 voltage rails
        for i, col in enumerate(self.voltage_columns[:8]):
            if col in self.df.columns:
                plt.subplot(2, 4, i+1)
                data = self.df[col].dropna()
                plt.plot(data.index, data.values)
                plt.title(col.replace('[V]', ''), fontsize=10)
                plt.xticks(rotation=45)
                plt.ylabel('Voltage (V)')
                
                # Mark anomalies if available
                if 'anomalies' in self.analysis_results and col in self.analysis_results['anomalies']:
                    anomaly_indices = self.analysis_results['anomalies'][col]['anomaly_indices']
                    if anomaly_indices:
                        anomaly_values = data.iloc[anomaly_indices]
                        plt.scatter(anomaly_values.index, anomaly_values.values, 
                                   color='red', s=20, alpha=0.7, label='Anomalies')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'voltage_trends.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_anomalies(self, output_dir):
        """Plot anomaly summary."""
        if 'anomalies' not in self.analysis_results:
            return
        
        # Prepare data for anomaly summary
        components = []
        anomaly_percentages = []
        
        for col, anomaly_data in self.analysis_results['anomalies'].items():
            if anomaly_data['anomaly_percentage'] > 0:
                components.append(col.replace('[°C]', '').replace('[V]', '').replace('Temperature', 'Temp')[:20])
                anomaly_percentages.append(anomaly_data['anomaly_percentage'])
        
        if not components:
            return
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(components, anomaly_percentages)
        plt.title('Anomaly Percentage by Component')
        plt.xlabel('Component')
        plt.ylabel('Anomaly Percentage (%)')
        plt.xticks(rotation=45, ha='right')
        
        # Color bars based on severity
        for i, bar in enumerate(bars):
            if anomaly_percentages[i] > 10:
                bar.set_color('red')
            elif anomaly_percentages[i] > 5:
                bar.set_color('orange')
            else:
                bar.set_color('yellow')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'anomaly_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_report(self, output_file='hwinfo_analysis_report.txt'):
        """Save a comprehensive text report."""
        print(f"\nSaving report to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("HWiNFO LOG ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            if 'diagnosis' in self.analysis_results:
                diagnosis = self.analysis_results['diagnosis']
                f.write(f"Analysis Date: {diagnosis['timestamp']}\n")
                f.write(f"File Analyzed: {diagnosis['file_analyzed']}\n\n")
                
                # Overall Health
                health = diagnosis['overall_health']
                f.write("OVERALL SYSTEM HEALTH\n")
                f.write("-" * 25 + "\n")
                f.write(f"Health Score: {health['score']}/100 ({health['status']})\n")
                if health['issues']:
                    f.write("Issues Detected:\n")
                    for issue in health['issues']:
                        f.write(f"  - {issue}\n")
                f.write("\n")
                
                # Temperature Analysis
                temp_analysis = diagnosis['temperature_analysis']
                f.write("TEMPERATURE ANALYSIS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Overheating Detected: {'Yes' if temp_analysis['overheating_detected'] else 'No'}\n")
                f.write(f"Thermal Throttling Risk: {'Yes' if temp_analysis['thermal_throttling_risk'] else 'No'}\n")
                
                if temp_analysis['critical_temps']:
                    f.write("Critical Temperatures:\n")
                    for temp in temp_analysis['critical_temps']:
                        # Clean up any problematic characters
                        clean_temp = temp.replace('°', 'deg').replace('�', 'deg')
                        f.write(f"  - {clean_temp}\n")
                
                if temp_analysis['temperature_spikes']:
                    f.write("Temperature Spikes:\n")
                    for spike in temp_analysis['temperature_spikes']:
                        f.write(f"  - {spike['component']}: {spike['spike_percentage']:.1f}% anomalies\n")
                f.write("\n")
                
                # Voltage Analysis
                voltage_analysis = diagnosis['voltage_analysis']
                f.write("VOLTAGE ANALYSIS\n")
                f.write("-" * 16 + "\n")
                
                if voltage_analysis['voltage_instability']:
                    f.write("Voltage Instability:\n")
                    for instability in voltage_analysis['voltage_instability']:
                        f.write(f"  - {instability['rail']}: {instability['instability']} variation\n")
                
                if voltage_analysis['voltage_spikes']:
                    f.write("Voltage Spikes:\n")
                    for spike in voltage_analysis['voltage_spikes']:
                        f.write(f"  - {spike['rail']}: {spike['spike_percentage']:.1f}% anomalies\n")
                f.write("\n")
            
            # Detailed Statistics
            if 'anomalies' in self.analysis_results:
                f.write("DETAILED STATISTICS\n")
                f.write("-" * 19 + "\n")
                
                for col, anomaly_data in self.analysis_results['anomalies'].items():
                    if anomaly_data['anomaly_percentage'] > 0:
                        f.write(f"\n{col}:\n")
                        stats = anomaly_data['statistics']
                        f.write(f"  Mean: {stats['mean']:.3f}\n")
                        f.write(f"  Std Dev: {stats['std']:.3f}\n")
                        f.write(f"  Min: {stats['min']:.3f}\n")
                        f.write(f"  Max: {stats['max']:.3f}\n")
                        f.write(f"  Anomalies: {len(anomaly_data['anomalies'])} ({anomaly_data['anomaly_percentage']:.1f}%)\n")
        
        print(f"Report saved successfully!")
    
    def run_full_analysis(self, output_dir='analysis_output'):
        """Run complete analysis pipeline."""
        print("Starting HWiNFO Analysis...")
        print("="*50)
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load data
        self.load_data()
        
        # Run analyses
        self.detect_anomalies()
        self.detect_patterns()
        self.detect_peaks()
        self.generate_diagnosis()
        
        # Create outputs
        plots_dir = os.path.join(output_dir, 'plots')
        self.create_visualizations(plots_dir)
        
        report_file = os.path.join(output_dir, 'analysis_report.txt')
        self.save_report(report_file)
        
        print("\n" + "="*50)
        print("Analysis Complete!")
        print(f"Results saved to: {output_dir}")
        
        # Print quick summary
        if 'diagnosis' in self.analysis_results:
            diagnosis = self.analysis_results['diagnosis']
            health = diagnosis['overall_health']
            print(f"System Health: {health['score']}/100 ({health['status']})")
            
            if health['issues']:
                print("Key Issues:")
                for issue in health['issues'][:3]:  # Show top 3 issues
                    print(f"  - {issue}")

def main():
    parser = argparse.ArgumentParser(description='Analyze HWiNFO CSV logs for anomalies and patterns')
    parser.add_argument('csv_file', help='Path to HWiNFO CSV log file')
    parser.add_argument('--output', '-o', default='analysis_output', 
                       help='Output directory for results (default: analysis_output)')
    parser.add_argument('--method', '-m', choices=['isolation_forest', 'statistical', 'iqr'],
                       default='isolation_forest', help='Anomaly detection method')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found!")
        return
    
    # Run analysis
    analyzer = HWInfoAnalyzer(args.csv_file)
    analyzer.run_full_analysis(args.output)

if __name__ == "__main__":
    main()