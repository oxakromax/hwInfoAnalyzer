"""
Anomaly Detection Module
Detects anomalies in temperature and voltage data using multiple algorithms.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class AnomalyDetector:
    """Detects anomalies in sensor data."""
    
    def __init__(self, contamination_rate=0.05):
        self.contamination_rate = contamination_rate
        self.methods = ['isolation_forest', 'statistical', 'iqr']
    
    def detect_anomalies(self, data, method='isolation_forest'):
        """Detect anomalies in a data series."""
        if len(data) < 10:
            return self._empty_result(len(data))
        
        # Clean data
        clean_data = data.dropna()
        clean_data = clean_data[np.isfinite(clean_data)]
        
        if len(clean_data) < 10:
            return self._empty_result(len(data))
        
        # Detect anomalies based on method
        if method == 'isolation_forest':
            anomaly_indices = self._isolation_forest_detection(clean_data)
        elif method == 'statistical':
            anomaly_indices = self._statistical_detection(clean_data)
        elif method == 'iqr':
            anomaly_indices = self._iqr_detection(clean_data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return self._compile_results(clean_data, anomaly_indices)
    
    def _empty_result(self, data_length):
        """Return empty result structure."""
        return {
            'total_points': data_length,
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
            },
            'severity': 'none'
        }
    
    def _isolation_forest_detection(self, data):
        """Use Isolation Forest for anomaly detection."""
        try:
            iso_forest = IsolationForest(
                contamination=self.contamination_rate, 
                random_state=42,
                n_estimators=100
            )
            anomaly_labels = iso_forest.fit_predict(data.values.reshape(-1, 1))
            return np.where(anomaly_labels == -1)[0]
        except Exception as e:
            print(f"Warning: Isolation Forest failed: {e}, falling back to statistical method")
            return self._statistical_detection(data)
    
    def _statistical_detection(self, data):
        """Use Z-score for anomaly detection."""
        z_scores = np.abs(stats.zscore(data))
        # Use 3 sigma rule, but adapt based on data distribution
        threshold = 3.0 if data.std() > 0 else float('inf')
        return np.where(z_scores > threshold)[0]
    
    def _iqr_detection(self, data):
        """Use Interquartile Range for anomaly detection."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Use 1.5 * IQR rule
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return np.where((data < lower_bound) | (data > upper_bound))[0]
    
    def _compile_results(self, data, anomaly_indices):
        """Compile detection results."""
        anomaly_values = data.iloc[anomaly_indices].tolist() if len(anomaly_indices) > 0 else []
        anomaly_percentage = len(anomaly_indices) / len(data) * 100
        
        # Determine severity based on percentage and values
        severity = self._determine_severity(anomaly_percentage, anomaly_values, data)
        
        return {
            'total_points': len(data),
            'anomalies': anomaly_values,
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_percentage': anomaly_percentage,
            'statistics': {
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'q1': data.quantile(0.25),
                'q3': data.quantile(0.75)
            },
            'severity': severity
        }
    
    def _determine_severity(self, percentage, anomaly_values, data):
        """Determine the severity of anomalies."""
        if percentage == 0:
            return 'none'
        elif percentage < 2:
            return 'low'
        elif percentage < 5:
            return 'moderate'
        elif percentage < 10:
            return 'high'
        else:
            return 'severe'
    
    def detect_patterns(self, data):
        """Detect patterns in time series data."""
        if len(data) < 20:
            return {
                'trend': 'insufficient_data',
                'cyclical': 'insufficient_data',
                'stability': 'insufficient_data',
                'change_points': []
            }
        
        clean_data = data.dropna()
        clean_data = clean_data[np.isfinite(clean_data)]
        
        if len(clean_data) < 20:
            return {
                'trend': 'insufficient_data',
                'cyclical': 'insufficient_data',
                'stability': 'insufficient_data',
                'change_points': []
            }
        
        return {
            'trend': self._detect_trend(clean_data),
            'cyclical': self._detect_cyclical(clean_data),
            'stability': self._assess_stability(clean_data),
            'change_points': self._detect_change_points(clean_data)
        }
    
    def _detect_trend(self, data):
        """Detect overall trend in the data."""
        if len(data) < 10:
            return "insufficient_data"
        
        try:
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            
            if p_value < 0.05:  # Significant trend
                if abs(slope) < 0.001:  # Very small slope
                    return "stable"
                elif slope > 0:
                    return f"increasing (slope: {slope:.4f}, R²: {r_value**2:.3f})"
                else:
                    return f"decreasing (slope: {slope:.4f}, R²: {r_value**2:.3f})"
            else:
                return "stable (no significant trend)"
        except Exception:
            return "unable_to_determine"
    
    def _detect_cyclical(self, data):
        """Detect cyclical patterns using autocorrelation."""
        if len(data) < 30:
            return "insufficient_data"
        
        try:
            # Calculate autocorrelation
            data_normalized = (data - data.mean()) / data.std()
            autocorr = np.correlate(data_normalized, data_normalized, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Look for peaks in autocorrelation (skip lag 0)
            min_period = 3  # Minimum period to consider
            max_period = min(len(data) // 4, 50)  # Maximum period to check
            
            significant_autocorr = autocorr[min_period:max_period]
            if len(significant_autocorr) > 0:
                max_autocorr = np.max(significant_autocorr)
                if max_autocorr > 0.3:  # Threshold for significant cyclical pattern
                    period = np.argmax(significant_autocorr) + min_period
                    return f"cyclical pattern detected (period: ~{period} samples, strength: {max_autocorr:.3f})"
            
            return "no clear cyclical pattern"
        except Exception:
            return "unable_to_determine"
    
    def _assess_stability(self, data):
        """Assess the stability of the data."""
        if data.mean() == 0:
            return "unable_to_assess (zero mean)"
        
        cv = data.std() / abs(data.mean())  # Coefficient of variation
        
        if cv < 0.02:
            return "very stable"
        elif cv < 0.05:
            return "stable"
        elif cv < 0.10:
            return "moderately stable"
        elif cv < 0.20:
            return "unstable"
        else:
            return "very unstable"
    
    def _detect_change_points(self, data):
        """Detect significant change points in the data."""
        if len(data) < 20:
            return []
        
        try:
            # Simple change point detection using rolling statistics
            window = max(5, len(data) // 10)
            rolling_mean = data.rolling(window=window, center=True).mean()
            
            # Calculate differences in rolling mean
            mean_diff = rolling_mean.diff().abs()
            threshold = mean_diff.quantile(0.9)  # Top 10% of changes
            
            change_points = data.index[mean_diff > threshold].tolist()
            return change_points[:5]  # Return up to 5 most significant change points
        except Exception:
            return []