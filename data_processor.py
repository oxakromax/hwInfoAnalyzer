"""
Data Processing Module
Handles CSV loading, cleaning, and preprocessing of HWiNFO data.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class HWInfoDataProcessor:
    """Processes and cleans HWiNFO CSV data."""
    
    def __init__(self):
        self.df = None
        self.temp_columns = []
        self.voltage_columns = []
        self.cpu_temp_columns = []
        self.gpu_temp_columns = []
        self.motherboard_temp_columns = []
        
    def load_csv(self, csv_file):
        """Load CSV with robust error handling."""
        print(f"Loading data from {csv_file}...")
        
        try:
            self.df = pd.read_csv(csv_file, on_bad_lines='skip', low_memory=False)
        except Exception as e:
            print(f"Error reading CSV with standard method: {e}")
            print("Trying alternative reading methods...")
            
            # Try different encodings
            for encoding in ['utf-8-sig', 'latin1', 'cp1252']:
                try:
                    self.df = pd.read_csv(csv_file, encoding=encoding, on_bad_lines='skip', 
                                        low_memory=False, skipinitialspace=True)
                    print(f"Successfully loaded with {encoding} encoding")
                    break
                except Exception as e2:
                    print(f"Failed with {encoding}: {e2}")
                    continue
            else:
                raise Exception("Could not read CSV file with any encoding method")
        
        return self._process_dataframe()
    
    def _process_dataframe(self):
        """Clean and process the loaded dataframe."""
        print(f"Initial data shape: {self.df.shape}")
        
        # Remove completely empty rows
        self.df = self.df.dropna(how='all')
        print(f"After removing empty rows: {self.df.shape}")
        
        # Process datetime
        self._process_datetime()
        
        # Identify column types
        self._identify_columns()
        
        # Clean numeric data
        self._clean_numeric_data()
        
        print(f"Found {len(self.cpu_temp_columns)} CPU temperature columns")
        print(f"Found {len(self.gpu_temp_columns)} GPU temperature columns")
        print(f"Found {len(self.motherboard_temp_columns)} other temperature columns")
        print(f"Found {len(self.voltage_columns)} voltage columns")
        print(f"Final data points: {len(self.df)}")
        
        return self.df
    
    def _process_datetime(self):
        """Process date and time columns."""
        if 'Date' in self.df.columns and 'Time' in self.df.columns:
            try:
                # Try different datetime formats
                self.df['DateTime'] = pd.to_datetime(
                    self.df['Date'].astype(str) + ' ' + self.df['Time'].astype(str), 
                    format='%d.%m.%Y %H:%M:%S.%f',
                    errors='coerce'
                )
                
                # If that fails, try without microseconds
                if self.df['DateTime'].isna().all():
                    self.df['DateTime'] = pd.to_datetime(
                        self.df['Date'].astype(str) + ' ' + self.df['Time'].astype(str), 
                        format='%d.%m.%Y %H:%M:%S',
                        errors='coerce'
                    )
                
                # Remove invalid datetime entries
                self.df = self.df.dropna(subset=['DateTime'])
                self.df = self.df.set_index('DateTime')
                
            except Exception as e:
                print(f"Warning: Could not parse datetime: {e}")
                self.df.index = pd.date_range(start='2024-01-01', periods=len(self.df), freq='1S')
        else:
            print("Warning: Date/Time columns not found. Using sequential index.")
            self.df.index = pd.date_range(start='2024-01-01', periods=len(self.df), freq='1S')
    
    def _identify_columns(self):
        """Identify different types of sensor columns."""
        all_columns = self.df.columns.tolist()
        
        # Temperature columns (handle different encodings for degree symbol)
        temp_patterns = ['[°C]', '[�C]', 'Temperature', 'Temp']
        self.temp_columns = [col for col in all_columns 
                           if any(pattern in col for pattern in temp_patterns)]
        
        # Categorize temperature columns
        self.cpu_temp_columns = [col for col in self.temp_columns 
                               if any(cpu_term in col.upper() for cpu_term in 
                                     ['CPU', 'CORE', 'CCD', 'IOD', 'TCTL', 'TDIE'])]
        
        self.gpu_temp_columns = [col for col in self.temp_columns 
                               if any(gpu_term in col.upper() for gpu_term in 
                                     ['GPU', 'GRAPHICS', 'VGA'])]
        
        self.motherboard_temp_columns = [col for col in self.temp_columns 
                                       if col not in self.cpu_temp_columns + self.gpu_temp_columns]
        
        # Voltage columns (exclude VID, RPM, MHz, etc.)
        exclude_patterns = ['VID', 'RPM', 'MHZ', 'RATIO', 'CLOCK', 'USAGE', 'LOAD']
        self.voltage_columns = [col for col in all_columns 
                              if '[V]' in col and not any(pattern in col.upper() for pattern in exclude_patterns)]
    
    def _clean_numeric_data(self):
        """Clean and convert numeric columns."""
        # Clean temperature columns
        for col in self.temp_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Clean voltage columns
        for col in self.voltage_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Remove columns that are completely non-numeric
        self.temp_columns = [col for col in self.temp_columns 
                           if col in self.df.columns and not self.df[col].isna().all()]
        self.voltage_columns = [col for col in self.voltage_columns 
                              if col in self.df.columns and not self.df[col].isna().all()]
        
        # Update categorized columns
        self.cpu_temp_columns = [col for col in self.cpu_temp_columns if col in self.temp_columns]
        self.gpu_temp_columns = [col for col in self.gpu_temp_columns if col in self.temp_columns]
        self.motherboard_temp_columns = [col for col in self.motherboard_temp_columns if col in self.temp_columns]
    
    def get_summary_statistics(self):
        """Get summary statistics for the dataset."""
        summary = {
            'total_samples': len(self.df),
            'time_span': None,
            'sampling_frequency': None,
            'cpu_temps': {},
            'gpu_temps': {},
            'voltages': {}
        }
        
        # Time analysis
        if isinstance(self.df.index, pd.DatetimeIndex):
            summary['time_span'] = (self.df.index[-1] - self.df.index[0]).total_seconds()
            if len(self.df) > 1:
                avg_interval = summary['time_span'] / (len(self.df) - 1)
                summary['sampling_frequency'] = f"{avg_interval:.1f} seconds"
        
        # CPU temperature summary
        for col in self.cpu_temp_columns:
            data = self.df[col].dropna()
            if len(data) > 0:
                summary['cpu_temps'][col] = {
                    'mean': data.mean(),
                    'max': data.max(),
                    'min': data.min(),
                    'std': data.std()
                }
        
        # GPU temperature summary
        for col in self.gpu_temp_columns:
            data = self.df[col].dropna()
            if len(data) > 0:
                summary['gpu_temps'][col] = {
                    'mean': data.mean(),
                    'max': data.max(),
                    'min': data.min(),
                    'std': data.std()
                }
        
        # Voltage summary
        for col in self.voltage_columns[:10]:  # Limit to first 10 for brevity
            data = self.df[col].dropna()
            if len(data) > 0:
                summary['voltages'][col] = {
                    'mean': data.mean(),
                    'max': data.max(),
                    'min': data.min(),
                    'std': data.std()
                }
        
        return summary
    
    def get_column_data(self, column_name):
        """Get cleaned data for a specific column."""
        if column_name not in self.df.columns:
            return pd.Series()
        
        data = pd.to_numeric(self.df[column_name], errors='coerce').dropna()
        return data[np.isfinite(data)]