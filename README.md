# HWiNFO Analyzer

A comprehensive Python tool for analyzing HWiNFO CSV logs with thermal monitoring, anomaly detection, and hardware diagnostics.

## Features

### Hardware Support
- **Intel CPUs**: Traditional and hybrid architectures (12th gen+)
  - P-core and E-core analysis with specific thermal thresholds
  - TjMax-based temperature limits (100-105°C)
- **AMD CPUs**: Ryzen series with 95°C thermal design
- **GPUs**: NVIDIA and AMD graphics cards
- **System Components**: Motherboard, VRM, storage devices

### Analysis Capabilities
- **Thermal Analysis**: Component-specific temperature monitoring
- **Anomaly Detection**: Multiple algorithms (Isolation Forest, Z-Score, IQR)
- **Voltage Monitoring**: Power delivery stability analysis
- **Pattern Recognition**: Thermal behavior analysis
- **Hardware Diagnostics**: Automated health assessment

### Visualization
- Temperature trends over time
- Statistical distributions
- Thermal heatmaps
- Voltage stability analysis
- Anomaly detection plots
- System dashboard
- Component correlations

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scipy >= 1.9.0
- scikit-learn >= 1.1.0

## Usage

### Basic Analysis
```bash
python hwinfo_analyzer.py data.csv
```

### Comprehensive Analysis with Visualizations
```bash
python hwinfo_analyzer.py data.csv --output analysis_results
```

### Using Analysis Presets
```bash
# Comprehensive analysis (all methods enabled)
python hwinfo_analyzer.py data.csv --preset comprehensive

# Quick analysis (essential methods only)
python hwinfo_analyzer.py data.csv --preset minimal

# Thermal-focused analysis
python hwinfo_analyzer.py data.csv --preset thermal_focus

# Voltage-focused analysis
python hwinfo_analyzer.py data.csv --preset voltage_focus
```

### Custom Configuration
```bash
python hwinfo_analyzer.py data.csv --config custom_config.json
```

### Disable Visualizations
```bash
python hwinfo_analyzer.py data.csv --no-plots
```

## Configuration

The tool supports configurable analysis methods through JSON configuration files:

### Analysis Methods
- **Anomaly Detection**: Isolation Forest, Z-Score, IQR
- **Thermal Analysis**: Component-specific thresholds
- **Voltage Analysis**: Power delivery monitoring
- **Visualization**: Customizable plot generation

### Thermal Thresholds

#### CPU Temperatures
- **Intel Traditional**: Critical 95°C, Warning 85°C
- **Intel Hybrid (12th gen+)**: 
  - P-cores: Critical 100°C, Warning 90°C
  - E-cores: Critical 95°C, Warning 85°C
- **AMD Ryzen**: Critical 90°C, Warning 85°C

#### GPU Temperatures
- **NVIDIA**: Critical 85°C, Warning 80°C
- **AMD**: Critical 90°C, Warning 85°C

## Project Structure

```
hwinfo-analyzer/
├── hwinfo_analyzer.py         # Main analysis script
├── data_processor.py          # CSV data processing
├── thermal_analyzer.py        # Thermal analysis engine
├── thermal_thresholds.py      # Hardware-specific thresholds
├── anomaly_detector.py        # Anomaly detection algorithms
├── visualizer.py              # Visualization generation
├── analysis_methods.py        # Configuration management
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
└── README.md                  # Documentation
```

## Output

### Analysis Results
- **System Health Score**: 0-100 rating
- **Component Status**: Per-component health assessment
- **Critical Issues**: Hardware problems requiring attention
- **Recommendations**: Actionable advice for hardware optimization

### Visualization Files
- `temperature_trends.png` - Temporal temperature analysis
- `temperature_distributions.png` - Statistical temperature distributions
- `thermal_heatmap.png` - Component thermal mapping
- `voltage_analysis.png` - Power delivery analysis
- `anomalies_analysis.png` - Detected anomalies visualization
- `system_dashboard.png` - System overview dashboard
- `correlations_analysis.png` - Component correlation analysis

### Report Files
- `detailed_analysis.txt` - Comprehensive analysis report
- `plots_summary.txt` - Visualization file summary

## Health Score Interpretation

| Score Range | Status | Description |
|-------------|--------|-------------|
| 90-100 | Excellent | Optimal system performance |
| 75-89 | Good | Normal operation |
| 60-74 | Fair | Monitoring recommended |
| 40-59 | Poor | Issues requiring attention |
| 0-39 | Critical | Immediate action required |

## Component Status Levels

- **Excellent**: No issues detected
- **Good**: Normal operation
- **Fair**: Minor concerns
- **Poor**: Significant issues
- **Critical**: Immediate attention required

## Supported Hardware

### CPU Architectures
- Intel Core series (all generations)
- Intel hybrid architectures (12th gen Alder Lake+)
- AMD Ryzen series (all generations)
- AMD EPYC series

### GPU Series
- NVIDIA GeForce (all series)
- NVIDIA RTX series
- AMD Radeon (all series)
- AMD RX series

### System Components
- Motherboard sensors
- VRM temperature monitoring
- Storage device temperatures
- Power supply monitoring

## Advanced Features

### Multi-Algorithm Anomaly Detection
- **Isolation Forest**: Machine learning-based detection
- **Z-Score**: Statistical outlier detection
- **IQR**: Interquartile range analysis

### Hybrid CPU Support
- Automatic P-core/E-core detection
- Core-specific thermal thresholds
- Differentiated analysis for performance and efficiency cores

### Intelligent Threshold Detection
- Automatic manufacturer detection
- Architecture-specific temperature limits
- Dynamic threshold adjustment

## Contributing

Contributions are welcome! Please ensure code follows the existing style and includes appropriate tests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on manufacturer thermal specifications from Intel, AMD, and NVIDIA
- Thermal thresholds derived from official hardware documentation
- Anomaly detection algorithms based on established statistical methods