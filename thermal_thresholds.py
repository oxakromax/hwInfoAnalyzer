"""
Thermal Thresholds Module
Defines scientifically accurate temperature and voltage thresholds for hardware components.
Based on manufacturer specifications from Intel, AMD, NVIDIA for 2024/2025.
"""

class ThermalThresholds:
    """Hardware-specific thermal thresholds based on manufacturer specifications."""
    
    # CPU Temperature Thresholds (Celsius)
    CPU_THRESHOLDS = {
        'intel': {
            'idle_max': 50,         # Intel CPUs should idle under 50°C
            'normal_load_max': 80,  # Under 80°C is safe for sustained loads
            'warning': 85,          # Warning at 85°C
            'critical': 95,         # Critical at 95°C (most Intel CPUs max at 100-105°C)
            'tj_max': 100          # Typical TjMax for most Intel CPUs
        },
        'amd': {
            'idle_max': 50,         # AMD Ryzen idle temps
            'normal_load_max': 80,  # Safe sustained load temp
            'warning': 85,          # Warning threshold
            'critical': 90,         # Critical (Ryzen 7000 designed for 95°C)
            'tj_max': 95           # Most AMD CPUs max at 95°C
        },
        'generic': {
            'idle_max': 50,
            'normal_load_max': 80,
            'warning': 85,
            'critical': 90,
            'tj_max': 95
        }
    }
    
    # GPU Temperature Thresholds (Celsius)
    GPU_THRESHOLDS = {
        'nvidia': {
            'idle_max': 50,         # Idle temp threshold
            'normal_load_max': 75,  # Normal gaming load
            'warning': 80,          # Start monitoring closely
            'critical': 85,         # NVIDIA cards throttle around 83-87°C
            'shutdown': 95          # Emergency shutdown temp
        },
        'amd': {
            'idle_max': 50,
            'normal_load_max': 80,  # AMD runs slightly warmer by design
            'warning': 85,
            'critical': 90,         # AMD cards can handle up to 95°C
            'shutdown': 100
        },
        'generic': {
            'idle_max': 50,
            'normal_load_max': 75,
            'warning': 80,
            'critical': 85,
            'shutdown': 95
        }
    }
    
    # Voltage Thresholds (percentage variation from nominal)
    VOLTAGE_THRESHOLDS = {
        'cpu_core': {
            'normal_variation': 0.03,    # ±3% is normal
            'warning_variation': 0.05,   # ±5% needs attention
            'critical_variation': 0.08   # ±8% is problematic
        },
        'gpu_core': {
            'normal_variation': 0.02,    # GPUs need more stable voltage
            'warning_variation': 0.04,
            'critical_variation': 0.06
        },
        'system_rails': {
            'normal_variation': 0.03,    # 12V, 5V, 3.3V rails
            'warning_variation': 0.05,
            'critical_variation': 0.08
        }
    }
    
    @staticmethod
    def get_cpu_thresholds(cpu_type='generic'):
        """Get CPU temperature thresholds for specific CPU type."""
        return ThermalThresholds.CPU_THRESHOLDS.get(cpu_type.lower(), 
                                                   ThermalThresholds.CPU_THRESHOLDS['generic'])
    
    @staticmethod
    def get_gpu_thresholds(gpu_type='generic'):
        """Get GPU temperature thresholds for specific GPU type."""
        return ThermalThresholds.GPU_THRESHOLDS.get(gpu_type.lower(), 
                                                   ThermalThresholds.GPU_THRESHOLDS['generic'])
    
    @staticmethod
    def classify_cpu_temperature(temp, cpu_type='generic'):
        """Classify CPU temperature into categories."""
        thresholds = ThermalThresholds.get_cpu_thresholds(cpu_type)
        
        if temp < thresholds['idle_max']:
            return 'excellent'
        elif temp < thresholds['normal_load_max']:
            return 'good'
        elif temp < thresholds['warning']:
            return 'elevated'
        elif temp < thresholds['critical']:
            return 'warning'
        else:
            return 'critical'
    
    @staticmethod
    def classify_gpu_temperature(temp, gpu_type='generic'):
        """Classify GPU temperature into categories."""
        thresholds = ThermalThresholds.get_gpu_thresholds(gpu_type)
        
        if temp < thresholds['idle_max']:
            return 'excellent'
        elif temp < thresholds['normal_load_max']:
            return 'good'
        elif temp < thresholds['warning']:
            return 'elevated'
        elif temp < thresholds['critical']:
            return 'warning'
        else:
            return 'critical'
    
    @staticmethod
    def get_thermal_advice(component_type, temperature, component_name=''):
        """Get specific advice based on component and temperature."""
        if 'cpu' in component_type.lower():
            classification = ThermalThresholds.classify_cpu_temperature(temperature)
        elif 'gpu' in component_type.lower():
            classification = ThermalThresholds.classify_gpu_temperature(temperature)
        else:
            # Generic temperature classification
            if temperature < 60:
                classification = 'excellent'
            elif temperature < 75:
                classification = 'good'
            elif temperature < 85:
                classification = 'elevated'
            elif temperature < 95:
                classification = 'warning'
            else:
                classification = 'critical'
        
        advice = {
            'excellent': [],
            'good': [],
            'elevated': [
                f"Monitor {component_name} temperatures more closely",
                "Consider improving case airflow"
            ],
            'warning': [
                f"Check {component_name} cooling solution",
                "Clean dust from heatsinks and fans",
                "Consider reducing overclock if applied"
            ],
            'critical': [
                f"Immediately address {component_name} cooling",
                "Check thermal paste application",
                "Verify all fans are working",
                "Reduce workload until temperatures normalize"
            ]
        }
        
        return classification, advice.get(classification, [])