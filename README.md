# HWiNFO Analyzer - Versión Mejorada

Analizador científicamente preciso y modular para logs CSV de HWiNFO con criterios térmicos actualizados basados en especificaciones de fabricantes (Intel, AMD, NVIDIA) 2024/2025.

## 🚀 Mejoras Principales

### ✅ **Criterios Térmicos Científicos**
- **CPU Intel**: Límites basados en TjMax real (100-105°C)
- **CPU AMD**: Límites para Ryzen 7000 (95°C) y arquitecturas anteriores
- **GPU NVIDIA**: Umbrales de throttling reales (83-87°C)
- **GPU AMD**: Tolerancia térmica superior (95-100°C)

### ✅ **Arquitectura Modular**
- `thermal_thresholds.py` - Umbrales térmicos por fabricante
- `data_processor.py` - Procesamiento robusto de CSV
- `anomaly_detector.py` - Detección avanzada de anomalías
- `thermal_analyzer.py` - Análisis térmico especializado
- `improved_analyzer.py` - Analizador principal mejorado

### ✅ **Análisis Más Preciso**
- Separación clara entre CPU y GPU
- Detección automática de fabricante (Intel/AMD/NVIDIA)
- Análisis por zonas térmicas específicas
- Puntuación ponderada de salud del sistema

## 📊 Comparación: Antes vs Después

### **Análisis Anterior** ❌
```
Estado del Sistema: CRÍTICO (0/100)
Problemas: CPU a 70°C marcado como crítico
Resultado: Falsos positivos masivos
```

### **Análisis Mejorado** ✅
```
System Health: [POOR] POOR (48.8/100)
Component Status:
  CPU: good (70°C es normal bajo carga)
  GPU: good (temperaturas dentro de rango)
  VOLTAGE: poor (inestabilidad real detectada)
```

## 🌡️ Nuevos Criterios Térmicos

### **CPU (Intel/AMD)**
- **Excelente**: < 50°C (idle)
- **Bueno**: 50-80°C (carga normal)
- **Elevado**: 80-85°C (necesita monitoreo)
- **Advertencia**: 85-90/95°C (según fabricante)
- **Crítico**: > 95°C (Intel) / > 90°C (AMD)

### **GPU (NVIDIA/AMD)**
- **Excelente**: < 50°C (idle)
- **Bueno**: 50-75°C (NVIDIA) / 50-80°C (AMD)
- **Elevado**: 75-80°C (NVIDIA) / 80-85°C (AMD)
- **Advertencia**: 80-85°C (NVIDIA) / 85-90°C (AMD)
- **Crítico**: > 85°C (NVIDIA) / > 90°C (AMD)

### **Voltajes**
- **Normal**: ±3% variación de nominal
- **Advertencia**: ±5% variación
- **Crítico**: ±8% variación

## 🛠️ Uso

### **Análisis Rápido**
```bash
python improved_analyzer.py test.CSV
```

### **Análisis Completo con Reporte**
```bash
python improved_analyzer.py test.CSV --output resultado_detallado
```

### **Script Principal**
**`improved_analyzer.py`** - Analizador modular y científicamente preciso

## 📈 Ejemplo de Salida Mejorada

```
System Health: [POOR] POOR
Health Score: 48.8/100

Component Status:
  CPU: good
  GPU: good  
  SYSTEM_THERMAL: excellent
  VOLTAGE: poor

WARNINGS:
  - GPU temperatures are elevated

RECOMMENDATIONS:
  - GPU is running hot
  - Monitor GPU temperatures during gaming
  - Check GPU fan operation
```

## 🔍 Detalles del Análisis

### **Detección de Componentes**
- **CPU**: Sensores con "CPU", "CORE", "CCD", "IOD", "TCTL", "TDIE"
- **GPU**: Sensores con "GPU", "GRAPHICS", "VGA"
- **Sistema**: VRM, chipset, motherboard, ambient

### **Algoritmos de Anomalías**
- **Isolation Forest** (ML) - Para patrones complejos
- **Z-Score Estadístico** - Para distribuciones normales
- **IQR** - Robusto contra outliers

### **Puntuación de Salud**
- **CPU**: 40% del peso total
- **GPU**: 30% del peso total
- **Sistema**: 15% del peso total
- **Voltajes**: 15% del peso total

## 🚨 Interpretación de Resultados

### **Health Score**
- **90-100**: Excelente - Sistema funcionando óptimamente
- **75-89**: Bueno - Funcionamiento normal
- **60-74**: Regular - Necesita monitoreo
- **40-59**: Malo - Problemas que necesitan atención
- **0-39**: Crítico - Acción inmediata requerida

### **Component Status**
- **excellent**: Sin problemas detectados
- **good**: Funcionamiento normal
- **fair**: Algunas preocupaciones menores
- **poor**: Problemas significativos
- **critical**: Requiere atención inmediata

## 🔧 Solución de Problemas Comunes

### **CPU Temperatures**
```
good (70°C) → Normal bajo carga
warning (85°C) → Mejorar refrigeración
critical (95°C) → Acción inmediata
```

### **GPU Temperatures**
```
good (75°C) → Normal para gaming
warning (80°C) → Verificar fans
critical (85°C+) → Reducir carga/mejorar cooling
```

### **Voltage Issues**
```
poor → Revisar PSU y conexiones
critical → Posible fallo de fuente
```

## 🎯 Basado en Especificaciones Reales

Los umbrales se basan en documentación oficial de:
- **Intel**: TjMax 100-105°C para CPUs modernos
- **AMD**: 95°C para Ryzen 7000, 89°C para 7800X3D
- **NVIDIA**: Throttling típico a 83-87°C
- **AMD GPU**: Diseñadas para hasta 95-100°C

## 📚 Fuentes Técnicas

- Intel Temperature Information (2024)
- AMD Ryzen Thermal Specifications
- NVIDIA GPU Temperature Guidelines
- Hardware monitoring best practices

---

**Nota**: Este analizador mejorado proporciona evaluaciones mucho más precisas basadas en especificaciones reales de fabricantes, eliminando falsos positivos y proporcionando diagnósticos útiles.