# HWiNFO Log Analyzer

Un script de Python para detectar anomalías, patrones y picos de temperatura/voltaje en logs CSV de HWiNFO, ayudando a diagnosticar problemas de hardware.

## Características

- **Detección de Anomalías**: Utiliza múltiples algoritmos (Isolation Forest, métodos estadísticos, IQR)
- **Análisis de Patrones**: Detecta tendencias, patrones cíclicos y puntos de cambio
- **Detección de Picos**: Identifica picos de temperatura y voltaje críticos
- **Diagnóstico Automático**: Genera un diagnóstico comprensivo del estado del sistema
- **Visualizaciones**: Crea gráficos de tendencias y anomalías
- **Informes Detallados**: Genera reportes en texto plano con resultados

## Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Análisis Rápido (Recomendado para principiantes)
```bash
# Análisis rápido con resultados en consola
python quick_analysis.py test.CSV
```

### Análisis Completo
```bash
# Análisis básico
python hwinfo_analyzer.py test.CSV

# Especificar directorio de salida
python hwinfo_analyzer.py test.CSV --output mi_analisis

# Usar método de detección diferente
python hwinfo_analyzer.py test.CSV --method statistical

# Ver ayuda
python hwinfo_analyzer.py --help
```

### Uso Programático
```python
from hwinfo_analyzer import HWInfoAnalyzer

# Crear analizador
analyzer = HWInfoAnalyzer('test.CSV')

# Ejecutar análisis completo
analyzer.run_full_analysis('resultados')

# O ejecutar análisis individuales
analyzer.load_data()
anomalies = analyzer.detect_anomalies()
patterns = analyzer.detect_patterns()
peaks = analyzer.detect_peaks()
diagnosis = analyzer.generate_diagnosis()
```

## Salidas del Análisis

El script genera los siguientes archivos en el directorio de salida:

- `analysis_report.txt`: Reporte detallado con diagnóstico del sistema
- `plots/temperature_trends.png`: Gráficos de tendencias de temperatura
- `plots/voltage_trends.png`: Gráficos de tendencias de voltaje  
- `plots/anomaly_summary.png`: Resumen de anomalías por componente

## Interpretación de Resultados

### Puntuación de Salud del Sistema
- **90-100**: Excelente - Sistema funcionando normalmente
- **75-89**: Bueno - Funcionamiento normal con anomalías menores
- **50-74**: Regular - Algunos problemas detectados
- **25-49**: Malo - Problemas significativos
- **0-24**: Crítico - Problemas graves que requieren atención inmediata

### Detección de Problemas

**Temperaturas:**
- CPU > 85°C: Sobrecalentamiento crítico
- GPU > 83°C: Sobrecalentamiento de GPU
- Cualquier componente > 90°C: Temperatura peligrosa

**Voltajes:**
- Variación > 5%: Inestabilidad de voltaje
- Picos frecuentes: Posibles problemas de fuente de alimentación

## Métodos de Detección

### Isolation Forest (Predeterminado)
- Método de machine learning para detección de anomalías
- Efectivo para datos multidimensionales
- Contamination rate: 5%

### Estadístico
- Basado en Z-score
- Considera anomalías valores con |Z-score| > 3
- Bueno para distribuciones normales

### IQR (Interquartile Range)
- Método robusto basado en cuartiles
- Detecta valores fuera de Q1-1.5*IQR y Q3+1.5*IQR
- Menos sensible a valores extremos

## Estructura del CSV de HWiNFO

El script espera un CSV con:
- Columnas 'Date' y 'Time' para timestamps
- Columnas de temperatura con '[°C]' en el nombre
- Columnas de voltaje con '[V]' en el nombre
- Formato de fecha: DD.MM.YYYY HH:MM:SS.fff

## Diagnósticos Comunes

### Sobrecalentamiento
- **Síntomas**: Temperaturas > 85°C, picos frecuentes
- **Causas**: Refrigeración inadecuada, pasta térmica vieja, ventiladores defectuosos
- **Soluciones**: Limpiar sistema, cambiar pasta térmica, verificar ventiladores

### Inestabilidad de Voltaje
- **Síntomas**: Variaciones > 5%, picos de voltaje
- **Causas**: Fuente de alimentación defectuosa, problemas de VRM
- **Soluciones**: Verificar fuente de alimentación, revisar conexiones

### Thermal Throttling
- **Síntomas**: Temperaturas cercanas a límites, rendimiento reducido
- **Causas**: Refrigeración insuficiente para la carga de trabajo
- **Soluciones**: Mejorar refrigeración, reducir overclock

## Limitaciones

- Requiere logs CSV de HWiNFO con formato específico
- La precisión depende de la duración y frecuencia del muestreo
- Los umbrales están optimizados para hardware de consumo típico
- No considera contexto de carga de trabajo (gaming vs idle)

## Scripts Disponibles

### `hwinfo_analyzer.py` 
Script principal que genera análisis completo con gráficos y reportes detallados.

### `quick_analysis.py`
Script de análisis rápido que muestra resultados directamente en consola, ideal para diagnósticos rápidos sin generar archivos.

## Soporte

Para problemas o mejoras, consulta la documentación de HWiNFO o ajusta los umbrales en el código según tu hardware específico.