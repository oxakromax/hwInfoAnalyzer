#!/usr/bin/env python3
"""
Análisis Rápido de HWiNFO
Script simplificado para análisis básico de logs de HWiNFO.
"""

import sys
import os
from hwinfo_analyzer import HWInfoAnalyzer

def quick_analysis(csv_file):
    """Ejecuta un análisis rápido y muestra resultados en consola."""
    
    if not os.path.exists(csv_file):
        print(f"Error: No se encontro el archivo {csv_file}")
        return
    
    print("Iniciando Analisis Rapido de HWiNFO...")
    print("=" * 50)
    
    try:
        # Crear analizador
        analyzer = HWInfoAnalyzer(csv_file)
        
        # Cargar datos
        analyzer.load_data()
        
        # Análisis básico
        anomalies = analyzer.detect_anomalies()
        diagnosis = analyzer.generate_diagnosis()
        
        # Mostrar resultados
        print("\nRESULTADOS DEL ANALISIS")
        print("-" * 30)
        
        if 'diagnosis' in analyzer.analysis_results:
            diag = analyzer.analysis_results['diagnosis']
            health = diag['overall_health']
            
            # Estado general
            health_status = {
                'Excellent': '[EXCELENTE]',
                'Good': '[BUENO]', 
                'Fair': '[REGULAR]',
                'Poor': '[MALO]',
                'Critical': '[CRITICO]'
            }
            
            print(f"Estado del Sistema: {health_status.get(health['status'], '[DESCONOCIDO]')} {health['status']}")
            print(f"Puntuacion de Salud: {health['score']}/100")
            
            # Problemas detectados
            if health['issues']:
                print(f"\nPROBLEMAS DETECTADOS:")
                for issue in health['issues']:
                    print(f"  - {issue}")
            
            # Temperaturas críticas
            temp_analysis = diag['temperature_analysis']
            if temp_analysis['critical_temps']:
                print(f"\nTEMPERATURAS CRITICAS:")
                for temp in temp_analysis['critical_temps']:
                    clean_temp = temp.replace('°', 'deg').replace('�', 'deg')
                    print(f"  - {clean_temp}")
            
            # Problemas de voltaje
            voltage_analysis = diag['voltage_analysis']
            if voltage_analysis['voltage_instability']:
                print(f"\nINESTABILIDAD DE VOLTAJE:")
                for instability in voltage_analysis['voltage_instability']:
                    print(f"  - {instability['rail']}: {instability['instability']} variacion")
            
            # Recomendaciones
            print(f"\nRECOMENDACIONES:")
            if temp_analysis['overheating_detected']:
                print("  - Revisar sistema de refrigeracion")
                print("  - Verificar pasta termica del CPU/GPU")
                print("  - Limpiar ventiladores y radiadores")
            
            if temp_analysis['thermal_throttling_risk']:
                print("  - Reducir overclock si esta aplicado")
                print("  - Mejorar flujo de aire del case")
            
            if voltage_analysis['voltage_instability']:
                print("  - Verificar fuente de alimentacion")
                print("  - Revisar conexiones de alimentacion")
                print("  - Considerar estabilizar voltajes")
            
            if health['score'] >= 75:
                print("  - Sistema funcionando correctamente")
                print("  - Continuar monitoreo regular")
        
        print(f"\nAnalisis completado. Datos procesados: {len(analyzer.df)} muestras")
        
    except Exception as e:
        print(f"Error durante el analisis: {e}")
        print("Intenta con: python hwinfo_analyzer.py tu_archivo.csv")

def main():
    if len(sys.argv) != 2:
        print("Uso: python quick_analysis.py archivo.csv")
        print("Ejemplo: python quick_analysis.py test.CSV")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    quick_analysis(csv_file)

if __name__ == "__main__":
    main()