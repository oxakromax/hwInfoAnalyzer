"""
Módulo de visualización para análisis de HWiNFO
Genera gráficos completos para diagnóstico térmico y de voltajes
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path
import warnings
warnings.filterwarnings('ignore')

class HWiNFOVisualizer:
    def __init__(self, output_dir="plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configurar estilo visual profesional
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        self.colors = {
            'cpu': '#FF6B6B',
            'gpu': '#4ECDC4', 
            'system': '#45B7D1',
            'voltage': '#96CEB4',
            'critical': '#FF4757',
            'warning': '#FFA726',
            'good': '#26A69A'
        }

    def create_complete_analysis_plots(self, data, processor, thermal_analyzer, anomalies):
        """Genera conjunto completo de gráficos para análisis"""
        plots_created = []
        
        # 1. Tendencias temporales principales
        plots_created.append(self._plot_temperature_trends(data, processor))
        
        # 2. Distribuciones de temperaturas
        plots_created.append(self._plot_temperature_distributions(data, processor))
        
        # 3. Heatmap de componentes
        plots_created.append(self._plot_component_heatmap(data, processor))
        
        # 4. Análisis de voltajes
        plots_created.append(self._plot_voltage_analysis(data, processor))
        
        # 5. Anomalías detectadas
        plots_created.append(self._plot_anomalies(data, anomalies, processor))
        
        # 6. Dashboard resumen
        plots_created.append(self._create_dashboard(data, processor, thermal_analyzer))
        
        # 7. Correlaciones entre componentes
        plots_created.append(self._plot_correlations(data, processor))
        
        return plots_created

    def _plot_temperature_trends(self, data, processor):
        """Gráfico de tendencias temporales de temperaturas"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis de Tendencias Térmicas HWiNFO', fontsize=16, fontweight='bold')
        
        # CPU Temperatures
        if processor.cpu_temp_columns:
            ax = axes[0, 0]
            for col in processor.cpu_temp_columns[:6]:  # Máximo 6 sensores
                if col in data.columns:
                    ax.plot(data.index, data[col], label=col.replace(' [°C]', ''), alpha=0.8)
            ax.set_title('Temperaturas CPU', fontweight='bold')
            ax.set_ylabel('Temperatura (°C)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Líneas de referencia térmica
            ax.axhline(y=85, color='orange', linestyle='--', alpha=0.7, label='Warning')
            ax.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='Critical')
        
        # GPU Temperatures
        if processor.gpu_temp_columns:
            ax = axes[0, 1]
            for col in processor.gpu_temp_columns[:4]:
                if col in data.columns:
                    ax.plot(data.index, data[col], label=col.replace(' [°C]', ''), alpha=0.8)
            ax.set_title('Temperaturas GPU', fontweight='bold')
            ax.set_ylabel('Temperatura (°C)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            ax.axhline(y=80, color='orange', linestyle='--', alpha=0.7)
            ax.axhline(y=85, color='red', linestyle='--', alpha=0.7)
        
        # System Temperatures
        if processor.system_temp_columns:
            ax = axes[1, 0]
            for col in processor.system_temp_columns[:4]:
                if col in data.columns:
                    ax.plot(data.index, data[col], label=col.replace(' [°C]', ''), alpha=0.8)
            ax.set_title('Temperaturas del Sistema', fontweight='bold')
            ax.set_ylabel('Temperatura (°C)')
            ax.set_xlabel('Tiempo')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Temperature Summary
        ax = axes[1, 1]
        temp_data = data[processor.temp_columns].dropna()
        if not temp_data.empty:
            ax.plot(temp_data.index, temp_data.mean(axis=1), 
                   color=self.colors['cpu'], linewidth=2, label='Promedio General')
            ax.fill_between(temp_data.index, 
                           temp_data.min(axis=1), 
                           temp_data.max(axis=1), 
                           alpha=0.3, color=self.colors['cpu'])
            ax.set_title('Resumen Térmico Global', fontweight='bold')
            ax.set_ylabel('Temperatura (°C)')
            ax.set_xlabel('Tiempo')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = self.output_dir / "temperature_trends.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename

    def _plot_temperature_distributions(self, data, processor):
        """Gráfico de distribuciones de temperaturas por componente"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Distribución de Temperaturas por Componente', fontsize=16, fontweight='bold')
        
        # CPU Distribution
        ax = axes[0, 0]
        cpu_data = []
        for col in processor.cpu_temp_columns:
            if col in data.columns:
                cpu_data.extend(data[col].dropna().values)
        
        if cpu_data:
            ax.hist(cpu_data, bins=50, alpha=0.7, color=self.colors['cpu'], edgecolor='black')
            ax.axvline(np.mean(cpu_data), color='red', linestyle='--', 
                      label=f'Media: {np.mean(cpu_data):.1f}°C')
            ax.axvline(85, color='orange', linestyle='--', alpha=0.7, label='Warning (85°C)')
            ax.axvline(95, color='red', linestyle='--', alpha=0.7, label='Critical (95°C)')
            ax.set_title('Distribución CPU')
            ax.set_xlabel('Temperatura (°C)')
            ax.set_ylabel('Frecuencia')
            ax.legend()
        
        # GPU Distribution
        ax = axes[0, 1]
        gpu_data = []
        for col in processor.gpu_temp_columns:
            if col in data.columns:
                gpu_data.extend(data[col].dropna().values)
        
        if gpu_data:
            ax.hist(gpu_data, bins=50, alpha=0.7, color=self.colors['gpu'], edgecolor='black')
            ax.axvline(np.mean(gpu_data), color='red', linestyle='--',
                      label=f'Media: {np.mean(gpu_data):.1f}°C')
            ax.axvline(80, color='orange', linestyle='--', alpha=0.7, label='Warning (80°C)')
            ax.axvline(85, color='red', linestyle='--', alpha=0.7, label='Critical (85°C)')
            ax.set_title('Distribución GPU')
            ax.set_xlabel('Temperatura (°C)')
            ax.set_ylabel('Frecuencia')
            ax.legend()
        
        # System Distribution
        ax = axes[1, 0]
        system_data = []
        for col in processor.system_temp_columns:
            if col in data.columns:
                system_data.extend(data[col].dropna().values)
        
        if system_data:
            ax.hist(system_data, bins=50, alpha=0.7, color=self.colors['system'], edgecolor='black')
            ax.axvline(np.mean(system_data), color='red', linestyle='--',
                      label=f'Media: {np.mean(system_data):.1f}°C')
            ax.set_title('Distribución Sistema')
            ax.set_xlabel('Temperatura (°C)')
            ax.set_ylabel('Frecuencia')
            ax.legend()
        
        # Box plot comparison
        ax = axes[1, 1]
        box_data = []
        labels = []
        
        if cpu_data:
            box_data.append(cpu_data)
            labels.append('CPU')
        if gpu_data:
            box_data.append(gpu_data)
            labels.append('GPU')
        if system_data:
            box_data.append(system_data)
            labels.append('Sistema')
        
        if box_data:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
            colors = [self.colors['cpu'], self.colors['gpu'], self.colors['system']]
            for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_title('Comparación por Componente')
            ax.set_ylabel('Temperatura (°C)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = self.output_dir / "temperature_distributions.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename

    def _plot_component_heatmap(self, data, processor):
        """Heatmap de temperaturas de componentes a lo largo del tiempo"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('Heatmap Térmico por Componentes', fontsize=16, fontweight='bold')
        
        # CPU Heatmap
        if processor.cpu_temp_columns:
            cpu_data = data[processor.cpu_temp_columns].dropna()
            if not cpu_data.empty:
                # Submuestrear para visualización
                step = max(1, len(cpu_data) // 100)
                cpu_sample = cpu_data.iloc[::step]
                
                im1 = axes[0].imshow(cpu_sample.T, aspect='auto', cmap='coolwarm', 
                                   vmin=30, vmax=100)
                axes[0].set_title('CPU Temperaturas')
                axes[0].set_ylabel('Sensores')
                axes[0].set_yticks(range(len(cpu_sample.columns)))
                axes[0].set_yticklabels([col.replace(' [°C]', '') for col in cpu_sample.columns])
                plt.colorbar(im1, ax=axes[0], label='Temperatura (°C)')
        
        # GPU Heatmap
        if processor.gpu_temp_columns:
            gpu_data = data[processor.gpu_temp_columns].dropna()
            if not gpu_data.empty:
                step = max(1, len(gpu_data) // 100)
                gpu_sample = gpu_data.iloc[::step]
                
                im2 = axes[1].imshow(gpu_sample.T, aspect='auto', cmap='coolwarm',
                                   vmin=30, vmax=90)
                axes[1].set_title('GPU Temperaturas')
                axes[1].set_ylabel('Sensores')
                axes[1].set_yticks(range(len(gpu_sample.columns)))
                axes[1].set_yticklabels([col.replace(' [°C]', '') for col in gpu_sample.columns])
                plt.colorbar(im2, ax=axes[1], label='Temperatura (°C)')
        
        # System Heatmap
        if processor.system_temp_columns:
            system_data = data[processor.system_temp_columns].dropna()
            if not system_data.empty:
                step = max(1, len(system_data) // 100)
                system_sample = system_data.iloc[::step]
                
                im3 = axes[2].imshow(system_sample.T, aspect='auto', cmap='coolwarm',
                                   vmin=20, vmax=80)
                axes[2].set_title('Sistema Temperaturas')
                axes[2].set_ylabel('Sensores')
                axes[2].set_xlabel('Tiempo (muestras)')
                axes[2].set_yticks(range(len(system_sample.columns)))
                axes[2].set_yticklabels([col.replace(' [°C]', '') for col in system_sample.columns])
                plt.colorbar(im3, ax=axes[2], label='Temperatura (°C)')
        
        plt.tight_layout()
        filename = self.output_dir / "thermal_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename

    def _plot_voltage_analysis(self, data, processor):
        """Análisis completo de voltajes"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis de Voltajes del Sistema', fontsize=16, fontweight='bold')
        
        voltage_cols = [col for col in data.columns if '[V]' in col]
        
        if voltage_cols:
            # Tendencias de voltaje
            ax = axes[0, 0]
            for col in voltage_cols[:6]:  # Mostrar máximo 6
                if col in data.columns:
                    ax.plot(data.index, data[col], label=col.replace(' [V]', ''), alpha=0.8)
            ax.set_title('Tendencias de Voltaje')
            ax.set_ylabel('Voltaje (V)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Distribución de voltajes
            ax = axes[0, 1]
            voltage_data = data[voltage_cols].dropna()
            if not voltage_data.empty:
                for col in voltage_cols[:4]:
                    ax.hist(data[col].dropna(), bins=30, alpha=0.6, 
                           label=col.replace(' [V]', ''))
                ax.set_title('Distribución de Voltajes')
                ax.set_xlabel('Voltaje (V)')
                ax.set_ylabel('Frecuencia')
                ax.legend()
            
            # Estabilidad de voltajes (coeficiente de variación)
            ax = axes[1, 0]
            cv_data = []
            labels = []
            for col in voltage_cols:
                if col in data.columns:
                    series = data[col].dropna()
                    if len(series) > 0 and series.mean() != 0:
                        cv = series.std() / series.mean() * 100
                        cv_data.append(cv)
                        labels.append(col.replace(' [V]', ''))
            
            if cv_data:
                bars = ax.bar(labels, cv_data, color=self.colors['voltage'])
                ax.set_title('Estabilidad de Voltajes (CV%)')
                ax.set_ylabel('Coeficiente de Variación (%)')
                ax.tick_params(axis='x', rotation=45)
                
                # Líneas de referencia
                ax.axhline(y=3, color='orange', linestyle='--', alpha=0.7, label='Warning (3%)')
                ax.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Critical (5%)')
                ax.legend()
            
            # Correlación voltajes vs temperaturas
            ax = axes[1, 1]
            if processor.temp_columns and voltage_cols:
                temp_mean = data[processor.temp_columns].mean(axis=1)
                voltage_mean = data[voltage_cols].mean(axis=1)
                
                valid_indices = temp_mean.notna() & voltage_mean.notna()
                if valid_indices.any():
                    ax.scatter(temp_mean[valid_indices], voltage_mean[valid_indices], 
                             alpha=0.6, color=self.colors['voltage'])
                    ax.set_title('Correlación Temperatura vs Voltaje')
                    ax.set_xlabel('Temperatura Media (°C)')
                    ax.set_ylabel('Voltaje Medio (V)')
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = self.output_dir / "voltage_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename

    def _plot_anomalies(self, data, anomalies, processor):
        """Visualización de anomalías detectadas"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Anomalías Detectadas en el Sistema', fontsize=16, fontweight='bold')
        
        # Anomalías en el tiempo
        ax = axes[0, 0]
        if processor.temp_columns:
            temp_mean = data[processor.temp_columns].mean(axis=1)
            ax.plot(temp_mean.index, temp_mean, color='blue', alpha=0.7, label='Temperatura Media')
            
            # Marcar anomalías
            for method, method_anomalies in anomalies.items():
                if method_anomalies:
                    anomaly_temps = temp_mean.iloc[method_anomalies]
                    ax.scatter(anomaly_temps.index, anomaly_temps.values, 
                             color='red', s=50, alpha=0.8, label=f'Anomalías {method}')
            
            ax.set_title('Anomalías Temporales')
            ax.set_ylabel('Temperatura (°C)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Distribución de anomalías por método
        ax = axes[0, 1]
        methods = list(anomalies.keys())
        counts = [len(anomalies[method]) for method in methods]
        
        if counts:
            bars = ax.bar(methods, counts, color=['red', 'orange', 'darkred'])
            ax.set_title('Anomalías por Método de Detección')
            ax.set_ylabel('Número de Anomalías')
            ax.tick_params(axis='x', rotation=45)
            
            # Agregar valores en las barras
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom')
        
        # Heatmap de intensidad de anomalías
        ax = axes[1, 0]
        if processor.temp_columns:
            anomaly_matrix = np.zeros((len(processor.temp_columns), len(data)))
            
            for i, col in enumerate(processor.temp_columns):
                if col in data.columns:
                    for method, method_anomalies in anomalies.items():
                        for anomaly_idx in method_anomalies:
                            if anomaly_idx < len(data):
                                anomaly_matrix[i, anomaly_idx] = 1
            
            if anomaly_matrix.any():
                step = max(1, anomaly_matrix.shape[1] // 100)
                sample_matrix = anomaly_matrix[:, ::step]
                
                im = ax.imshow(sample_matrix, aspect='auto', cmap='Reds', 
                              vmin=0, vmax=1)
                ax.set_title('Mapa de Anomalías por Sensor')
                ax.set_ylabel('Sensores de Temperatura')
                ax.set_xlabel('Tiempo (muestras)')
                plt.colorbar(im, ax=ax, label='Intensidad de Anomalía')
        
        # Estadísticas de anomalías
        ax = axes[1, 1]
        if anomalies:
            all_anomalies = set()
            for method_anomalies in anomalies.values():
                all_anomalies.update(method_anomalies)
            
            if all_anomalies and processor.temp_columns:
                anomaly_temps = data[processor.temp_columns].iloc[list(all_anomalies)]
                normal_temps = data[processor.temp_columns].drop(anomaly_temps.index)
                
                box_data = [normal_temps.values.flatten(), anomaly_temps.values.flatten()]
                labels = ['Normal', 'Anomalías']
                
                bp = ax.boxplot([d[~np.isnan(d)] for d in box_data], labels=labels, patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][1].set_facecolor('lightcoral')
                
                ax.set_title('Comparación: Normal vs Anomalías')
                ax.set_ylabel('Temperatura (°C)')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = self.output_dir / "anomalies_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename

    def _create_dashboard(self, data, processor, thermal_analyzer):
        """Dashboard resumen del sistema"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Dashboard HWiNFO - Resumen del Sistema', fontsize=20, fontweight='bold')
        
        # Métricas principales (primera fila)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1]) 
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])
        
        # CPU Stats
        if processor.cpu_temp_columns:
            cpu_data = data[processor.cpu_temp_columns].dropna()
            if not cpu_data.empty:
                cpu_mean = cpu_data.mean().mean()
                cpu_max = cpu_data.max().max()
                
                ax1.text(0.5, 0.7, f'{cpu_mean:.1f}°C', ha='center', va='center', 
                        fontsize=24, fontweight='bold', color=self.colors['cpu'])
                ax1.text(0.5, 0.4, f'Max: {cpu_max:.1f}°C', ha='center', va='center', fontsize=12)
                ax1.set_title('CPU Promedio', fontweight='bold')
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)
                ax1.axis('off')
        
        # GPU Stats
        if processor.gpu_temp_columns:
            gpu_data = data[processor.gpu_temp_columns].dropna()
            if not gpu_data.empty:
                gpu_mean = gpu_data.mean().mean()
                gpu_max = gpu_data.max().max()
                
                ax2.text(0.5, 0.7, f'{gpu_mean:.1f}°C', ha='center', va='center',
                        fontsize=24, fontweight='bold', color=self.colors['gpu'])
                ax2.text(0.5, 0.4, f'Max: {gpu_max:.1f}°C', ha='center', va='center', fontsize=12)
                ax2.set_title('GPU Promedio', fontweight='bold')
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.axis('off')
        
        # System Stats
        if processor.system_temp_columns:
            system_data = data[processor.system_temp_columns].dropna()
            if not system_data.empty:
                system_mean = system_data.mean().mean()
                system_max = system_data.max().max()
                
                ax3.text(0.5, 0.7, f'{system_mean:.1f}°C', ha='center', va='center',
                        fontsize=24, fontweight='bold', color=self.colors['system'])
                ax3.text(0.5, 0.4, f'Max: {system_max:.1f}°C', ha='center', va='center', fontsize=12)
                ax3.set_title('Sistema Promedio', fontweight='bold')
                ax3.set_xlim(0, 1)
                ax3.set_ylim(0, 1)
                ax3.axis('off')
        
        # Tiempo de sesión
        session_time = len(data) * 2  # Asumiendo 2 segundos por muestra
        hours = session_time // 3600
        minutes = (session_time % 3600) // 60
        
        ax4.text(0.5, 0.7, f'{hours}h {minutes}m', ha='center', va='center',
                fontsize=24, fontweight='bold', color='darkblue')
        ax4.text(0.5, 0.4, f'{len(data)} muestras', ha='center', va='center', fontsize=12)
        ax4.set_title('Tiempo de Sesión', fontweight='bold')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        # Gráfico de tendencias (segunda fila, span 2 columnas)
        ax5 = fig.add_subplot(gs[1, :2])
        if processor.temp_columns:
            for i, comp_type in enumerate(['cpu', 'gpu', 'system']):
                if comp_type == 'cpu' and processor.cpu_temp_columns:
                    temp_data = data[processor.cpu_temp_columns].mean(axis=1)
                    color = self.colors['cpu']
                elif comp_type == 'gpu' and processor.gpu_temp_columns:
                    temp_data = data[processor.gpu_temp_columns].mean(axis=1)
                    color = self.colors['gpu']
                elif comp_type == 'system' and processor.system_temp_columns:
                    temp_data = data[processor.system_temp_columns].mean(axis=1)
                    color = self.colors['system']
                else:
                    continue
                
                ax5.plot(temp_data.index, temp_data, label=comp_type.upper(), 
                        color=color, linewidth=2)
            
            ax5.set_title('Tendencias Térmicas por Componente', fontweight='bold')
            ax5.set_ylabel('Temperatura (°C)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Distribución general (segunda fila, derecha)
        ax6 = fig.add_subplot(gs[1, 2:])
        if processor.temp_columns:
            all_temps = data[processor.temp_columns].values.flatten()
            all_temps = all_temps[~np.isnan(all_temps)]
            
            if len(all_temps) > 0:
                ax6.hist(all_temps, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                ax6.axvline(np.mean(all_temps), color='red', linestyle='--', 
                           label=f'Media: {np.mean(all_temps):.1f}°C')
                ax6.axvline(85, color='orange', linestyle='--', alpha=0.7, label='Warning')
                ax6.axvline(95, color='red', linestyle='--', alpha=0.7, label='Critical')
                ax6.set_title('Distribución General de Temperaturas', fontweight='bold')
                ax6.set_xlabel('Temperatura (°C)')
                ax6.set_ylabel('Frecuencia')
                ax6.legend()
        
        # Tabla de estadísticas (tercera fila)
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Crear tabla de estadísticas
        stats_data = []
        if processor.cpu_temp_columns:
            cpu_stats = data[processor.cpu_temp_columns].describe().loc[['mean', 'max', 'std']]
            stats_data.append(['CPU', f'{cpu_stats.loc["mean"].mean():.1f}°C', 
                             f'{cpu_stats.loc["max"].max():.1f}°C', 
                             f'{cpu_stats.loc["std"].mean():.1f}°C'])
        
        if processor.gpu_temp_columns:
            gpu_stats = data[processor.gpu_temp_columns].describe().loc[['mean', 'max', 'std']]
            stats_data.append(['GPU', f'{gpu_stats.loc["mean"].mean():.1f}°C',
                             f'{gpu_stats.loc["max"].max():.1f}°C',
                             f'{gpu_stats.loc["std"].mean():.1f}°C'])
        
        if processor.system_temp_columns:
            sys_stats = data[processor.system_temp_columns].describe().loc[['mean', 'max', 'std']]
            stats_data.append(['Sistema', f'{sys_stats.loc["mean"].mean():.1f}°C',
                             f'{sys_stats.loc["max"].max():.1f}°C',
                             f'{sys_stats.loc["std"].mean():.1f}°C'])
        
        if stats_data:
            table = ax7.table(cellText=stats_data,
                            colLabels=['Componente', 'Temp. Media', 'Temp. Máxima', 'Desv. Estándar'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0.2, 0.3, 0.6, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            
            # Colorear la tabla
            for i in range(len(stats_data)):
                table[(i+1, 0)].set_facecolor(list(self.colors.values())[i])
                table[(i+1, 0)].set_alpha(0.3)
        
        filename = self.output_dir / "system_dashboard.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename

    def _plot_correlations(self, data, processor):
        """Matriz de correlaciones entre componentes"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Análisis de Correlaciones', fontsize=16, fontweight='bold')
        
        # Correlación entre sensores de temperatura
        ax = axes[0]
        temp_data = data[processor.temp_columns].dropna()
        if not temp_data.empty and len(temp_data.columns) > 1:
            # Tomar muestra para correlación
            sample_size = min(1000, len(temp_data))
            temp_sample = temp_data.sample(n=sample_size)
            
            corr_matrix = temp_sample.corr()
            
            # Crear etiquetas más cortas
            short_labels = [col.replace(' [°C]', '').replace('CPU ', '')[:10] 
                           for col in corr_matrix.columns]
            
            im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(len(short_labels)))
            ax.set_yticks(range(len(short_labels)))
            ax.set_xticklabels(short_labels, rotation=45, ha='right')
            ax.set_yticklabels(short_labels)
            ax.set_title('Correlación Entre Sensores')
            
            # Agregar valores de correlación
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix.columns)):
                    value = corr_matrix.iloc[i, j]
                    color = 'white' if abs(value) > 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                           color=color, fontsize=8)
            
            plt.colorbar(im, ax=ax, label='Correlación')
        
        # Correlación temperatura vs voltaje por componente
        ax = axes[1]
        voltage_cols = [col for col in data.columns if '[V]' in col]
        
        if processor.temp_columns and voltage_cols:
            # Crear pares de temperatura-voltaje para correlación
            correlations = []
            labels = []
            
            # CPU correlations
            if processor.cpu_temp_columns:
                cpu_temp_mean = data[processor.cpu_temp_columns].mean(axis=1)
                for v_col in voltage_cols[:3]:  # Top 3 voltages
                    if v_col in data.columns:
                        corr = cpu_temp_mean.corr(data[v_col])
                        if not np.isnan(corr):
                            correlations.append(corr)
                            labels.append(f'CPU-{v_col.replace(" [V]", "")[:8]}')
            
            # GPU correlations
            if processor.gpu_temp_columns:
                gpu_temp_mean = data[processor.gpu_temp_columns].mean(axis=1)
                for v_col in voltage_cols[:3]:
                    if v_col in data.columns:
                        corr = gpu_temp_mean.corr(data[v_col])
                        if not np.isnan(corr):
                            correlations.append(corr)
                            labels.append(f'GPU-{v_col.replace(" [V]", "")[:8]}')
            
            if correlations:
                colors = ['red' if abs(c) > 0.5 else 'orange' if abs(c) > 0.3 else 'green' 
                         for c in correlations]
                bars = ax.barh(labels, correlations, color=colors, alpha=0.7)
                ax.set_title('Correlación Temperatura-Voltaje')
                ax.set_xlabel('Coeficiente de Correlación')
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Fuerte (+)')
                ax.axvline(x=-0.5, color='red', linestyle='--', alpha=0.5, label='Fuerte (-)')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = self.output_dir / "correlations_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename

    def create_summary_report(self, plots_created):
        """Genera un reporte textual de los gráficos creados"""
        report_path = self.output_dir / "plots_summary.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE VISUALIZACIONES - HWiNFO ANALYZER\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Gráficos generados: {len(plots_created)}\n")
            f.write(f"Directorio de salida: {self.output_dir}\n\n")
            
            f.write("ARCHIVOS GENERADOS:\n")
            f.write("-" * 20 + "\n")
            for i, plot_file in enumerate(plots_created, 1):
                f.write(f"{i}. {plot_file.name}\n")
            
            f.write("\nDESCRIPCIÓN DE GRÁFICOS:\n")
            f.write("-" * 25 + "\n")
            f.write("• temperature_trends.png - Tendencias temporales de temperaturas\n")
            f.write("• temperature_distributions.png - Distribuciones estadísticas\n")
            f.write("• thermal_heatmap.png - Mapas de calor por componente\n")
            f.write("• voltage_analysis.png - Análisis completo de voltajes\n")
            f.write("• anomalies_analysis.png - Visualización de anomalías\n")
            f.write("• system_dashboard.png - Dashboard resumen del sistema\n")
            f.write("• correlations_analysis.png - Correlaciones entre componentes\n")
        
        return report_path