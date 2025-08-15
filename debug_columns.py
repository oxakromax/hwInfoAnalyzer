import pandas as pd

# Debug column detection
df = pd.read_csv('Reporte.csv', encoding='latin1', low_memory=False)
all_columns = df.columns.tolist()

print("=== DEBUGGING COLUMN DETECTION ===")

# Test temperature patterns
temp_patterns = ['[°C]', '[�C]', 'TEMPERATURE', 'TEMP', 'TEMPERATURA']
temp_columns = [col for col in all_columns if any(pattern in col.upper() for pattern in temp_patterns)]
print(f"Initial temp columns found: {len(temp_columns)}")

# Test exclusion patterns
exclude_temp_patterns = ['DISTANCIA', 'DISTANCE', 'CRÍTICA', 'CRITICAL', 'YES/NO', 'DESACELERACIÓN', 'THROTTLE', 'TJMAX']
filtered_temp_columns = [col for col in temp_columns 
                       if not any(exclude_pattern in col.upper() for exclude_pattern in exclude_temp_patterns)]
print(f"After filtering: {len(filtered_temp_columns)}")

print("\nFiltered temperature columns:")
for col in filtered_temp_columns[:15]:
    print(f"  {col}")

# Test CPU patterns
cpu_patterns = ['CPU', 'CORE', 'CCD', 'IOD', 'TCTL', 'TDIE', 'P-CORE', 'E-CORE', 'NÚCLEO', 'NÚCLEOS']
cpu_temp_columns = [col for col in filtered_temp_columns 
                   if any(cpu_term in col.upper() for cpu_term in cpu_patterns)]
print(f"\nCPU temp columns found: {len(cpu_temp_columns)}")
for col in cpu_temp_columns[:10]:
    print(f"  {col}")

# Check what's being excluded
print("\nWhat's being excluded:")
excluded = [col for col in temp_columns if col not in filtered_temp_columns]
for col in excluded[:10]:
    print(f"  {col}")