import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Configuración de visualización
sns.set_theme(style="whitegrid")

# ==============================================================================
# 0. Carga, Preparación y Fundamentos de Álgebra Lineal
# ==============================================================================
print("--- 0. Carga y Álgebra Lineal ---")
data = load_breast_cancer(as_frame=True)
df = data.frame
x_df = data.data
y = data.target
x = x_df.values # Array NumPy para Álgebra Lineal y scikit-learn

# Fundamento 1: Cálculo de Producto Punto y Norma de un Vector
producto_punto = np.dot(x[0], x[1])
norma_a = np.linalg.norm(x[0])
print(f"Producto Punto (Muestra 1 · Muestra 2): {producto_punto:.2f}")
print(f"Norma L2 del Vector Muestra 1: {norma_a:.2f}")

# Fundamento 2: Resolución de un Sistema de Ecuaciones Lineales
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
solucion = np.linalg.solve(A, b)
print(f"Solución de Sistema de Ecuaciones: {solucion}")
print("-" * 70)


# ==============================================================================
# 1. Análisis Estadístico y Preprocesamiento
# ==============================================================================
print("--- 1. Estadística y Preprocesamiento ---")
feature_name = 'mean area'
data_feature = x_df[feature_name]

# Fundamento 3: Cálculo de Media y Mediana
media = data_feature.mean()
mediana = data_feature.median()
desviacion_estandar = data_feature.std()
print(f"'{feature_name}': Media={media:.2f}, Mediana={mediana:.2f}")

# Fundamento 4: Identificación de Valores Atípicos (3-sigma rule)
umbral_superior = media + 3 * desviacion_estandar
atipicos = data_feature[data_feature > umbral_superior]
print(f"Valores Atípicos (> 3*STD): {len(atipicos)} encontrados.")

# Normalización Numérica y División de Datos
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
print("Datos normalizados y divididos para el entrenamiento.")
print("-" * 70)


# ==============================================================================
# 2. Modelo de Clasificación (Regresión Logística)
# ==============================================================================
print("--- 2. Modelo de Clasificación y Métricas ---")

model = LogisticRegression(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Métricas del Modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión (Accuracy Score): {accuracy:.4f}")
print('\nInforme de Clasificación:\n', classification_report(y_test, y_pred))

# Predicción Individual (CORRECCIÓN APLICADA AQUÍ)
muestra_index = 0
muestra = x_test[muestra_index].reshape(1,-1)

# Usamos .iloc para acceder por POSICIÓN a la etiqueta real en la Serie 'y_test'
etiqueta_real = y_test.iloc[muestra_index] 
muestra_predict = model.predict(muestra)[0]

print('\n--- PREDICCIÓN INDIVIDUAL ---')
print(f"Etiqueta REAL: {etiqueta_real}")
print(f"Predicción del Modelo: {muestra_predict}")
print("-" * 70)


# ==============================================================================
# 3. Visualización (Matplotlib y Seaborn)
# ==============================================================================
print("--- 3. Visualización de Datos ---")

plt.figure(figsize=(12, 5))

# Histograma
plt.subplot(1, 2, 1)
sns.histplot(data_feature, kde=True, color='teal')
plt.axvline(media, color='red', linestyle='dashed', linewidth=2, label='Media')
plt.title(f'Histograma de "{feature_name}"')
plt.legend()

# Gráfico de Dispersión
plt.subplot(1, 2, 2)
sns.scatterplot(
    x='mean perimeter',
    y='mean area',
    hue='target',
    data=df,
    palette='viridis',
    alpha=0.8
)
plt.title('Gráfico de Dispersión: Perímetro vs Área')
plt.tight_layout()
# plt.show()
print("Visualizaciones generadas para el análisis de distribución y relación.")