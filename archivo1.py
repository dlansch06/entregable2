import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score


# 1.  Datos Sintéticos (NumPy) y Carga a DataFrame (Pandas)
# ==============================================================================
print("--- 1.  Dataset Sintético (Regresión) ---")

np.random.seed(42) 
X_sintetico = np.random.rand(100, 2) * 10 


ruido = np.random.normal(0, 1, 100)
y_sintetico = 3 * X_sintetico[:, 0] + 2 * X_sintetico[:, 1] + 5 + ruido

# Carga de  datos a un DataFrame (Pandas)
df_sintetico = pd.DataFrame(X_sintetico, columns=['Feature_1', 'Feature_2'])
df_sintetico['Target'] = y_sintetico

print(f"Dataset sintético creado con {len(df_sintetico)} muestras.")
print(f"Primeras 5 filas del DataFrame:\n{df_sintetico.head()}")
print("-" * 40)


# 2. Álgebra Lineal y Análisis Estadístico Básico
# ==============================================================================
print("--- 2. Fundamentos (Álgebra Lineal y Estadística) ---")

# Álgebra Lineal(NumPy)
vector_X1 = df_sintetico['Feature_1'].values
vector_X2 = df_sintetico['Feature_2'].values
suma_vectores = vector_X1[:5] + vector_X2[:5] 
print(f"Suma de los 5 primeros elementos de Feature_1 y Feature_2: {suma_vectores}")


media_target = df_sintetico['Target'].mean()
std_target = df_sintetico['Target'].std()
print(f"Media de la variable objetivo (Target): {media_target:.4f}")
print(f"Desviación Estándar de la variable objetivo: {std_target:.4f}")
print("-" * 40)



# 3. Implementación del Modelo de Machine Learning (Regresión)
# ==============================================================================
X = df_sintetico[['Feature_1', 'Feature_2']]
y = df_sintetico['Target']

# División de Datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#(Estandarización)
print("--- 3. Preprocesamiento y Entrenamiento (Regresión) ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Normalización Numérica (StandardScaler) aplicada.")

# (Regresión Lineal)
modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train_scaled, y_train)
print("Modelo de Regresión Lineal entrenado.")
print("-" * 40)



# 4. Predicciones y Métricas
# ==============================================================================
print("--- 4. Predicciones y Métricas ---")

y_pred_test = modelo_regresion.predict(X_test_scaled)
print(f"Predicciones generadas para {len(y_pred_test)} muestras de prueba.")

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

print(f"\nMétricas del Modelo:")
print(f"  Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"  Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
print(f"  Coeficiente de Determinación (R2 Score): {r2:.4f}")


muestra_individual = X_test_scaled[0].reshape(1, -1) 
prediccion_individual = modelo_regresion.predict(muestra_individual)[0]

print(f"\nPredicción Individual:")
print(f"  Características (escaladas): {muestra_individual}")
print(f"  Valor Real (no escalado): {y_test.iloc[0]:.4f}")
print(f"  Predicción del Modelo: {prediccion_individual:.4f}")
print("-" * 40)


# 5. Cálculo de Varianza y Desviación Estándar (Manual)
# ==============================================================================
print("--- 5. Cálculo Manual de Varianza y Desviación Estándar ---")
datos_analisis = df_sintetico['Target'].values

def calcular_varianza_manual(data):
    """Calcula la varianza muestral (dividido por n-1)."""
    n = len(data)
    if n < 2: return 0
    media = sum(data) / n
    suma_diferencias_cuadrado = sum((x - media) ** 2 for x in data)
    return suma_diferencias_cuadrado / (n - 1)

def calcular_desviacion_estandar_manual(data):
    """Calcula la desviación estándar (raíz de la varianza)."""
    return calcular_varianza_manual(data) ** 0.5

var_manual = calcular_varianza_manual(datos_analisis)
std_manual = calcular_desviacion_estandar_manual(datos_analisis)

print(f"Varianza de 'Target' (Manual): {var_manual:.4f}")
print(f"Desviación Estándar de 'Target' (Manual): {std_manual:.4f}")
print(f"Verificación NumPy: {np.std(datos_analisis, ddof=1):.4f}")