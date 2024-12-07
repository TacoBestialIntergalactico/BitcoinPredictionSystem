# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:05:32 2024
Updated on Mon Nov 11 04:42:00 2024

@author: fearo
"""

import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Acceso al Dataset
os.chdir("C:/Programacion/Python")
df = pd.read_csv("BTC_USD.csv")

# Convertir la columna 'Date' a tipo datetime
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Información básica del dataset
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.describe())
print(df.info())

# Detección de valores nulos
print(df.isnull().sum())

# Visualización de las variables más importantes
sns.pairplot(df, vars=['Open', 'High', 'Low', 'Close'])
plt.show()

# Normalización de los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close']])
                                       
# Matriz de correlación (excluyendo la columna 'Date')
correlation_matrix = df.drop(['Date'], axis=1).corr()

# Dataframe con los precios de cierre
tabla_precios_cierre = df[['Date', 'Close']]

# Generar el gráfico de precios de cierre
plt.figure(figsize=(14, 7))
plt.plot(tabla_precios_cierre['Date'], tabla_precios_cierre['Close'], color='blue', label='Precio de Cierre')
plt.title('Precios de Cierre del Bitcoin')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre (USD)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Crear secuencias de entrenamiento
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 3])  # Precio de cierre
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Dividir en conjuntos de entrenamiento y prueba
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------------------------------- Sistema de Prediccion de Precios (LSTM) --------------------------------------------------

# Construcción del modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 4)))
model.add(LSTM(50))
model.add(Dense(1))

# Compilación del modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenamiento del modelo
model.fit(X_train, y_train, epochs=20, batch_size=32)
          
# Predicciones
predictions_lstm = model.predict(X_test)
predictions_lstm = scaler.inverse_transform(np.concatenate((np.zeros((predictions_lstm.shape[0], 3)), predictions_lstm), axis=1))[:, 3]

# Desnormalización de los valores reales
y_test_descaled = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 3)), y_test.reshape(-1, 1)), axis=1))[:, 3]

# Crear DataFrame para comparación
comparison_df = pd.DataFrame({
    'Fecha': df['Date'].values[-len(y_test):],
    'Precio Real': y_test_descaled,
    'Predicción LSTM': predictions_lstm
})
print(comparison_df)

# Evaluación del modelo LSTM
mse_lstm = mean_squared_error(y_test_descaled, predictions_lstm)
r2_lstm = r2_score(y_test_descaled, predictions_lstm)
print(f'LSTM - MSE: {mse_lstm}')
print(f'LSTM - R²: {r2_lstm}')

# Crear DataFrame para comparación
comparison_df = pd.DataFrame({
    'Fecha': df['Date'].values[-len(y_test):],
    'Precio Real': y_test_descaled,
    'Predicción LSTM': predictions_lstm
})
print(comparison_df)

# Visualización de las predicciones LSTM con fechas
plt.figure(figsize=(14, 7))
plt.plot(comparison_df['Fecha'], comparison_df['Precio Real'], color='blue', label='Precio Real')
plt.plot(comparison_df['Fecha'], comparison_df['Predicción LSTM'], color='red', label='Predicción LSTM')
plt.title('Predicción del Precio de Bitcoin (LSTM)')
plt.xlabel('Fecha')
plt.ylabel('Precio de Bitcoin')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# -------------------------------------------------- Sistema de Prediccion de Precios (Random Forest) --------------------------------------------------

from sklearn.ensemble import RandomForestRegressor

# Redefinir X y y para el modelo Random Forest (sin secuencias)
X_rf = scaled_data[:-1]
y_rf = df['Close'][1:].values  # Precio de cierre, con un desplazamiento de 1 día

# Dividir en conjuntos de entrenamiento y prueba
split_rf = int(0.8 * len(X_rf))
X_train_rf, X_test_rf = X_rf[:split_rf], X_rf[split_rf:]
y_train_rf, y_test_rf = y_rf[:split_rf], y_rf[split_rf:]

# Construcción del modelo Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train_rf, y_train_rf)

# Predicciones
predictions_rf = model_rf.predict(X_test_rf)

# Ajustar la longitud de y_test_rf y predictions_rf para que coincidan
min_length = min(len(y_test_descaled), len(predictions_rf))
y_test_rf = y_test_descaled[-min_length:]
predictions_rf = predictions_rf[-min_length:]
dates_test_rf = df['Date'].values[-min_length:]

# Verificación de longitudes
print(f'Longitud de y_test_rf: {len(y_test_rf)}')
print(f'Longitud de predictions_rf: {len(predictions_rf)}')

# Evaluación del modelo Random Forest
mse_rf = mean_squared_error(y_test_rf, predictions_rf)
r2_rf = r2_score(y_test_rf, predictions_rf)
print(f'Random Forest - MSE: {mse_rf}')
print(f'Random Forest - R²: {r2_rf}')

# Crear DataFrame para comparación
comparison_df_rf = pd.DataFrame({
    'Fecha': dates_test_rf,
    'Precio Real': y_test_rf,
    'Predicción Random Forest': predictions_rf
})
print(comparison_df_rf)

# Visualización de las predicciones Random Forest con fechas
plt.figure(figsize=(14, 7))
plt.plot(dates_test_rf, y_test_rf, color='blue', label='Precio Real')
plt.plot(dates_test_rf, predictions_rf, color='green', label='Predicción Random Forest')
plt.title('Predicción del Precio de Bitcoin (Random Forest)')
plt.xlabel('Fecha')
plt.ylabel('Precio de Bitcoin')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Crear DataFrame combinado para comparación
comparison_df_combined = pd.DataFrame({
    'Fecha': dates_test_rf,
    'Precio Real': y_test_rf,
    'Predicción LSTM': predictions_lstm[-min_length:],
    'Predicción Random Forest': predictions_rf
})

# Visualización de la comparación de modelos
plt.figure(figsize=(14, 7))
plt.plot(comparison_df_combined['Fecha'], comparison_df_combined['Precio Real'], color='blue', label='Precio Real')
plt.plot(comparison_df_combined['Fecha'], comparison_df_combined['Predicción LSTM'], color='red', label='Predicción LSTM')
plt.plot(comparison_df_combined['Fecha'], comparison_df_combined['Predicción Random Forest'], color='green', label='Predicción Random Forest')
plt.title('Comparación de Modelos de Predicción del Precio de Bitcoin')
plt.xlabel('Fecha')
plt.ylabel('Precio de Bitcoin')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# -------------------------------------------------- Sistema de Prediccion de 1 Dia (LSTM) --------------------------------------------------

# Seleccionar los últimos 60 días de datos para la predicción
last_60_days = scaled_data[-60:]
X_input = np.array([last_60_days])

# Predicción para el siguiente día
predicted_price_1_day_scaled = model.predict(X_input)
predicted_price_1_day = scaler.inverse_transform(
    np.concatenate((last_60_days[-1, :-1].reshape(1, -1), predicted_price_1_day_scaled), axis=1))[:, -1]
predicted_date_1_day = df['Date'].values[-1] + np.timedelta64(1, 'D')
print(f'Predicción del precio de Bitcoin para el siguiente día ({predicted_date_1_day}): {predicted_price_1_day[0]:.2f}')

# Crear DataFrame para los últimos 7 días y la predicción del siguiente día
last_7_days_prices = df['Close'].values[-7:]
last_7_days_dates = df['Date'].values[-7:]
predicted_data = {
    'Fecha': np.append(last_7_days_dates, predicted_date_1_day),
    'Precio Real': np.append(last_7_days_prices, [np.nan]),
    'Predicción LSTM': np.append([np.nan]*7, [predicted_price_1_day[0]])
}
prediction_df = pd.DataFrame(predicted_data)
print(prediction_df)

# Gráfica con los precios reales de los últimos 7 días y el día predicho
plt.figure(figsize=(14, 7))
plt.plot(prediction_df['Fecha'][:-1], prediction_df['Precio Real'][:-1], color='blue', label='Precio Real (últimos 7 días)')
plt.plot(prediction_df['Fecha'].iloc[-1], prediction_df['Predicción LSTM'].iloc[-1], 'ro', label='Predicción (siguiente día)')
plt.title('Predicción del Precio de Bitcoin (Siguiente Día)')
plt.xlabel('Fecha')
plt.ylabel('Precio de Bitcoin')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# -------------------------------------------------- Sistema de Prediccion de 7 Dias (LSTM) --------------------------------------------------

# Predicción para los próximos 7 días
predicted_prices_7_days = []
current_input_7days = last_60_days.copy()

for i in range(7):
    X_input_7days = np.array([current_input_7days])
    predicted_price_scaled_7days = model.predict(X_input_7days)
    
    # Imprimir las dimensiones y el contenido de las variables
    print(f"Dimensiones de current_input_7days: {current_input_7days.shape}")
    print(f"Contenido de current_input_7days: {current_input_7days}")
    print(f"Dimensiones de predicted_price_scaled_7days: {predicted_price_scaled_7days.shape}")
    print(f"Contenido de predicted_price_scaled_7days: {predicted_price_scaled_7days}")
    
    # Crear una matriz para actualizar current_input_7days
    predicted_price_scaled_7days_full = current_input_7days[-1].copy()
    predicted_price_scaled_7days_full[-1] = predicted_price_scaled_7days
    predicted_price_full_7days = scaler.inverse_transform(
        np.concatenate((current_input_7days[-1, :-1].reshape(1, -1), predicted_price_scaled_7days), axis=1)
    )
    predicted_price_7days = predicted_price_full_7days[:, -1]
    predicted_prices_7_days.append(predicted_price_7days[0])
    
    # Actualizar current_input_7days para la siguiente iteración
    current_input_7days = np.append(current_input_7days[1:], predicted_price_scaled_7days_full.reshape(1, -1), axis=0)
    
    # Imprimir las dimensiones y el contenido de current_input_7days después de la actualización
    print(f"Dimensiones de current_input_7days después de la actualización: {current_input_7days.shape}")
    print(f"Contenido de current_input_7days después de la actualización: {current_input_7days}")

predicted_dates_7_days = [df['Date'].values[-1] + np.timedelta64(i + 1, 'D') for i in range(7)]

# Crear DataFrame para los últimos 7 días y la predicción de los próximos 7 días
last_7_days_prices = df['Close'].values[-7:]
last_7_days_dates = df['Date'].values[-7:]
predicted_data_7_days = {
    'Fecha': np.append(last_7_days_dates, predicted_dates_7_days),
    'Precio Real': np.append(last_7_days_prices, [np.nan] * 7),
    'Predicción LSTM': np.append([np.nan] * 7, predicted_prices_7_days)
}
prediction_7_days_df = pd.DataFrame(predicted_data_7_days)
print(prediction_7_days_df)

# Gráfica con los precios reales de los últimos 7 días y los 7 días predichos
plt.figure(figsize=(14, 7))
plt.plot(prediction_7_days_df['Fecha'][:7], prediction_7_days_df['Precio Real'][:7], color='blue', label='Precio Real (últimos 7 días)')
plt.plot(prediction_7_days_df['Fecha'][7:], prediction_7_days_df['Predicción LSTM'][7:], 'ro-', label='Predicción (próximos 7 días)')
plt.title('Predicción del Precio de Bitcoin (Próximos 7 Días)')
plt.xlabel('Fecha')
plt.ylabel('Precio de Bitcoin')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
