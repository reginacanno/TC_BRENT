# %%
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipeadatapy import timeseries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense

# %%
# Carregar Dados Diretamente da API
@st.cache_data
def load_data():
    # Código da série do Petróleo Brent
    codigo_serie = "EIA366_PBRENT366"

    # Obter os dados da API
    df = timeseries(codigo_serie)

    # Converter 'RAW DATE' para datetime
    df['RAW DATE'] = pd.to_datetime(df['RAW DATE'], errors='coerce', utc=True)

    # Remover linhas inválidas (sem data ou valor de preço)
    df = df.dropna(subset=['RAW DATE', 'VALUE (US$)'])

    # Renomear as colunas
    df.rename(columns={'VALUE (US$)': 'price', 'RAW DATE': 'date'}, inplace=True)

    return df

# Carregar os dados
df = load_data()

# %%
# Detectar outliers (usando o método IQR)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

# %%
# Dividir os dados em treinamento e teste aleatoriamente
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Ordenar os dados por data após a amostragem
train_data = train_data.sort_values('DATE')
test_data = test_data.sort_values('DATE')

# %%
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data['price'].values.reshape(-1, 1))
scaled_test_data = scaler.transform(test_data['price'].values.reshape(-1, 1))

# %%
# Função para criar uma estrutura de dados com janelas de tempo
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Criar o conjunto de dados para o modelo LSTM
time_step = 10
X_train, y_train = create_dataset(scaled_train_data, time_step)
X_test, y_test = create_dataset(scaled_test_data, time_step)

# Redimensionar a entrada para [amostras, time steps, features] que é necessário para LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# %%
# Criar o modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)

# %%
# Salvar o modelo
model.save("modelo_lstm_brent.h5")

# Salvar os dados de treino e teste
joblib.dump((X_train, X_test, y_train, y_test, scaler), "dados_treinamento.pkl")


