import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

def exibir_projecoes(df=None):
    """
    Exibe as previsões e métricas no Streamlit.
    """
    st.header("Projeções de Preços do Petróleo Brent")

    # Carregar o modelo
    model = load_model(r"C:\Users\regin\Documents\Postech\Projeto-BRENT\tc-brent\modelo_lstm_brent.h5")

    # Carregar os dados
    file_path = r"C:\Users\regin\Documents\Postech\Projeto-BRENT\tc-brent\dados_treinamento.pkl"
    X_train, X_test, y_train, y_test, scaler = joblib.load(file_path)

    # Fazer previsões
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverter a escala das previsões e dos dados reais
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train = scaler.inverse_transform([y_train])
    y_test = scaler.inverse_transform([y_test])

    # Avaliar o modelo
    mse = mean_squared_error(y_test[0], test_predict[:, 0])
    mae = mean_absolute_error(y_test[0], test_predict[:, 0])
    r2 = r2_score(y_test[0], test_predict[:, 0])

    # Exibir métricas
    st.subheader("Métricas de Avaliação")
    st.write(f"Erro Quadrático Médio (MSE): {mse:.2f}")
    st.write(f"Erro Médio Absoluto (MAE): {mae:.2f}")
    st.write(f"Coeficiente de Determinação (R²): {r2:.2f}")

    # Gráfico de comparação
    st.subheader("Preços Reais vs. Previstos")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test[0], label="Real", color="blue")
    ax.plot(test_predict[:, 0], label="Previsto", color="orange")
    ax.set_title("Comparação de Preços Reais e Previstos", fontsize=16)
    ax.set_xlabel("Tempo", fontsize=14)
    ax.set_ylabel("Preço (US$)", fontsize=14)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
