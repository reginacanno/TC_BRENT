import streamlit as st
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
from pandas.tseries.offsets import BDay  # Para calcular dias úteis

def exibir_projecoes(df=None):
    """
    Exibe as previsões e métricas no Streamlit.
    """
    st.header("Projeções de Preços do Petróleo Brent")

    # Introdução sobre o modelo e a aba
    st.markdown("""
    Esta seção utiliza um modelo de aprendizado profundo baseado em Redes Neurais Recorrentes (RNN) do tipo **LSTM** 
    (*Long Short-Term Memory*), ideal para lidar com séries temporais como os preços do petróleo. 
    Com base nos dados históricos, o modelo faz previsões e avalia o desempenho nas comparações entre valores reais e previstos. 

    Além disso, apresentamos uma projeção dos preços para os próximos **30 dias úteis**, com base nos padrões aprendidos pelo modelo.
    """)

    # Carregar o modelo
    model_path = os.path.join(os.path.dirname(__file__), "modelo_lstm_brent.h5")
    model = load_model(model_path)

    # Carregar os dados
    file_path = os.path.join(os.path.dirname(__file__), "dados_treinamento.pkl")
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

    # Apresentação das métricas de forma amigável
    st.subheader("Como o Modelo Está Performando?")
    st.markdown(f"""
    - **Precisão do Modelo (R²)**: O modelo consegue explicar cerca de **{r2:.2%}** da variação nos preços do petróleo.
    - **Erro Médio Absoluto (MAE)**: Em média, o modelo erra em torno de **${mae:.2f}** por barril, o que é considerado razoável.
    - **Erro Quadrático Médio (MSE)**: O erro médio quadrático foi de **{mse:.2f}**, um valor aceitável considerando a volatilidade dos preços.
    """)

    # Gráfico de comparação
    st.subheader("Comparação de Preços Reais e Previstos")
    datas = df['date'][-len(y_test[0]):]  # Selecionar datas correspondentes aos valores previstos
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(datas, y_test[0], label="Real", color="blue", linewidth=2)
    ax.plot(datas, test_predict[:, 0], label="Previsto", color="orange", linewidth=2)
    ax.set_title("Comparação de Preços Reais e Previstos", fontsize=16)
    ax.set_xlabel("Data", fontsize=14)
    ax.set_ylabel("Preço (US$)", fontsize=14)
    ax.legend()
    ax.grid(visible=False)  # Sem linhas de grade
    ax.set_facecolor('white')  # Fundo branco no gráfico

    # Ajustar rótulos para formato brasileiro, exibindo rótulos espaçados
    espacamento = max(1, len(datas) // 10)  # Exibir no máximo 10 rótulos
    datas_formatadas = [data.strftime('%d/%m/%Y') if i % espacamento == 0 else '' for i, data in enumerate(datas)]
    ax.set_xticks(datas)
    ax.set_xticklabels(datas_formatadas, rotation=45, fontsize=10, ha='right')

    st.pyplot(fig)

    # Predição para os próximos 30 dias úteis
    st.subheader("Projeção para os Próximos 30 Dias Úteis")
    st.markdown("""
    Abaixo estão os preços previstos para os próximos **30 dias úteis**, considerando os padrões identificados nos dados históricos.
    """)

    ultima_amostra = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])  # Última entrada

    # Criar a projeção dia a dia
    previsoes_futuras = []
    for _ in range(30):
        predicao = model.predict(ultima_amostra)
        previsoes_futuras.append(predicao[0, 0])
        # Atualizar última amostra para incluir a previsão
        ultima_amostra = np.append(ultima_amostra[:, 1:, :], [[predicao[0]]], axis=1)

    # Reverter a escala das previsões futuras
    previsoes_futuras = scaler.inverse_transform(np.array(previsoes_futuras).reshape(-1, 1))

    # Adicionar datas para os próximos 30 dias úteis
    ultima_data = df['date'].max() + BDay()  # Próximo dia útil após a última data
    proximas_datas = pd.date_range(start=ultima_data, periods=30, freq=BDay())  # 30 dias úteis

    # Gráfico de projeções futuras
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(proximas_datas, previsoes_futuras, label="Previsão", color="green", marker="o", linestyle="--")
    ax2.set_title("Projeção dos Preços para os Próximos 30 Dias Úteis", fontsize=16)
    ax2.set_xlabel("Data", fontsize=14)
    ax2.set_ylabel("Preço (US$)", fontsize=14)
    ax2.legend()
    ax2.grid(visible=False)  # Sem linhas de grade
    ax2.set_facecolor('white')  # Fundo branco no gráfico

    # Ajustar rótulos para formato brasileiro, exibindo no máximo 10 rótulos
    espacamento_futuro = max(1, len(proximas_datas) // 10)  # Exibir no máximo 10 rótulos
    proximas_datas_formatadas = [data.strftime('%d/%m/%Y') if i % espacamento_futuro == 0 else '' for i, data in enumerate(proximas_datas)]
    ax2.set_xticks(proximas_datas)
    ax2.set_xticklabels(proximas_datas_formatadas, rotation=45, fontsize=10, ha='right')

    st.pyplot(fig2)

    # Exibir tabela de previsões
    tabela_previsoes = pd.DataFrame({
        "Data": [data.strftime('%d/%m/%Y') for data in proximas_datas],
        "Preço Previsto (US$)": previsoes_futuras.flatten()
    })

    # Ajustar estilo da tabela
    st.subheader("Tabela de Projeções")
    st.dataframe(tabela_previsoes.style.format({"Preço Previsto (US$)": "{:.2f}"}), use_container_width=True)
