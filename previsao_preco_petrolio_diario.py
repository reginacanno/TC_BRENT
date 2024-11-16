import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def exibir_projecoes(df):
    st.header("Previsão do Preço do Petrólio Diário")

    # Gráfico de Projeções Simuladas
    st.subheader("Tendências e Impactos Futuros")
    st.markdown("""
    - **Transição Energética**: A adoção de energias renováveis deve desacelerar a demanda por petróleo.
    - **Políticas Ambientais**: Metas de emissão líquida zero moldam a evolução do mercado.
    - **Incertezas Geopolíticas**: Tensões em regiões produtoras continuarão a influenciar os preços.
    - **Inovações**: Tecnologias de captura de carbono ajudam a reduzir o impacto ambiental.
    """)

    # Gráfico de Projeções
    st.subheader("Gráfico de Projeções (Simulado)")
    dados_agrupados = df.groupby(df['date'].dt.year)['price'].mean()
    anos_futuros = np.arange(2023, 2033)
    precos_futuros = np.linspace(dados_agrupados.mean(), dados_agrupados.mean() * 0.8, len(anos_futuros))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(anos_futuros, precos_futuros, marker='o', linestyle='--', color='green', label='Projeção de Preço')
    ax.set_title("Projeção do Preço do Petróleo (2023-2032)")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Preço Estimado (US$)")
    ax.grid(alpha=0.5)
    ax.legend()
    st.pyplot(fig)

