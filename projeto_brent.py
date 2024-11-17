import streamlit as st
import pandas as pd
from ipeadatapy import timeseries
import matplotlib.pyplot as plt
import numpy as np
import previsoes  # Importa a lógica da segunda aba

# Autoria no canto direito da tela, em itálico e fonte menor
st.markdown("""
<div style="text-align: right; font-size: 12px; font-style: italic;">
Projeto desenvolvido por Regina Canno para o TechChallenge FIAP - Novembro de 2024
</div>
""", unsafe_allow_html=True)

# Título do Aplicativo
st.title("Dashboard Interativo - Preço do Petróleo Brent")

# Introdução
st.markdown("""
## O que é o Petróleo Brent?

O Brent é um dos principais referenciais globais de precificação do petróleo. Ele representa uma mistura de petróleo extraído do Mar do Norte, na região entre a Noruega e o Reino Unido. Por ser leve e conter baixo teor de enxofre, é ideal para a produção de gasolina e diesel de alta qualidade.

O Brent é amplamente utilizado como padrão para precificar o petróleo no mercado internacional, influenciando diretamente os preços globais de energia. Este dashboard apresenta dados históricos e projeções do preço do petróleo Brent, permitindo uma análise detalhada das tendências e eventos que impactaram o mercado.
""")

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

# Ajustar formato para datas no padrão brasileiro
df['date_br'] = df['date'].dt.strftime('%d/%m/%Y')

# Criar abas
abas = st.tabs(["Dados Históricos", "Previsões"])

# Aba 1: Dados Históricos
with abas[0]:
    st.header("Dados Históricos")

    # Filtro com seleção de intervalo de datas no formato calendário
    st.sidebar.subheader("Filtros")

    # Valores padrão para o filtro de datas
    valor_padrao_inicio = df['date'].min().date()
    valor_padrao_fim = df['date'].max().date()

    # Data Inicial
    data_inicio = st.sidebar.date_input(
        "Data Inicial:",
        value=valor_padrao_inicio,
        min_value=valor_padrao_inicio,
        max_value=valor_padrao_fim,
        format="DD/MM/YYYY"
    )

    # Data Final
    data_fim = st.sidebar.date_input(
        "Data Final:",
        value=valor_padrao_fim,
        min_value=valor_padrao_inicio,
        max_value=valor_padrao_fim,
        format="DD/MM/YYYY"
    )

    # Garantir que a data final seja maior ou igual à inicial
    if data_fim < data_inicio:
        st.sidebar.error("A data final não pode ser anterior à data inicial.")
        st.stop()

    # Converter datas para formato compatível com a coluna 'date' (com fuso horário UTC)
    data_inicio = pd.Timestamp(data_inicio).tz_localize("UTC")
    data_fim = pd.Timestamp(data_fim).tz_localize("UTC")

    # Filtrar os dados com base nas datas selecionadas
    dados_filtrados = df[(df['date'] >= data_inicio) & (df['date'] <= data_fim)]

    # Métricas Gerais
    st.subheader("Métricas Gerais")
    col1, col2, col3 = st.columns(3)
    col1.metric("Preço Máximo", f"${dados_filtrados['price'].max():.2f}")
    col2.metric("Preço Mínimo", f"${dados_filtrados['price'].min():.2f}")
    col3.metric("Preço Médio", f"${dados_filtrados['price'].mean():.2f}")

    # Gráfico de Evolução Temporal
    st.subheader("Evolução do Preço do Petróleo Brent")
    fig, ax = plt.subplots(figsize=(18, 9))  # Aumentar tamanho do gráfico
    ax.plot(dados_filtrados['date'], dados_filtrados['price'], label='Preço do Petróleo Brent', color='blue')

    if not dados_filtrados.empty:
        # Garantir sempre 31 rótulos no eixo X ou menos
        num_ticks = min(31, len(dados_filtrados))  # Limita a 31 rótulos ou ao total disponível
        indices = np.linspace(0, len(dados_filtrados) - 1, num_ticks, dtype=int)
        tick_positions = dados_filtrados.iloc[indices]['date']
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            tick_positions.dt.strftime('%d/%m/%Y'), 
            rotation=45, 
            ha="right", 
            fontsize=16  # Aumentar fonte dos rótulos
        )

    ax.set_title("Evolução do Preço do Petróleo Brent", fontsize=20)
    ax.set_xlabel("Data", fontsize=18)
    ax.set_ylabel("Preço (US$)", fontsize=18)
    ax.grid(alpha=0.5)
    st.pyplot(fig)


    # Insights Explicativos
    st.subheader("Grandes Impactos")
    st.markdown("""
    #### 1998: Crise Financeira Asiática e Colapso dos Preços do Petróleo
    - A crise financeira asiática, iniciada em 1997, impactou severamente economias emergentes como Tailândia, Indonésia e Coreia do Sul, levando a uma recessão na região.
    - A demanda por petróleo diminuiu consideravelmente na Ásia, um dos maiores consumidores globais.
    - A OPEP manteve altos níveis de produção, criando um excesso de oferta no mercado.
    - A crise financeira na Rússia em 1998 agravou a situação global, aumentando a oferta e reduzindo ainda mais os preços. O petróleo atingiu níveis historicamente baixos, com o Brent sendo comercializado a cerca de US$ 10 por barril.

    #### 2008: Crise Financeira Global
    - A falência do Lehman Brothers em setembro de 2008 desencadeou uma crise financeira global, resultando em uma recessão generalizada.
    - Após atingir um pico de US$ 147 por barril em julho de 2008, o preço despencou para menos de US$ 40 no final do ano.
    - A desaceleração econômica global reduziu drasticamente a demanda por petróleo, especialmente nos setores de transporte e manufatura.

    #### 2011 e 2012: Altas Históricas nos Preços do Petróleo
    - A Primavera Árabe (2010-2012) levou a conflitos e instabilidade em países produtores de petróleo, como Líbia e Egito.
    - Economias emergentes, como China e Índia, apresentaram forte crescimento econômico, aumentando significativamente a demanda por energia.
    - O Brent ultrapassou os US$ 110 por barril em 2011 e 2012, marcando um dos períodos mais caros da história recente.
    - A combinação de incertezas políticas no Oriente Médio e a crescente demanda global sustentaram preços elevados.

    #### 2020: Pandemia de COVID-19
    - O início da pandemia levou a lockdowns globais e ao fechamento de fronteiras, interrompendo as cadeias de suprimentos e reduzindo a atividade econômica.
    - A demanda por petróleo despencou devido à interrupção de viagens e transporte terrestre.
    - Em abril de 2020, o preço do petróleo nos Estados Unidos chegou a ser negativo pela primeira vez na história, refletindo o excesso de oferta e a falta de capacidade de armazenamento.

    #### 2022: Conflito Rússia-Ucrânia
    - A invasão da Ucrânia pela Rússia em fevereiro de 2022 gerou uma crise geopolítica global e sanções econômicas contra a Rússia.
    - A Rússia, um dos maiores produtores de petróleo, enfrentou sanções que restringiram suas exportações.
    - O Brent atingiu cerca de US$ 130 por barril em março de 2022, refletindo as incertezas sobre o fornecimento de energia.
    """)

    # Comparação Anual
    st.subheader("Comparação Anual: Preço Médio do Petróleo")
    dados_filtrados['year'] = dados_filtrados['date'].dt.year
    media_anual = dados_filtrados.groupby('year')['price'].mean()

    fig2, ax2 = plt.subplots(figsize=(18, 9))  # Aumentar tamanho do gráfico
    bar_width = 0.6
    positions = np.arange(len(media_anual))
    bars = ax2.bar(positions, media_anual, width=bar_width, color='orange', label='Preço Médio Anual')
    ax2.set_xticks(positions)
    ax2.set_xticklabels(media_anual.index.astype(str), rotation=45, fontsize=16)  # Aumentar fonte
    ax2.set_title("Média Anual do Preço do Petróleo", fontsize=20)
    ax2.set_xlabel("Ano", fontsize=18)
    ax2.set_ylabel("Preço Médio (US$)", fontsize=18)

    for bar in bars:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.1f}",
            ha='center',
            va='bottom',
            fontsize=16  # Aumentar fonte dos rótulos
        )
    st.pyplot(fig2)

    

# Aba 2: Previsões (Chamar lógica de outro arquivo)
with abas[1]:
    previsoes.exibir_projecoes(df)
