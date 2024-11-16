import streamlit as st
import pandas as pd
from ipeadatapy import timeseries
import matplotlib.pyplot as plt
import numpy as np
import previsao_preco_petrolio_diario  # Importa a lógica da segunda aba

# Título do Aplicativo
st.title("Dashboard Interativo - Preço do Petróleo Brent")

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

# Verificar se 'date' é datetime
if not pd.api.types.is_datetime64_any_dtype(df['date']):
    st.error("A coluna 'date' não está no formato datetime.")

# Obter o último ano disponível nos dados
ultimo_ano = df['date'].dt.year.max()

# Obter lista completa de anos
todos_os_anos = df['date'].dt.year.unique()

# Criar abas
abas = st.tabs(["Dados Históricos", "Previsões"])

# Aba 1: Dados Históricos
with abas[0]:
    st.header("Dados Históricos")

    # Filtro por Ano
    st.sidebar.subheader("Filtros")
    anos = st.sidebar.multiselect(
        "Selecione o(s) Ano(s)",
        options=todos_os_anos,
        default=[ultimo_ano]
    )

    dados_filtrados = df[df['date'].dt.year.isin(anos)]

    # Métricas Gerais
    st.subheader("Métricas Gerais")
    col1, col2, col3 = st.columns(3)
    col1.metric("Preço Máximo", f"${dados_filtrados['price'].max():.2f}")
    col2.metric("Preço Mínimo", f"${dados_filtrados['price'].min():.2f}")
    col3.metric("Preço Médio", f"${dados_filtrados['price'].mean():.2f}")

    # Gráfico de Evolução Temporal
    st.subheader("Evolução do Preço do Petróleo Brent (Filtrado)")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dados_filtrados['date'], dados_filtrados['price'], label='Preço do Petróleo Brent', color='blue')

    if not dados_filtrados.empty:
        meses_disponiveis = dados_filtrados['date'].dt.month.nunique()
        num_ticks = min(12, meses_disponiveis)
        indices = np.linspace(0, len(dados_filtrados) - 1, num_ticks, dtype=int)
        tick_positions = dados_filtrados.iloc[indices]['date']
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions.dt.strftime('%Y-%m'), rotation=45, ha="right")

    # Anotações de eventos
    eventos = {
        "Crise Financeira Asiática (1998)": "1998-01-01",
        "Crise Financeira Global (2008)": "2008-09-15",
        "Pandemia de COVID-19 (2020)": "2020-03-11",
        "Conflito Rússia-Ucrânia (2022)": "2022-02-24",
    }

    for evento, data in eventos.items():
        data_evento = pd.to_datetime(data)
        if data_evento in dados_filtrados['date'].values:
            ax.axvline(data_evento, color='red', linestyle='--', alpha=0.7)
            ax.text(data_evento, dados_filtrados['price'].max(), evento, rotation=90, color='red')

    ax.set_title("Evolução do Preço do Petróleo Brent")
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço (US$)")
    ax.grid(alpha=0.5)
    st.pyplot(fig)

    # Comparação Anual
    st.subheader("Comparação Anual: Preço Médio do Petróleo")
    dados_filtrados['year'] = dados_filtrados['date'].dt.year
    media_anual = dados_filtrados.groupby('year')['price'].mean()

    fig2, ax2 = plt.subplots(figsize=(14, 6))
    bar_width = 0.6
    positions = np.arange(len(media_anual))
    bars = ax2.bar(positions, media_anual, width=bar_width, color='orange', label='Preço Médio Anual')
    ax2.set_xticks(positions)
    ax2.set_xticklabels(media_anual.index.astype(str), rotation=45)
    ax2.set_title("Média Anual do Preço do Petróleo")
    ax2.set_xlabel("Ano")
    ax2.set_ylabel("Preço Médio (US$)")

    for bar in bars:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.1f}",
            ha='center',
            va='bottom',
            fontsize=9
        )
    st.pyplot(fig2)

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

# Aba 2: Projeções (Chamar lógica de outro arquivo)
with abas[1]:
    previsao_preco_petrolio_diario.exibir_projecoes(df)

