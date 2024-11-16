import streamlit as st
import pandas as pd
from ipeadatapy import timeseries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

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

# Filtro por Ano
st.sidebar.subheader("Filtros")

# Checkbox para selecionar tudo
select_all = st.sidebar.checkbox("Selecionar Tudo", value=False)

# Multiselect para anos
if select_all:
    anos = st.sidebar.multiselect(
        "Selecione o(s) Ano(s)",
        options=todos_os_anos,
        default=todos_os_anos  # Seleciona todos os anos automaticamente
    )
else:
    anos = st.sidebar.multiselect(
        "Selecione o(s) Ano(s)",
        options=todos_os_anos,
        default=[ultimo_ano]  # Seleciona o último ano como padrão
    )

# Filtrar os dados com base nos anos selecionados
dados_filtrados = df[df['date'].dt.year.isin(anos)]

# Exibir Métricas Gerais (com base nos dados filtrados)
st.subheader("Métricas Gerais")
col1, col2, col3 = st.columns(3)
col1.metric("Preço Máximo", f"${dados_filtrados['price'].max():.2f}")
col2.metric("Preço Mínimo", f"${dados_filtrados['price'].min():.2f}")
col3.metric("Preço Médio", f"${dados_filtrados['price'].mean():.2f}")

# Gráfico de Evolução Temporal (com base nos dados filtrados)
st.subheader("Evolução do Preço do Petróleo Brent (Filtrado)")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dados_filtrados['date'], dados_filtrados['price'], label='Preço do Petróleo Brent', color='blue')

# Configurar os rótulos do eixo X dinamicamente
if not dados_filtrados.empty:
    # Obter o número de meses disponíveis
    meses_disponiveis = dados_filtrados['date'].dt.month.nunique()

    # Se houver menos de 12 meses, ajustar para o número disponível
    num_ticks = min(12, meses_disponiveis)
    
    # Gerar posições uniformes para os rótulos
    indices = np.linspace(0, len(dados_filtrados) - 1, num_ticks, dtype=int)
    tick_positions = dados_filtrados.iloc[indices]['date']
    
    # Aplicar os rótulos calculados
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions.dt.strftime('%Y-%m'))

# Configurar título e rótulos
ax.set_title("Evolução do Preço do Petróleo Brent (Filtrado por Ano)")
ax.set_xlabel("Data")
ax.set_ylabel("Preço (US$)")
ax.grid(alpha=0.5)

# Rotacionar os rótulos para melhor visualização
plt.xticks(rotation=45)
st.pyplot(fig)
