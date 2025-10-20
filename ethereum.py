import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Teste AwesomeAPI", layout="centered")
st.title("💱 Teste de conexão com AwesomeAPI")

# Pares confirmados pela documentação oficial
pares_validos = [
    "USD-BRL", "USDT-BRL", "CAD-BRL", "AUD-BRL", "EUR-BRL",
    "GBP-BRL", "ARS-BRL", "JPY-BRL", "CHF-BRL", "BTC-BRL",
    "LTC-BRL", "CNY-BRL", "ILS-BRL"
]

# Seleção do par
par = st.selectbox("Seleciona o par de moedas", pares_validos)

# Botão para atualizar
atualizar = st.button("🔄 Atualizar dados")

# Função com cache estendido
@st.cache_data(ttl=900)
def get_data(par):
    url = f"https://economia.awesomeapi.com.br/json/daily/{par}/30"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0 and 'bid' in data[0]:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df['price'] = df['bid'].astype(float)
                df.set_index('timestamp', inplace=True)
                return df[['price']]
            else:
                st.error("❌ Dados inválidos ou vazios retornados pela API.")
                return pd.DataFrame()
        elif response.status_code == 429:
            st.error("❌ Erro 429: Limite de requisições excedido. Tente novamente em alguns minutos.")
            return pd.DataFrame()
        else:
            st.error(f"❌ Erro na API: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Erro de conexão: {e}")
        return pd.DataFrame()

# Executar somente se o botão for clicado
if atualizar:
    df = get_data(par)
    if not df.empty:
        st.line_chart(df['price'])
        st.success("✅ Dados carregados com sucesso.")
    else:
        st.warning("Nenhum dado disponível para este par.")
else:
    st.info("Clique em 'Atualizar dados' para buscar os preços.")
