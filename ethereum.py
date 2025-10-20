import requests
import pandas as pd

def get_forex_data(symbol="USD-BRL", days=30):
    url = f"https://economia.awesomeapi.com.br/json/daily/{symbol}/{days}"
    response = requests.get(url)
    
    if response.status_code == 200:
        try:
            data = response.json()
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('timestamp')
            df['price'] = df['bid'].astype(float)
            df.set_index('timestamp', inplace=True)
            return df[['price']]
        except Exception as e:
            print("Erro ao processar os dados:", e)
            return pd.DataFrame()
    else:
        print("Erro na API:", response.status_code)
        return pd.DataFrame()

# Teste local
df = get_forex_data("USD-BRL", 10)
print(df.tail())
