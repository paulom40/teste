import requests
import pandas as pd

def get_forex_data(symbol="USD-BRL", days=30):
    url = f"https://economia.awesomeapi.com.br/json/daily/{symbol}/{days}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('timestamp')
        df['price'] = df['bid'].astype(float)
        df.set_index('timestamp', inplace=True)
        return df[['price']]
    else:
        print("Erro na API. Verifica o par ou o formato.")
        return pd.DataFrame()

df = get_forex_data()
print(df.tail())
