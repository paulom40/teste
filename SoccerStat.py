import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import poisson
import numpy as np
import requests
from io import StringIO

# Download & process (use local CSVs if URL fails)
def process_league(code):
    url = f'http://www.football-data.co.uk/mmz4281/2425/{code}.csv'
    try:
        df = pd.read_csv(url)
    except:
        df = pd.read_csv(f'{code}_2425.csv')  # Local fallback
    df = df.dropna(subset=['FTHG', 'FTAG'])
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    df['HomeForm'] = df['FTHG'].rolling(5).mean().shift(1).fillna(df['FTHG'].mean())
    df['AwayForm'] = df['FTAG'].rolling(5).mean().shift(1).fillna(df['FTAG'].mean())
    return df

# Leagues
leagues = {'E0': 'EPL', 'I1': 'Serie A', 'D1': 'Bundesliga', 'F1': 'Ligue 1', 'SP1': 'La Liga'}
models = {}
results = []

for code, name in leagues.items():
    df = process_league(code)
    if len(df) > 20:  # Min data
        # Poisson for home goals (season-tuned)
        model = smf.poisson('FTHG ~ HomeForm + AwayForm', data=df).fit(disp=0)
        models[name] = model
        
        # Metrics
        home_pred = model.predict(df[['HomeForm', 'AwayForm']])
        away_pred = df['FTAG'].mean()  # Simplified
        mae = np.mean(np.abs(home_pred - df['FTHG']))
        p_home_win = np.mean(poisson.cdf(np.arange(0, 10), away_pred) * np.cumsum(poisson.pmf(np.arange(1, 11), home_pred[::-1])[::-1]))
        win_acc = np.mean((df['FTHG'] > df['FTAG']).astype(int) == (home_pred > away_pred).astype(int))
        over_acc = np.mean((df['TotalGoals'] > 2.5).astype(int) == (home_pred + away_pred > 2.5).astype(int))
        
        results.append({
            'League': name,
            'Matches': len(df),
            'MAE (Goals)': round(mae, 2),
            'Win Acc': f"{win_acc:.0%}",
            'Over 2.5 Acc': f"{over_acc:.0%}",
            'Avg P(Home Win)': f"{p_home_win:.0%}"
        })

# Output
df_results = pd.DataFrame(results)
print(df_results.to_markdown(index=False))
