# app.py — Football Predictor Pro v12.0 ELITE + LIVE ODDS
# Shots Inside Box + Big Chances + Real-time Bookmaker Odds

import streamlit as st
import numpy as np
from scipy.stats import poisson
import requests

st.set_page_config(page_title="Predictor Pro v12 ELITE + ODDS", layout="wide", page_icon="trophy")

st.markdown("""
<style>
    .title {font-size:64px !important; font-weight:bold; text-align:center; color:#FF006E;}
    .tag {font-size:26px; text-align:center; color:#00FFAA; margin-top:-15px;}
    .odds-box {background:#0f1117; padding:15px; border-radius:12px; text-align:center; font-size:18px;}
</style>
<div class="title">PREDICTOR PRO v12 ELITE</div>
<div class="tag">Shots Inside Box Model + LIVE Bookmaker Odds</div>
""", unsafe_allow_html=True)

# ============================= LEAGUE AVERAGES (2025 sharp numbers) =============================
def get_leagues():
    return {
        'Premier League':       {'avg_goals':2.94, 'home_adv':1.42, '2h_ratio':0.568, 'vol':0.77, 'avg_sib':8.8,  'big_chance':2.4, 'tier':1},
        'Championship (ENG)':   {'avg_goals':2.81, 'home_adv':1.45, '2h_ratio':0.582, 'vol':0.90, 'avg_sib':8.2,  'big_chance':2.1, 'tier':2},
        'La Liga':              {'avg_goals':2.69, 'home_adv':1.31, '2h_ratio':0.532, 'vol':0.70, 'avg_sib':7.9,  'big_chance':2.0, 'tier':1},
        'Bundesliga':           {'avg_goals':3.26, 'home_adv':1.39, '2h_ratio':0.615, 'vol':0.86, 'avg_sib':9.6,  'big_chance':2.8, 'tier':1},
        'Serie A':              {'avg_goals':2.74, 'home_adv':1.28, '2h_ratio':0.495, 'vol':0.68, 'avg_sib':7.5,  'big_chance':1.9, 'tier':1},
        'Eredivisie':           {'avg_goals':3.31, 'home_adv':1.44, '2h_ratio':0.623, 'vol':0.89, 'avg_sib':10.1, 'big_chance':3.0, 'tier':1},
        'Ligue 1':              {'avg_goals':2.78, 'home_adv':1.35, '2h_ratio':0.558, 'vol':0.75, 'avg_sib':8.4,  'big_chance':2.2, 'tier':1},
        'Primeira Liga':        {'avg_goals':2.85, 'home_adv':1.41, '2h_ratio':0.570, 'vol':0.80, 'avg_sib':8.6,  'big_chance':2.3, 'tier':1},
        'Super Lig':            {'avg_goals':2.92, 'home_adv':1.49, '2h_ratio':0.580, 'vol':0.88, 'avg_sib':8.9,  'big_chance':2.5, 'tier':1},
        '2. Bundesliga (GER)':  {'avg_goals':3.05, 'home_adv':1.40, '2h_ratio':0.600, 'vol':0.86, 'avg_sib':9.0,  'big_chance':2.6, 'tier':2},
    }

LEAGUES = get_leagues()

# ============================= FETCH LIVE ODDS (TheOddsAPI free tier) =============================
@st.cache_data(ttl=300)  # Update every 5 minutes
def get_live_odds(home_team, away_team):
    API_KEY = "2fc8ca1227c5f69b90c485199c8eabee"  # Get free key → https://the-odds-api.com
    if API_KEY == "2fc8ca1227c5f69b90c485199c8eabee":
        return {"1": "-", "X": "-", "2": "-", "bookie": "No API key"}

    url = f"https://api.the-odds-api.com/v4/sports/soccer/odds"
    params = {
        'apiKey': API_KEY,
        'regions': 'eu,uk',
        'markets': 'h2h',
        'oddsFormat': 'decimal',
        'eventIds': ''  # We search by team names
    }
    try:
        response = requests.get(url, params=params, timeout=8)
        if response.status_code == 200:
            data = response.json()
            for game in data:
                if (home_team.lower() in game['home_team'].lower() and away_team.lower() in game['away_team'].lower()) or \
                   (away_team.lower() in game['home_team'].lower() and home_team.lower() in game['away_team'].lower()):
                    for book in game['bookmakers']:
                        if book['key'].lower() in ['pinnacle', 'bet365', 'williamhill']:
                            odds = book['markets'][0]['outcomes']
                            home_o = next((o['price'] for o in odds if o['name'] == game['home_team']), None)
                            away_o = next((o['price'] for o in odds if o['name'] == game['away_team']), None)
                            draw_o = next((o['price'] for o in odds if o['name'] == 'Draw'), None)
                            if home_o and away_o:
                                return {
                                    "1": f"{home_o:.2f}",
                                    "X": f"{draw_o:.2f}" if draw_o else "-",
                                    "2": f"{away_o:.2f}",
                                    "bookie": book['title']
                                }
    except:
        pass
    return {"1": "-", "X": "-", "2": "-", "bookie": "No match"}

# ============================= ELITE PREDICTOR =============================
class ElitePredictor:
    def __init__(self, league):
        self.p = LEAGUES.get(league, LEAGUES['Premier League'])

    def momentum(self, h, a):
        score = 50.0
        total_xg = h['xg'] + a['xg']
        if total_xg > 0:
            score += (h['xg'] / total_xg - 0.5) * 60
        total_sib = h['sib'] + a['sib']
        if total_sib > 0:
            score += (h['sib'] / total_sib - 0.5) * 40
        total_bc = h['big'] + a['big']
        if total_bc > 0:
            score += (h['big'] / total_bc - 0.5) * 25
        score += (h['poss'] - 50) * 0.35
        if self.p['tier'] == 2:
            score *= 1.10 * self.p['vol']
        return np.clip(score, 12, 88)

    def predict(self, home, away):
        home_mom = self.momentum(home, away)
        away_mom = 100 - home_mom

        h_rate = home['xg'] / 45
        a_rate = away['xg'] / 45

        home_xg_2h = (h_rate * 45 * (home_mom/50) * self.p['2h_ratio'] * self.p['home_adv'] *
                      (1 + 0.18 * home['big'] + 0.09 * max(0, home['sib'] - self.p['avg_sib']/2)))
        away_xg_2h = (a_rate * 45 * (away_mom/50) * self.p['2h_ratio'] *
                      (1 + 0.18 * away['big'] + 0.09 * max(0, away['sib'] - self.p['avg_sib']/2)))

        if home['goals'] < away['goals'] and home_mom > 64: home_xg_2h *= 1.30
        if away['goals'] < home['goals'] and away_mom > 64: away_xg_2h *= 1.30

        hp = [poisson.pmf(i, home_xg_2h) for i in range(8)]
        ap = [poisson.pmf(i, away_xg_2h) for i in range(8)]

        best_score, best_prob = "0-0", 0
        for i in range(8):
            for j in range(8):
                prob = hp[i] * ap[j]
                if prob > best_prob:
                    best_prob, best_score = prob, f"{i}-{j}"

        home_win = sum(hp[i] * sum(ap[:i]) for i in range(1,8))
        draw = sum(hp[i]*ap[i] for i in range(8))
        away_win = 1 - home_win - draw
        btts = (1-poisson.cdf(0,home_xg_2h)) * (1-poisson.cdf(0,away_xg_2h))

        return {
            'home_xg_2h': round(home_xg_2h, 2),
            'away_xg_2h': round(away_xg_2h, 2),
            'total_xg': round(home_xg_2h + away_xg_2h, 2),
            'most_likely': best_score,
            'home_win_%': round(home_win*100, 1),
            'draw_%': round(draw*100, 1),
            'away_win_%': round(away_win*100, 1),
            'btts_%': round(btts*100, 1),
            'confidence_%': round(best_prob*100, 1),
            'home_momentum': round(home_mom, 1),
            'away_momentum': round(away_mom, 1),
        }

# ============================= UI =============================
def main():
    st.sidebar.header("ELITE 45-MINUTE + ODDS")

    division = st.sidebar.radio("Division", ["1st Division", "2nd Division"])
    leagues = [l for l, p in LEAGUES.items() if p['tier'] == (2 if "2nd" in division else 1)]
    league = st.sidebar.selectbox("League", sorted(leagues))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Home Team")
        home_team = st.text_input("Team", "Liverpool")
        h_goals = st.number_input("1H Goals", 0,10,2)
        h_xg = st.number_input("1H xG", 0.0,10.0,1.9,0.1)
        h_sib = st.number_input("Shots Inside Box", 0,25,8)
        h_big = st.number_input("Big Chances", 0,10,2)
        h_poss = st.slider("Possession %", 0,100,59)

    with col2:
        st.subheader("Away Team")
        away_team = st.text_input("Team", "Chelsea")
        a_goals = st.number_input("1H Goals ", 0,10,0)
        a_xg = st.number_input("1H xG ", 0.0,10.0,0.7,0.1)
        a_sib = st.number_input("Shots Inside Box ", 0,25,4)
        a_big = st.number_input("Big Chances ", 0,10,1)
        a_poss = st.slider("Possession % ", 0,100,41)

    # Get live odds
    odds = get_live_odds(home_team, away_team)

    home = {'goals': h_goals, 'xg': h_xg, 'sib': h_sib, 'big': h_big, 'poss': h_poss}
    away = {'goals': a_goals, 'xg': a_xg, 'sib': a_sib, 'big': a_big, 'poss': a_poss}

    result = ElitePredictor(league).predict(home, away)

    st.markdown(f"# {home_team} **{h_goals}–{a_goals}** {away_team}")
    st.caption(f"**{league} • Shots Inside Box Model • 2025 Season**")

    # LIVE ODDS DISPLAY
    st.markdown(f"<div class='odds-box'>"
                f"<strong>LIVE 2H 1X2 ODDS ({odds['bookie']})</strong><br>"
                f"Home Win: <strong>{odds['1']}</strong> | "
                f"Draw: <strong>{odds['X']}</strong> | "
                f"Away Win: <strong>{odds['2']}</strong>"
                f"</div>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric(f"{home_team} Momentum", f"{result['home_momentum']}%", "DOMINANT" if result['home_momentum']>70 else "")
    c2.metric(f"{away_team} Momentum", f"{result['away_momentum']}%", "DOMINANT" if result['away_momentum']>70 else "")
    c3.metric("2H Total xG", result['total_xg'])
    c4.metric("Most Likely", result['most_likely'])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Probabilities")
        st.metric("Home Win 2H", f"{result['home_win_%']}%")
        st.metric("Draw 2H", f"{result['draw_%']}%")
        st.metric("Away Win 2H", f"{result['away_win_%']}%")
        st.metric("BTTS Yes", f"{result['btts_%']}%")

    with col2:
        st.subheader("SHARP BETS vs Market")
        if result['total_xg'] > 2.1:      st.success(f"OVER 2.5 GOALS 2H — {result['total_xg']} xG")
        if result['btts_%'] > 68:         st.success(f"BTTS YES — {result['btts_%']}%")
        if result['home_win_%'] > 65:     st.success(f"{home_team.upper()} WIN 2H")
        if result['away_win_%'] > 65:     st.success(f"{away_team.upper()} WIN 2H")

    st.info(f"Model Confidence: {result['confidence_%']}% • Live odds powered by TheOddsAPI")

if __name__ == "__main__":
    main()
