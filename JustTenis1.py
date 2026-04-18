import pandas as pd
import numpy as np
import streamlit as st
import re, io, requests, unicodedata, os
from datetime import datetime
from difflib import get_close_matches
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="🎾 Tennis Hedge Fund", layout="wide")

BET_LOG = "bet_log.csv"

# ================= INIT =================
if not os.path.exists(BET_LOG):
    pd.DataFrame(columns=[
        "date","match","pick","odd","prob","stake","result","profit"
    ]).to_csv(BET_LOG, index=False)

# ================= NORMALIZE =================
def normalize(x):
    if pd.isna(x): return None
    x = str(x).lower()
    x = unicodedata.normalize('NFKD', x)
    x = ''.join(c for c in x if not unicodedata.combining(c))
    return re.sub(r'[^a-z ]','', x)

def match_name(name, players):
    name = normalize(name)
    m = get_close_matches(name, players, n=1, cutoff=0.7)
    return m[0] if m else None

# ================= LOAD =================
@st.cache_data
def load(file):
    df = pd.read_excel(file)
    df.columns = [c.lower().replace(" ","_") for c in df.columns]

    df.rename(columns={
        "winner_name":"winner",
        "loser_name":"loser"
    }, inplace=True, errors="ignore")

    df["date"] = pd.to_datetime(df.get("tourney_date", df.get("date")), errors="coerce")
    df["surface"] = df.get("surface","Hard").fillna("Hard")
    df["total_games"] = df.get("t_games",22)

    df["winner"] = df["winner"].apply(normalize)
    df["loser"] = df["loser"].apply(normalize)

    return df.dropna(subset=["winner","loser","date"])

# ================= STATS =================
def compute_stats(df):
    stats = {}
    df = df.sort_values("date")

    players = set(df["winner"]) | set(df["loser"])

    for p in players:
        m = df[(df["winner"]==p)|(df["loser"]==p)]
        if len(m) < 15: continue

        last20 = m.tail(20)
        last10 = m.tail(10)

        # ===== OVER/UNDER STATS =====
        ou_avg = m["total_games"].mean()
        ou_std = m["total_games"].std() if len(m) > 1 else 0
        ou_last20 = m.tail(20)["total_games"].mean() if len(m) >= 20 else ou_avg
        
        # Games per set (3-set format: 6 games/set minimum)
        ou_volatility = ou_std / (ou_avg + 1)  # How much variation
        
        # % of matches over 21.5
        ou_over_pct = (m["total_games"] > 21.5).mean()

        surf_wr = {}
        surf_welo = {}
        surf_trend = {}
        surf_ou_avg = {}

        for s in ["Hard","Clay","Grass"]:
            sm = m[m["surface"]==s]
            if len(sm) == 0:
                surf_wr[s] = 0.5
                surf_welo[s] = 1500
                surf_trend[s] = 0
                surf_ou_avg[s] = 21.5
                continue

            l20 = sm.tail(20)
            l10 = sm.tail(10)
            wr = (l20["winner"]==p).mean()
            wr_recent = (l10["winner"]==p).mean() if len(l10) > 0 else 0.5

            surf_wr[s] = wr
            surf_welo[s] = 1500 + (wr - 0.5)*400
            surf_trend[s] = wr_recent - wr
            surf_ou_avg[s] = sm["total_games"].mean()

        stats[p] = {
            "win": (m["winner"]==p).mean(),
            "last20": (last20["winner"]==p).mean(),
            "last10": (last10["winner"]==p).mean(),
            "surf_wr": surf_wr,
            "surf_welo": surf_welo,
            "surf_trend": surf_trend,
            "matches": len(m),
            "momentum": (last10["winner"]==p).mean() - (m["winner"]==p).mean(),
            
            # ===== OVER/UNDER FEATURES =====
            "ou_avg": ou_avg,
            "ou_std": ou_std,
            "ou_last20": ou_last20,
            "ou_volatility": ou_volatility,
            "ou_over_pct": ou_over_pct,
            "surf_ou_avg": surf_ou_avg,
        }

    return stats

# ================= FEATURES =================
def features_winner(p1,p2,s,stats):
    """Features for winner prediction"""
    if p1 not in stats or p2 not in stats: return None
    s1,s2 = stats[p1], stats[p2]

    return [
        s1["surf_welo"][s] - s2["surf_welo"][s],
        s1["last20"] - s2["last20"],
        s1["last10"] - s2["last10"],
        s1["surf_wr"][s] - s2["surf_wr"][s],
        s1["win"] - s2["win"],
        np.log(s1["matches"]+1) - np.log(s2["matches"]+1),
        abs(s1["surf_welo"][s] - s2["surf_welo"][s]),
        s1["momentum"] - s2["momentum"],
        s1["surf_trend"][s] - s2["surf_trend"][s],
    ]

def features_ou(p1,p2,s,stats):
    """Features for over/under 21.5 prediction"""
    if p1 not in stats or p2 not in stats: return None
    s1,s2 = stats[p1], stats[p2]

    return [
        (s1["ou_avg"] + s2["ou_avg"]) / 2,           # Average games both players
        s1["ou_volatility"] + s2["ou_volatility"],   # Combined volatility (higher = more games)
        s1["ou_last20"] + s2["ou_last20"],           # Recent games trend
        s1["ou_over_pct"] + s2["ou_over_pct"],       # % of matches going over
        (s1["surf_ou_avg"][s] + s2["surf_ou_avg"][s]) / 2,  # Surface-specific games
        abs(s1["ou_volatility"] - s2["ou_volatility"]),     # Volatility mismatch (defensive vs offensive)
        max(s1["ou_std"], s2["ou_std"]),             # Max volatility (one player causes variance)
    ]

# ================= TRAIN =================
def train_winner(df, stats):
    split = df["date"].quantile(0.8)
    train_df = df[df["date"] <= split]

    X, y = [], []

    for _, r in train_df.iterrows():
        f = features_winner(r["winner"], r["loser"], r["surface"], stats)
        if f:
            X.append(f)
            y.append(1)

        f2 = features_winner(r["loser"], r["winner"], r["surface"], stats)
        if f2:
            X.append(f2)
            y.append(0)

    X = np.array(X)
    y = np.array(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base_model = XGBClassifier(
        n_estimators=500, 
        max_depth=6, 
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        random_state=42
    )
    
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    model.fit(X_scaled, y)

    return model, scaler

def train_ou(df, stats):
    """Train over/under 21.5 model"""
    split = df["date"].quantile(0.8)
    train_df = df[df["date"] <= split]

    X, y = [], []

    for _, r in train_df.iterrows():
        f = features_ou(r["winner"], r["loser"], r["surface"], stats)
        if f:
            X.append(f)
            # y=1 if over 21.5, y=0 if under 21.5
            y.append(1 if r["total_games"] > 21.5 else 0)

    X = np.array(X)
    y = np.array(y)
    
    scaler_ou = StandardScaler()
    X_scaled = scaler_ou.fit_transform(X)

    base_model = XGBClassifier(
        n_estimators=500, 
        max_depth=5, 
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    model.fit(X_scaled, y)

    return model, scaler_ou

# ================= BETTING =================
def edge(p, o): 
    return p - (1/o)

def kelly(p, o):
    b = o - 1
    q = 1 - p
    return max((b*p - q)/b, 0) * 0.25

# ================= PREDICT =================
def predict_winner(model_w, scaler_w, stats, p1, p2, s, o1, o2):
    players = list(stats.keys())

    raw1, raw2 = p1, p2
    p1 = match_name(p1, players)
    p2 = match_name(p2, players)

    if not p1 or not p2: 
        return None

    f = features_winner(p1, p2, s, stats)
    if not f: 
        return None

    f_scaled = scaler_w.transform([f])
    prob = model_w.predict_proba(f_scaled)[0][1]

    winner = p1 if prob > 0.5 else p2
    win_prob = max(prob, 1-prob)

    odd = o1 if winner==p1 else o2

    return {
        "Match": f"{raw1} vs {raw2}",
        "Pick": winner,
        "Type": "Winner",
        "Prob": win_prob,
        "Odd": odd,
        "Edge": edge(win_prob, odd),
        "Stake": kelly(win_prob, odd)
    }

def predict_ou(model_ou, scaler_ou, stats, p1, p2, s, o_under, o_over):
    """Predict over/under 21.5 total games"""
    players = list(stats.keys())

    raw1, raw2 = p1, p2
    p1 = match_name(p1, players)
    p2 = match_name(p2, players)

    if not p1 or not p2: 
        return None

    f = features_ou(p1, p2, s, stats)
    if not f: 
        return None

    f_scaled = scaler_ou.transform([f])
    prob_over = model_ou.predict_proba(f_scaled)[0][1]  # Probability of OVER 21.5

    # Decide based on which has better edge
    if prob_over > 0.5:
        pick = "OVER 21.5"
        pick_prob = prob_over
        odd = o_over
    else:
        pick = "UNDER 21.5"
        pick_prob = 1 - prob_over
        odd = o_under

    return {
        "Match": f"{raw1} vs {raw2}",
        "Pick": pick,
        "Type": "O/U 21.5",
        "Prob": pick_prob,
        "Odd": odd,
        "Edge": edge(pick_prob, odd),
        "Stake": kelly(pick_prob, odd)
    }

# ================= SCRAPER =================
def get_matches():
    try:
        url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{datetime.utcnow().strftime('%Y-%m-%d')}"
        data = requests.get(url, timeout=5).json()

        matches = []
        for e in data.get("events",[]):
            if "WTA" in e["tournament"]["category"]["name"]: 
                continue
            matches.append({
                "p1": e["homeTeam"]["name"],
                "p2": e["awayTeam"]["name"],
                "surface": "Hard"
            })
        return matches
    except:
        return []

# ================= APP =================
st.title("🎾 Tennis Hedge Fund PRO v3 - Winner & O/U Predictions")

file = st.file_uploader("Upload dataset", type=["xlsx"])

if file:
    df = load(file)
    stats = compute_stats(df)
    
    # Train both models
    model_winner, scaler_w = train_winner(df, stats)
    model_ou, scaler_ou = train_ou(df, stats)

    st.success("✅ Models trained: Winner + Over/Under 21.5")

    if st.button("📅 Load Today Matches"):
        st.session_state.matches = get_matches()

    bankroll = st.number_input("💰 Bankroll (€)", value=1000)
    
    min_edge = st.slider("Min Edge (%)", 2, 10, 5) / 100
    min_prob = st.slider("Min Probability (%)", 55, 80, 67) / 100

    all_picks = []

    if "matches" in st.session_state:

        for idx, m in enumerate(st.session_state.matches):

            st.subheader(f"📊 {m['p1']} vs {m['p2']}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                o1 = st.number_input("Odd P1", 1.01, 10.0, 1.8, key=f"p1_{idx}")
            with col2:
                o2 = st.number_input("Odd P2", 1.01, 10.0, 2.0, key=f"p2_{idx}")
            with col3:
                o_under = st.number_input("Odd U21.5", 1.01, 5.0, 2.0, key=f"ou_{idx}")
            with col4:
                o_over = st.number_input("Odd O21.5", 1.01, 5.0, 1.7, key=f"oo_{idx}")

            # ===== WINNER PREDICTION =====
            pred_winner = predict_winner(model_winner, scaler_w, stats, m["p1"], m["p2"], m["surface"], o1, o2)

            if pred_winner and pred_winner["Prob"] > min_prob and pred_winner["Edge"] > min_edge:
                stake_w = pred_winner["Stake"] * bankroll
                pred_winner["Stake €"] = stake_w

                st.success(f"""
                🏆 **{pred_winner['Pick']}** @ {pred_winner['Odd']:.2f}
                📊 Prob: {pred_winner['Prob']:.1%} | Edge: {pred_winner['Edge']:.1%}
                💰 Stake: €{stake_w:.2f}
                """)

                all_picks.append(pred_winner)

                if st.button(f"Log Winner {pred_winner['Match']}", key=f"log_w_{idx}"):
                    log = pd.read_csv(BET_LOG)
                    log.loc[len(log)] = [
                        datetime.now(),
                        pred_winner["Match"],
                        pred_winner["Pick"],
                        pred_winner["Odd"],
                        pred_winner["Prob"],
                        stake_w,
                        None,
                        0
                    ]
                    log.to_csv(BET_LOG, index=False)
                    st.success("✅ Winner bet logged")

            # ===== OVER/UNDER PREDICTION =====
            pred_ou = predict_ou(model_ou, scaler_ou, stats, m["p1"], m["p2"], m["surface"], o_under, o_over)

            if pred_ou and pred_ou["Prob"] > min_prob and pred_ou["Edge"] > min_edge:
                stake_ou = pred_ou["Stake"] * bankroll
                pred_ou["Stake €"] = stake_ou

                st.info(f"""
                🎯 **{pred_ou['Pick']}** @ {pred_ou['Odd']:.2f}
                📊 Prob: {pred_ou['Prob']:.1%} | Edge: {pred_ou['Edge']:.1%}
                💰 Stake: €{stake_ou:.2f}
                """)

                all_picks.append(pred_ou)

                if st.button(f"Log O/U {pred_ou['Match']}", key=f"log_ou_{idx}"):
                    log = pd.read_csv(BET_LOG)
                    log.loc[len(log)] = [
                        datetime.now(),
                        pred_ou["Match"],
                        pred_ou["Pick"],
                        pred_ou["Odd"],
                        pred_ou["Prob"],
                        stake_ou,
                        None,
                        0
                    ]
                    log.to_csv(BET_LOG, index=False)
                    st.success("✅ O/U bet logged")

    # ================= EXPORT PICKS =================
    if all_picks:
        st.subheader("📥 Export All Picks")
        df_exp = pd.DataFrame(all_picks)
        buffer = io.BytesIO()
        df_exp.to_excel(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            "📥 Download All Picks Excel",
            buffer,
            "picks_v3.xlsx"
        )

# ================= DASHBOARD =================
st.header("📊 Performance Dashboard")

log = pd.read_csv(BET_LOG)

if len(log) > 0:

    total_profit = log["profit"].sum()
    total_stake = log["stake"].sum()
    roi = total_profit / total_stake if total_stake > 0 else 0
    winrate = len(log[log["result"]==1]) / len(log) if len(log) > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Profit", f"€{total_profit:.2f}")
    col2.metric("ROI", f"{roi:.2%}")
    col3.metric("Winrate", f"{winrate:.2%}")
    col4.metric("Bets", len(log))

    # Separate by type
    winners = log[log["pick"].str.contains("vs", regex=False, na=False)]
    ou_bets = log[log["pick"].str.contains("OVER|UNDER", regex=True, na=False)]

    if len(winners) > 0:
        st.subheader("🏆 Winner Bets")
        st.dataframe(winners, use_container_width=True)

    if len(ou_bets) > 0:
        st.subheader("🎯 Over/Under Bets")
        st.dataframe(ou_bets, use_container_width=True)

    # ================= EXPORT LOG =================
    st.subheader("📥 Export Bet Log")
    buffer2 = io.BytesIO()
    log.to_excel(buffer2, index=False)
    buffer2.seek(0)

    st.download_button(
        "📥 Download Bet Log Excel",
        buffer2,
        "bet_log.xlsx"
    )

    st.subheader("Update Result")

    idx = st.number_input("Index", 0, len(log)-1, 0)
    res = st.selectbox("Result", [1, 0])

    if st.button("Update Bet"):
        if res == 1:
            profit = log.loc[idx, "stake"] * (log.loc[idx, "odd"] - 1)
        else:
            profit = -log.loc[idx, "stake"]

        log.loc[idx, "result"] = res
        log.loc[idx, "profit"] = profit
        log.to_csv(BET_LOG, index=False)

        st.success("✅ Updated")
