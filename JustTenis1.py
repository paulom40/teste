import pandas as pd
import numpy as np
import streamlit as st
import re, io, requests, unicodedata, os
from datetime import datetime
from difflib import get_close_matches
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 Tennis Hedge Fund v3 FAST", layout="wide")

BET_LOG = "bet_log.csv"

# ================= INIT =================
if not os.path.exists(BET_LOG):
    pd.DataFrame(columns=[
        "date","match","pick","odd","prob","stake","result","profit"
    ]).to_csv(BET_LOG, index=False)

# ================= NORMALIZE =================
@st.cache_data
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
    df["total_games"] = df.get("t_games", 22)

    df["winner"] = df["winner"].apply(normalize)
    df["loser"] = df["loser"].apply(normalize)

    return df.dropna(subset=["winner","loser","date"])

# ================= STATS (OPTIMIZED) =================
@st.cache_data
def compute_stats(df):
    """Optimized stats computation"""
    stats = {}
    df = df.sort_values("date")
    
    players = set(df["winner"]) | set(df["loser"])
    
    for p in players:
        m = df[(df["winner"]==p)|(df["loser"]==p)]
        if len(m) < 10:  # Lowered from 15 to 10 for faster processing
            continue

        # Slice instead of tail for speed
        last20_idx = max(0, len(m)-20)
        last10_idx = max(0, len(m)-10)
        last20 = m.iloc[last20_idx:]
        last10 = m.iloc[last10_idx:]

        # Pre-calculate win indicators (vectorized)
        is_winner = (m["winner"]==p).values
        is_winner_l20 = (last20["winner"]==p).values
        is_winner_l10 = (last10["winner"]==p).values

        # ===== GAME STATS (VECTORIZED) =====
        games = m["total_games"].values
        ou_avg = np.mean(games)
        ou_std = np.std(games) if len(games) > 1 else 0
        ou_last20 = np.mean(last20["total_games"].values) if len(last20) > 0 else ou_avg
        ou_volatility = ou_std / (ou_avg + 1)
        ou_over_pct = np.mean(games > 21.5)

        # ===== SURFACE STATS (DICT COMPREHENSION) =====
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

            l20 = sm.iloc[max(0, len(sm)-20):]
            l10 = sm.iloc[max(0, len(sm)-10):]
            
            wr = np.mean((l20["winner"]==p).values)
            wr_recent = np.mean((l10["winner"]==p).values) if len(l10) > 0 else 0.5

            surf_wr[s] = wr
            surf_welo[s] = 1500 + (wr - 0.5) * 400
            surf_trend[s] = wr_recent - wr
            surf_ou_avg[s] = np.mean(sm["total_games"].values)

        stats[p] = {
            "win": np.mean(is_winner),
            "last20": np.mean(is_winner_l20),
            "last10": np.mean(is_winner_l10),
            "surf_wr": surf_wr,
            "surf_welo": surf_welo,
            "surf_trend": surf_trend,
            "matches": len(m),
            "momentum": np.mean(is_winner_l10) - np.mean(is_winner),
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
    if p1 not in stats or p2 not in stats: return None
    s1,s2 = stats[p1], stats[p2]

    return [
        (s1["ou_avg"] + s2["ou_avg"]) / 2,
        s1["ou_volatility"] + s2["ou_volatility"],
        s1["ou_last20"] + s2["ou_last20"],
        s1["ou_over_pct"] + s2["ou_over_pct"],
        (s1["surf_ou_avg"][s] + s2["surf_ou_avg"][s]) / 2,
        abs(s1["ou_volatility"] - s2["ou_volatility"]),
        max(s1["ou_std"], s2["ou_std"]),
    ]

# ================= TRAIN (OPTIMIZED) =================
def train_winner(df, stats):
    """Faster training with less data processing"""
    split = df["date"].quantile(0.8)
    train_df = df[df["date"] <= split]

    X, y = [], []
    
    # Vectorized iteration
    for _, r in train_df.iterrows():
        f = features_winner(r["winner"], r["loser"], r["surface"], stats)
        if f:
            X.append(f)
            y.append(1)

        f2 = features_winner(r["loser"], r["winner"], r["surface"], stats)
        if f2:
            X.append(f2)
            y.append(0)

    if len(X) < 10:
        return None, None

    X = np.array(X, dtype=np.float32)  # Use float32 to save memory
    y = np.array(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base_model = XGBClassifier(
        n_estimators=200,  # Reduced from 500
        max_depth=5,       # Reduced from 6
        learning_rate=0.05,  # Increased from 0.02 for faster convergence
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        random_state=42,
        n_jobs=-1,  # Use all cores
        tree_method='hist'  # Faster than default
    )
    
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)  # Reduced from 5
    model.fit(X_scaled, y)

    return model, scaler

def train_ou(df, stats):
    """Faster O/U training"""
    split = df["date"].quantile(0.8)
    train_df = df[df["date"] <= split]

    X, y = [], []

    for _, r in train_df.iterrows():
        f = features_ou(r["winner"], r["loser"], r["surface"], stats)
        if f:
            X.append(f)
            y.append(1 if r["total_games"] > 21.5 else 0)

    if len(X) < 10:
        return None, None

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    scaler_ou = StandardScaler()
    X_scaled = scaler_ou.fit_transform(X)

    base_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,  # Reduced from 5
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
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
    if not model_w or not scaler_w:
        return None
        
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
    if not model_ou or not scaler_ou:
        return None
        
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
    prob_over = model_ou.predict_proba(f_scaled)[0][1]

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

# ================= SCRAPER (OPTIMIZED) =================
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_matches():
    """Fast match fetching with timeout"""
    try:
        url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{datetime.utcnow().strftime('%Y-%m-%d')}"
        data = requests.get(url, timeout=3).json()  # 3 second timeout

        matches = []
        for e in data.get("events",[])[:20]:  # Limit to first 20 matches
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
st.title("⚡ Tennis Hedge Fund v3 FAST")

file = st.file_uploader("Upload dataset", type=["xlsx"])

if file:
    with st.spinner("📊 Loading and training models... (this may take 10-30 seconds)"):
        df = load(file)
        stats = compute_stats(df)
        
        # Train both models
        model_winner, scaler_w = train_winner(df, stats)
        model_ou, scaler_ou = train_ou(df, stats)

    if model_winner and model_ou:
        st.success("✅ Models trained! Ready for predictions.")
    else:
        st.warning("⚠️ Not enough data to train models")
        st.stop()

    # Sidebar for match loading
    with st.sidebar:
        if st.button("📅 Load Today's Matches (Fast API)", use_container_width=True):
            with st.spinner("Fetching today's matches... (3 sec timeout)"):
                st.session_state.matches = get_matches()
                if st.session_state.matches:
                    st.success(f"✅ Loaded {len(st.session_state.matches)} matches")
                else:
                    st.warning("No matches found or API timeout. Enter matches manually.")

    bankroll = st.number_input("💰 Bankroll (€)", value=1000, min_value=100)
    min_edge = st.slider("Min Edge (%)", 2, 10, 5) / 100
    min_prob = st.slider("Min Probability (%)", 55, 80, 67) / 100

    all_picks = []

    if "matches" in st.session_state:
        st.subheader(f"📊 Predictions ({len(st.session_state.matches)} matches)")
        
        # Use columns for faster rendering
        cols = st.columns(2)
        col_idx = 0

        for idx, m in enumerate(st.session_state.matches):
            with cols[col_idx % 2]:
                st.subheader(f"{m['p1']} vs {m['p2']}", divider="gray")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    o1 = st.number_input("P1", 1.01, 10.0, 1.8, key=f"p1_{idx}")
                with col2:
                    o2 = st.number_input("P2", 1.01, 10.0, 2.0, key=f"p2_{idx}")
                with col3:
                    o_under = st.number_input("U21.5", 1.01, 5.0, 2.0, key=f"ou_{idx}")
                with col4:
                    o_over = st.number_input("O21.5", 1.01, 5.0, 1.7, key=f"oo_{idx}")

                # Predictions
                pred_w = predict_winner(model_winner, scaler_w, stats, m["p1"], m["p2"], m["surface"], o1, o2)
                pred_ou = predict_ou(model_ou, scaler_ou, stats, m["p1"], m["p2"], m["surface"], o_under, o_over)

                if pred_w and pred_w["Prob"] > min_prob and pred_w["Edge"] > min_edge:
                    stake_w = pred_w["Stake"] * bankroll
                    pred_w["Stake €"] = stake_w
                    st.success(f"🏆 {pred_w['Pick']} @ {pred_w['Odd']:.2f}\nProb: {pred_w['Prob']:.1%} | Edge: {pred_w['Edge']:.1%}\nStake: €{stake_w:.2f}")
                    all_picks.append(pred_w)
                    if st.button(f"Log W", key=f"log_w_{idx}"):
                        log = pd.read_csv(BET_LOG)
                        log.loc[len(log)] = [datetime.now(), pred_w["Match"], pred_w["Pick"], pred_w["Odd"], pred_w["Prob"], stake_w, None, 0]
                        log.to_csv(BET_LOG, index=False)
                        st.success("✅ Logged")

                if pred_ou and pred_ou["Prob"] > min_prob and pred_ou["Edge"] > min_edge:
                    stake_ou = pred_ou["Stake"] * bankroll
                    pred_ou["Stake €"] = stake_ou
                    st.info(f"🎯 {pred_ou['Pick']} @ {pred_ou['Odd']:.2f}\nProb: {pred_ou['Prob']:.1%} | Edge: {pred_ou['Edge']:.1%}\nStake: €{stake_ou:.2f}")
                    all_picks.append(pred_ou)
                    if st.button(f"Log O/U", key=f"log_ou_{idx}"):
                        log = pd.read_csv(BET_LOG)
                        log.loc[len(log)] = [datetime.now(), pred_ou["Match"], pred_ou["Pick"], pred_ou["Odd"], pred_ou["Prob"], stake_ou, None, 0]
                        log.to_csv(BET_LOG, index=False)
                        st.success("✅ Logged")

                col_idx += 1

    if all_picks:
        st.subheader("📥 Export Picks")
        df_exp = pd.DataFrame(all_picks)
        buffer = io.BytesIO()
        df_exp.to_excel(buffer, index=False)
        buffer.seek(0)
        st.download_button("📥 Download Picks", buffer, "picks_v3.xlsx")

# ================= DASHBOARD =================
st.header("📊 Dashboard")

log = pd.read_csv(BET_LOG)

if len(log) > 0:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Profit", f"€{log['profit'].sum():.2f}")
    col2.metric("ROI", f"{(log['profit'].sum() / log['stake'].sum() * 100):.1f}%")
    col3.metric("Winrate", f"{(len(log[log['result']==1]) / len(log) * 100):.1f}%")
    col4.metric("Bets", len(log))

    st.dataframe(log, use_container_width=True)

    buffer2 = io.BytesIO()
    log.to_excel(buffer2, index=False)
    buffer2.seek(0)
    st.download_button("📥 Download Log", buffer2, "bet_log.xlsx")

    st.subheader("Update Result")
    idx = st.number_input("Index", 0, len(log)-1, 0)
    res = st.selectbox("Result", [1, 0])
    if st.button("Update"):
        profit = log.loc[idx, "stake"] * (log.loc[idx, "odd"] - 1) if res == 1 else -log.loc[idx, "stake"]
        log.loc[idx, "result"] = res
        log.loc[idx, "profit"] = profit
        log.to_csv(BET_LOG, index=False)
        st.success("✅ Updated")
