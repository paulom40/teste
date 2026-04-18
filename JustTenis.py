import pandas as pd
import numpy as np
import streamlit as st
import re, io, requests, unicodedata, os
from datetime import datetime
from difflib import get_close_matches
from xgboost import XGBClassifier

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

        surf_wr = {}
        surf_welo = {}

        for s in ["Hard","Clay","Grass"]:
            sm = m[m["surface"]==s]
            if len(sm) == 0:
                surf_wr[s] = 0.5
                surf_welo[s] = 1500
                continue

            l20 = sm.tail(20)
            wr = (l20["winner"]==p).mean()

            surf_wr[s] = wr
            surf_welo[s] = 1500 + (wr - 0.5)*400

        stats[p] = {
            "win": (m["winner"]==p).mean(),
            "last20": (last20["winner"]==p).mean(),
            "surf_wr": surf_wr,
            "surf_welo": surf_welo,
            "matches": len(m)
        }

    return stats

# ================= FEATURES =================
def features(p1,p2,s,stats):
    if p1 not in stats or p2 not in stats: return None
    s1,s2 = stats[p1], stats[p2]

    return [
        s1["surf_welo"][s] - s2["surf_welo"][s],
        s1["last20"] - s2["last20"],
        s1["surf_wr"][s] - s2["surf_wr"][s],
        s1["win"] - s2["win"],
        np.log(s1["matches"]+1) - np.log(s2["matches"]+1),
        abs(s1["surf_welo"][s] - s2["surf_welo"][s])
    ]

# ================= TRAIN =================
def train(df,stats):
    split = df["date"].quantile(0.8)
    train_df = df[df["date"] <= split]

    X,y = [],[]

    for _,r in train_df.iterrows():
        f = features(r["winner"], r["loser"], r["surface"], stats)
        if f:
            X.append(f); y.append(1)

        f2 = features(r["loser"], r["winner"], r["surface"], stats)
        if f2:
            X.append(f2); y.append(0)

    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03)
    model.fit(X,y)

    return model

# ================= BETTING =================
def edge(p,o): return p - (1/o)

def kelly(p,o):
    b = o - 1
    q = 1 - p
    return max((b*p - q)/b, 0) * 0.5

# ================= PREDICT =================
def predict(model,stats,p1,p2,s,o1,o2):
    players = list(stats.keys())

    raw1, raw2 = p1, p2
    p1 = match_name(p1, players)
    p2 = match_name(p2, players)

    if not p1 or not p2: return None

    f = features(p1,p2,s,stats)
    if not f: return None

    prob = model.predict_proba([f])[0][1]

    winner = p1 if prob > 0.5 else p2
    win_prob = max(prob, 1-prob)

    odd = o1 if winner==p1 else o2

    return {
        "Match": f"{raw1} vs {raw2}",
        "Pick": winner,
        "Prob": win_prob,
        "Odd": odd,
        "Edge": edge(win_prob, odd),
        "Stake": kelly(win_prob, odd)
    }

# ================= SCRAPER =================
def get_matches():
    try:
        url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{datetime.utcnow().strftime('%Y-%m-%d')}"
        data = requests.get(url).json()

        matches = []
        for e in data.get("events",[]):
            if "WTA" in e["tournament"]["category"]["name"]: continue
            matches.append({
                "p1": e["homeTeam"]["name"],
                "p2": e["awayTeam"]["name"],
                "surface": "Hard"
            })
        return matches
    except:
        return []

# ================= APP =================
st.title("🎾 Tennis Hedge Fund PRO")

file = st.file_uploader("Upload dataset", type=["xlsx"])

if file:
    df = load(file)
    stats = compute_stats(df)
    model = train(df,stats)

    st.success("Model trained")

    if st.button("📅 Load Today Matches"):
        st.session_state.matches = get_matches()

    bankroll = st.number_input("💰 Bankroll (€)", value=1000)

    picks = []

    if "matches" in st.session_state:

        for m in st.session_state.matches:

            st.subheader(f"{m['p1']} vs {m['p2']}")

            col1,col2 = st.columns(2)
            with col1:
                o1 = st.number_input("Odd P1",1.01,10.0,1.8,key=m['p1'])
            with col2:
                o2 = st.number_input("Odd P2",1.01,10.0,2.0,key=m['p2'])

            pred = predict(model,stats,m["p1"],m["p2"],m["surface"],o1,o2)

            if pred and pred["Prob"]>0.67 and pred["Edge"]>0.05:

                stake = pred["Stake"] * bankroll
                pred["Stake €"] = stake

                st.success(f"""
                🏆 {pred['Pick']} @ {pred['Odd']}
                📊 Prob: {pred['Prob']:.2%}
                💰 Edge: {pred['Edge']:.2%}
                🎯 Stake: €{stake:.2f}
                """)

                picks.append(pred)

                if st.button(f"Log {pred['Match']}"):
                    log = pd.read_csv(BET_LOG)
                    log.loc[len(log)] = [
                        datetime.now(),
                        pred["Match"],
                        pred["Pick"],
                        pred["Odd"],
                        pred["Prob"],
                        stake,
                        None,
                        0
                    ]
                    log.to_csv(BET_LOG,index=False)
                    st.success("Logged")

    # ================= EXPORT PICKS =================
    if picks:
        st.subheader("📥 Export Picks")
        df_exp = pd.DataFrame(picks)
        buffer = io.BytesIO()
        df_exp.to_excel(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            "📥 Download Picks Excel",
            buffer,
            "picks.xlsx"
        )

# ================= DASHBOARD =================
st.header("📊 Performance Dashboard")

log = pd.read_csv(BET_LOG)

if len(log) > 0:

    total_profit = log["profit"].sum()
    total_stake = log["stake"].sum()
    roi = total_profit / total_stake if total_stake>0 else 0
    winrate = len(log[log["result"]==1]) / len(log) if len(log)>0 else 0

    st.metric("Profit", f"{total_profit:.2f}€")
    st.metric("ROI", f"{roi:.2%}")
    st.metric("Winrate", f"{winrate:.2%}")
    st.metric("Bets", len(log))

    st.dataframe(log)

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

    idx = st.number_input("Index",0,len(log)-1,0)
    res = st.selectbox("Result",[1,0])

    if st.button("Update Bet"):
        if res == 1:
            profit = log.loc[idx,"stake"]*(log.loc[idx,"odd"]-1)
        else:
            profit = -log.loc[idx,"stake"]

        log.loc[idx,"result"] = res
        log.loc[idx,"profit"] = profit
        log.to_csv(BET_LOG,index=False)

        st.success("Updated")
