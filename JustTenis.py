import pandas as pd
import numpy as np
import streamlit as st
import re
import io
import requests
import unicodedata

from datetime import datetime, timedelta
from difflib import get_close_matches
from xgboost import XGBClassifier

st.set_page_config(page_title="🎾 Tennis Tipster PRO", layout="wide")

# =========================
# 🎨 UI
# =========================
st.markdown("""
<style>
.card {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    padding: 20px;
    border-radius: 15px;
    color: white;
    margin-bottom: 15px;
}
.high {color:#00ff88; font-weight:bold;}
.medium {color:#ffd700; font-weight:bold;}
.low {color:#ff6b6b; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# =========================
# NORMALIZAÇÃO NOMES
# =========================
def normalize_name(name):
    if pd.isna(name): return None
    name = str(name).lower().strip()
    name = unicodedata.normalize('NFKD', name)
    name = ''.join([c for c in name if not unicodedata.combining(c)])
    name = re.sub(r'[^a-z\s]', '', name)
    return re.sub(r'\s+', ' ', name)

def match_player(name, players):
    name = normalize_name(name)
    m = get_close_matches(name, players, n=1, cutoff=0.7)
    return m[0] if m else None

# =========================
# SURFACE
# =========================
def normalize_surface(s):
    if pd.isna(s): return "Hard"
    s = str(s).lower()
    if "clay" in s: return "Clay"
    if "grass" in s: return "Grass"
    return "Hard"

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    if "winner_name" in df: df.rename(columns={"winner_name":"winner"}, inplace=True)
    if "loser_name" in df: df.rename(columns={"loser_name":"loser"}, inplace=True)

    if "tourney_date" in df:
        df["date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["surface"] = df.get("surface", "Hard").apply(normalize_surface)
    df["total_games"] = df.get("t_games", 22)

    df["winner"] = df["winner"].apply(normalize_name)
    df["loser"] = df["loser"].apply(normalize_name)

    return df

# =========================
# STATS (WELO + LAST20)
# =========================
def compute_stats(df):
    stats = {}
    df = df.sort_values("date")

    players = set(df["winner"]) | set(df["loser"])

    for p in players:
        m = df[(df["winner"]==p) | (df["loser"]==p)]
        if len(m) < 10: continue

        wins = (m["winner"]==p).sum()
        last20 = m.tail(20)

        surface_wr = {}
        surface_welo = {}

        for s in ["Hard","Clay","Grass"]:
            sm = m[m["surface"]==s]
            if len(sm)==0:
                surface_wr[s]=0.5
                surface_welo[s]=1500
                continue

            l20 = sm.tail(20)
            wr = (l20["winner"]==p).mean()

            surface_wr[s]=wr
            surface_welo[s]=1500+(wr-0.5)*400

        stats[p]={
            "win_rate":wins/len(m),
            "last20":(last20["winner"]==p).mean(),
            "surface_wr":surface_wr,
            "surface_welo":surface_welo,
            "matches":len(m)
        }

    return stats

# =========================
# FEATURES
# =========================
def build_features(p1,p2,surf,stats):

    if p1 not in stats or p2 not in stats:
        return None

    if surf not in ["Hard","Clay","Grass"]:
        surf="Hard"

    s1,s2=stats[p1],stats[p2]

    return [
        s1["surface_welo"][surf]-s2["surface_welo"][surf],
        s1["surface_welo"][surf]-s2["surface_welo"][surf],
        s1["last20"]-s2["last20"],
        s1["surface_wr"][surf]-s2["surface_wr"][surf],
        s1["win_rate"]-s2["win_rate"],
        np.log(s1["matches"]+1)-np.log(s2["matches"]+1),
        (s1["surface_wr"][surf]+s2["surface_wr"][surf])/2,
        abs(s1["surface_welo"][surf]-s2["surface_welo"][surf])
    ]

# =========================
# TRAIN
# =========================
def train(df,stats):
    X,y=[],[]
    X_ou,y_ou=[],[]

    for _,r in df.iterrows():
        f1=build_features(r["winner"],r["loser"],r["surface"],stats)

        if f1:
            X.append(f1); y.append(1)
            X_ou.append(f1)
            y_ou.append(1 if r["total_games"]>21.5 else 0)

        f2=build_features(r["loser"],r["winner"],r["surface"],stats)
        if f2:
            X.append(f2); y.append(0)

    X,y=np.array(X),np.array(y)
    X_ou,y_ou=np.array(X_ou),np.array(y_ou)

    model=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.03)
    model_ou=XGBClassifier(n_estimators=250,max_depth=4,learning_rate=0.03)

    model.fit(X,y)
    model_ou.fit(X_ou,y_ou)

    return model,model_ou

# =========================
# PREDICT
# =========================
def predict(model,model_ou,stats,p1,p2,surf):

    players=list(stats.keys())

    raw1,raw2=p1,p2
    p1=match_player(p1,players)
    p2=match_player(p2,players)

    if not p1 or not p2:
        return None,raw1,raw2

    f=build_features(p1,p2,surf,stats)
    if f is None:
        return None,raw1,raw2

    prob=model.predict_proba([f])[0][1]
    ou=model_ou.predict_proba([f])[0][1]

    return {
        "winner":p1 if prob>0.5 else p2,
        "prob":max(prob,1-prob),
        "ou":"Over 21.5" if ou>0.5 else "Under 21.5",
        "ou_prob":max(ou,1-ou)
    },raw1,raw2

# =========================
# SCRAPER
# =========================
def get_matches(day=0):
    try:
        d=(datetime.utcnow()+timedelta(days=day)).strftime("%Y-%m-%d")
        url=f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{d}"
        r=requests.get(url)
        data=r.json()

        matches=[]
        for e in data.get("events",[]):
            if "WTA" in e["tournament"]["category"]["name"]: continue
            matches.append({
                "tournament":e["tournament"]["name"],
                "player1":e["homeTeam"]["name"],
                "player2":e["awayTeam"]["name"],
                "surface":"Hard"
            })
        return matches
    except:
        return []

# =========================
# APP
# =========================
st.title("🎾 Tennis Tipster PRO (65%+)")

file=st.file_uploader("Upload dataset ATP",type=["xlsx"])

if file:
    df=load_data(file)
    stats=compute_stats(df)
    model,model_ou=train(df,stats)

    st.success("Modelo pronto!")

    col1,col2=st.columns(2)
    with col1:
        if st.button("Hoje"):
            st.session_state.matches=get_matches(0)
    with col2:
        if st.button("Amanhã"):
            st.session_state.matches=get_matches(1)

    if "matches" in st.session_state:

        st.header("🔥 TOP PICKS")

        picks=[]

        for m in st.session_state.matches:
            pred,raw1,raw2=predict(model,model_ou,stats,m["player1"],m["player2"],m["surface"])

            if pred and pred["prob"]>=0.65:
                picks.append({
                    "match":f"{raw1} vs {raw2}",
                    "winner":pred["winner"],
                    "prob":pred["prob"],
                    "ou":pred["ou"],
                    "ou_prob":pred["ou_prob"]
                })

        picks=sorted(picks,key=lambda x:x["prob"],reverse=True)

        if not picks:
            st.warning("Sem picks hoje")

        for p in picks:
            cls="high" if p["prob"]>=0.7 else "medium"

            st.markdown(f"""
            <div class="card">
            <h3>{p['match']}</h3>
            🏆 <span class="{cls}">{p['winner']} ({p['prob']:.1%})</span><br>
            🎲 {p['ou']} ({p['ou_prob']:.1%})
            </div>
            """,unsafe_allow_html=True)

        if picks:
            df_exp=pd.DataFrame(picks)
            buffer=io.BytesIO()
            df_exp.to_excel(buffer,index=False)
            buffer.seek(0)

            st.download_button("📥 Export Picks",buffer,"picks.xlsx")
