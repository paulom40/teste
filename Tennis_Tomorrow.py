import pandas as pd
import numpy as np
import streamlit as st
import re, io, unicodedata, os
from datetime import datetime, timedelta
from difflib import get_close_matches
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
import time

st.set_page_config(page_title="🎾 Tennis Hedge Fund v5 Flashscore", layout="wide")

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
        if len(m) < 10:
            continue

        last20_idx = max(0, len(m)-20)
        last10_idx = max(0, len(m)-10)
        last20 = m.iloc[last20_idx:]
        last10 = m.iloc[last10_idx:]

        is_winner = (m["winner"]==p).values
        is_winner_l20 = (last20["winner"]==p).values
        is_winner_l10 = (last10["winner"]==p).values

        games = m["total_games"].values
        ou_avg = np.mean(games)
        ou_std = np.std(games) if len(games) > 1 else 0
        ou_last20 = np.mean(last20["total_games"].values) if len(last20) > 0 else ou_avg
        ou_volatility = ou_std / (ou_avg + 1)
        ou_over_pct = np.mean(games > 21.5)

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

    if len(X) < 10:
        return None, None

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base_model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
        random_state=42, n_jobs=-1, tree_method='hist'
    )
    
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
    model.fit(X_scaled, y)

    return model, scaler

def train_ou(df, stats):
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
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, tree_method='hist'
    )
    
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
    model.fit(X_scaled, y)

    return model, scaler_ou

# ================= BETTING =================
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
    
    # Calibration
    if win_prob < 0.65:
        calibrated_prob = win_prob * 0.95
    elif win_prob < 0.75:
        calibrated_prob = win_prob * 0.88
    elif win_prob < 0.85:
        calibrated_prob = win_prob * 0.85
    else:
        calibrated_prob = win_prob * 0.80
    
    calibrated_prob = max(0.50, min(0.95, calibrated_prob))
    
    odd = float(o1) if winner==p1 else float(o2)
    implied_prob = 1.0 / odd
    edge_value = calibrated_prob - implied_prob

    return {
        "Match": f"{raw1} vs {raw2}",
        "Pick": winner,
        "Type": "Winner",
        "Prob": calibrated_prob,
        "Odd": odd,
        "Edge": edge_value,
        "Stake": kelly(calibrated_prob, odd)
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
        odd = float(o_over)
    else:
        pick = "UNDER 21.5"
        pick_prob = 1 - prob_over
        odd = float(o_under)

    # Calibration
    if pick_prob < 0.65:
        calibrated_prob = pick_prob * 0.95
    elif pick_prob < 0.75:
        calibrated_prob = pick_prob * 0.88
    elif pick_prob < 0.85:
        calibrated_prob = pick_prob * 0.85
    else:
        calibrated_prob = pick_prob * 0.80
    
    calibrated_prob = max(0.50, min(0.95, calibrated_prob))

    implied_prob = 1.0 / odd
    edge_value = calibrated_prob - implied_prob

    return {
        "Match": f"{raw1} vs {raw2}",
        "Pick": pick,
        "Type": "O/U 21.5",
        "Prob": calibrated_prob,
        "Odd": odd,
        "Edge": edge_value,
        "Stake": kelly(calibrated_prob, odd)
    }

# ================= FLASHSCORE SCRAPER (ROBUST) =================
def scrape_flashscore_matches(days_ahead=0):
    """
    Robust Flashscore scraper with error handling
    """
    driver = None
    try:
        target_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%d.%m.%Y')
        
        # Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--start-maximized')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        
        # Initialize driver with webdriver-manager
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Navigate to Flashscore
        url = f"https://www.flashscore.com/tennis/{target_date}/"
        driver.get(url)
        
        # Wait and scroll to load content
        time.sleep(3)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        matches = []
        
        # Try multiple selectors
        selectors = [
            (By.CLASS_NAME, "event__match"),
            (By.CLASS_NAME, "eventRow"),
            (By.XPATH, "//div[@class='event__match']"),
            (By.XPATH, "//tr[@class='event__row']"),
        ]
        
        match_elements = []
        for by, selector in selectors:
            try:
                match_elements = driver.find_elements(by, selector)
                if match_elements:
                    break
            except:
                continue
        
        if not match_elements:
            return []
        
        for elem in match_elements[:30]:  # Limit to 30 matches
            try:
                # Get match info
                match_text = elem.text.strip()
                
                if not match_text or len(match_text) < 5:
                    continue
                
                # Parse match (format: "Player1\nPlayer2\nScore or Time")
                lines = match_text.split('\n')
                
                if len(lines) < 2:
                    continue
                
                p1 = lines[0].strip()
                p2 = lines[1].strip()
                
                # Validate player names
                if not p1 or not p2 or len(p1) < 2 or len(p2) < 2:
                    continue
                
                # Filter out invalid entries
                if any(x in p1.lower() for x in ['live', 'score', 'result', 'starts', 'cancelled']):
                    continue
                if any(x in p2.lower() for x in ['live', 'score', 'result', 'starts', 'cancelled']):
                    continue
                
                # Get tournament info for surface
                surface = "Hard"  # Default
                try:
                    tournament_elem = elem.find_element(By.CLASS_NAME, "event__title")
                    tournament_text = tournament_elem.text.lower()
                    
                    if "clay" in tournament_text or "roland" in tournament_text or "rg" in tournament_text:
                        surface = "Clay"
                    elif "grass" in tournament_text or "wimbledon" in tournament_text or "atp grass" in tournament_text:
                        surface = "Grass"
                except:
                    pass
                
                matches.append({
                    "p1": p1,
                    "p2": p2,
                    "surface": surface
                })
                
            except Exception as e:
                continue
        
        return matches[:20]  # Return first 20
        
    except TimeoutException:
        return []
    except WebDriverException as e:
        return []
    except Exception as e:
        return []
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

# ================= APP =================
st.title("⚡ Tennis Hedge Fund v5 - Flashscore")

file = st.file_uploader("Upload dataset", type=["xlsx"])

if file:
    with st.spinner("📊 Loading and training models... (10-30 seconds)"):
        df = load(file)
        stats = compute_stats(df)
        
        model_winner, scaler_w = train_winner(df, stats)
        model_ou, scaler_ou = train_ou(df, stats)

    if model_winner and model_ou:
        st.success("✅ Models trained!")
    else:
        st.warning("⚠️ Not enough data")
        st.stop()

    with st.sidebar:
        st.subheader("📅 Load Matches")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📅 TODAY", use_container_width=True, key="today_btn"):
                with st.spinner("Scraping Flashscore... (15-20 sec)"):
                    matches = scrape_flashscore_matches(days_ahead=0)
                    if matches:
                        st.session_state.matches = matches
                        st.session_state.current_date = "TODAY"
                        st.success(f"✅ Loaded {len(matches)} matches")
                    else:
                        st.warning("⚠️ No matches found")

        with col2:
            if st.button("📆 TOMORROW", use_container_width=True, key="tomorrow_btn"):
                with st.spinner("Scraping Flashscore... (15-20 sec)"):
                    matches = scrape_flashscore_matches(days_ahead=1)
                    if matches:
                        st.session_state.matches = matches
                        st.session_state.current_date = "TOMORROW"
                        st.success(f"✅ Loaded {len(matches)} matches")
                    else:
                        st.warning("⚠️ No matches found")

        with col3:
            if st.button("📅 +2 DAYS", use_container_width=True, key="twodays_btn"):
                with st.spinner("Scraping Flashscore... (15-20 sec)"):
                    matches = scrape_flashscore_matches(days_ahead=2)
                    if matches:
                        st.session_state.matches = matches
                        st.session_state.current_date = "IN 2 DAYS"
                        st.success(f"✅ Loaded {len(matches)} matches")
                    else:
                        st.warning("⚠️ No matches found")

    bankroll = st.number_input("💰 Bankroll (€)", value=1000, min_value=100)
    min_edge = st.slider("Min Edge (%)", 2, 10, 5) / 100
    min_prob = st.slider("Min Probability (%)", 55, 80, 67) / 100

    all_picks = []

    if "matches" in st.session_state and st.session_state.matches:
        date_label = st.session_state.get("current_date", "Matches")
        st.subheader(f"📊 Predictions - {date_label} ({len(st.session_state.matches)} matches)")
        
        cols = st.columns(2)
        col_idx = 0

        for idx, m in enumerate(st.session_state.matches):
            with cols[col_idx % 2]:
                st.subheader(f"{m['p1']} vs {m['p2']}", divider="gray")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    o1 = float(st.number_input("P1", 1.01, 10.0, 1.8, step=0.01, key=f"p1_{idx}"))
                with col2:
                    o2 = float(st.number_input("P2", 1.01, 10.0, 2.0, step=0.01, key=f"p2_{idx}"))
                with col3:
                    o_under = float(st.number_input("U21.5", 1.01, 5.0, 2.0, step=0.01, key=f"ou_{idx}"))
                with col4:
                    o_over = float(st.number_input("O21.5", 1.01, 5.0, 1.7, step=0.01, key=f"oo_{idx}"))

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
        st.subheader("📥 Export")
        df_exp = pd.DataFrame(all_picks)
        buffer = io.BytesIO()
        df_exp.to_excel(buffer, index=False)
        buffer.seek(0)
        st.download_button("📥 Download Picks", buffer, "picks_v5.xlsx")

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
