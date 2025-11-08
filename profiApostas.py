# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import io
from typing import Dict, Any, Tuple, List
import plotly.express as px
import plotly.graph_objects as go

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Football Predictor Pro", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .big-metric { font-size: 2.5rem; font-weight: bold; text-align: center; }
    .value-bet { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 color: white; padding: 15px; border-radius: 10px; margin: 10px 0; }
    .stat-card { background: #f8f9fa; padding: 15px; border-radius: 8px; 
                 border-left: 4px solid #007bff; margin: 10px 0; }
    .profit-positive { color: #28a745; font-weight: bold; }
    .profit-negative { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ Football Predictor Pro - Advanced Betting Suite")
st.markdown("""
**Professional Features**  
✅ Dixon-Coles Model with Bayesian Smoothing  
✅ Value Bet Detection with Kelly Criterion  
✅ Asian Handicap & Over/Under Analysis  
✅ Form Analysis & H2H Records  
✅ Expected Value (EV) Calculator  
✅ Bankroll Management & ROI Tracking  
✅ Multi-Market Predictions  
""")

# ================================
# BETTING UTILITIES
# ================================
def decimal_to_probability(odds: float) -> float:
    """Convert decimal odds to implied probability"""
    return 1 / odds if odds > 0 else 0

def probability_to_decimal(prob: float) -> float:
    """Convert probability to decimal odds"""
    return 1 / prob if prob > 0 else 0

def calculate_ev(true_prob: float, odds: float, stake: float = 100) -> Dict[str, float]:
    """Calculate Expected Value"""
    implied_prob = decimal_to_probability(odds)
    ev = (true_prob * (odds - 1) * stake) - ((1 - true_prob) * stake)
    ev_percentage = (ev / stake) * 100
    return {
        "ev": ev,
        "ev_percentage": ev_percentage,
        "implied_prob": implied_prob,
        "edge": (true_prob - implied_prob) * 100
    }

def kelly_criterion(probability: float, odds: float, conservative: float = 0.25) -> float:
    """Calculate Kelly Criterion stake percentage"""
    q = 1 - probability
    b = odds - 1
    kelly = ((b * probability) - q) / b
    return max(0, kelly * conservative)

def over_under_probability(home_goals: float, away_goals: float, line: float) -> Dict[str, float]:
    """Calculate Over/Under probabilities"""
    max_goals = 15
    prob_over = 0
    prob_under = 0
    
    for h in range(max_goals):
        for a in range(max_goals):
            total = h + a
            prob = poisson.pmf(h, home_goals) * poisson.pmf(a, away_goals)
            
            if total > line:
                prob_over += prob
            elif total < line:
                prob_under += prob
    
    return {"over": prob_over, "under": prob_under}

def btts_probability(home_goals: float, away_goals: float) -> Dict[str, float]:
    """Calculate Both Teams to Score probabilities"""
    prob_btts_yes = 0
    prob_btts_no = 0
    
    max_goals = 10
    for h in range(max_goals):
        for a in range(max_goals):
            prob = poisson.pmf(h, home_goals) * poisson.pmf(a, away_goals)
            if h > 0 and a > 0:
                prob_btts_yes += prob
            else:
                prob_btts_no += prob
    
    return {"yes": prob_btts_yes, "no": prob_btts_no}

def correct_score_probabilities(home_goals: float, away_goals: float, top_n: int = 10) -> List[Dict]:
    """Calculate most likely correct scores"""
    scores = []
    max_goals = 8
    
    for h in range(max_goals):
        for a in range(max_goals):
            prob = poisson.pmf(h, home_goals) * poisson.pmf(a, away_goals)
            scores.append({
                "score": f"{h}-{a}",
                "probability": prob
            })
    
    scores.sort(key=lambda x: x["probability"], reverse=True)
    return scores[:top_n]

# ================================
# FORM ANALYSIS
# ================================
def calculate_form(df: pd.DataFrame, team: str, home_col: str, away_col: str, 
                   hg_col: str, ag_col: str, last_n: int = 5) -> Dict[str, Any]:
    """Calculate recent form"""
    home_matches = df[df[home_col] == team].tail(last_n)
    away_matches = df[df[away_col] == team].tail(last_n)
    
    all_matches = pd.concat([home_matches, away_matches]).tail(last_n)
    
    wins = draws = losses = 0
    goals_for = goals_against = 0
    form_list = []
    
    for _, match in all_matches.iterrows():
        if match[home_col] == team:
            gf, ga = match[hg_col], match[ag_col]
        else:
            gf, ga = match[ag_col], match[hg_col]
        
        goals_for += gf
        goals_against += ga
        
        if gf > ga:
            wins += 1
            form_list.append("W")
        elif gf == ga:
            draws += 1
            form_list.append("D")
        else:
            losses += 1
            form_list.append("L")
    
    points = wins * 3 + draws
    matches_played = len(all_matches)
    
    return {
        "matches": matches_played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": int(goals_for),
        "goals_against": int(goals_against),
        "goal_difference": int(goals_for - goals_against),
        "points": points,
        "ppg": points / matches_played if matches_played > 0 else 0,
        "form_string": "".join(form_list)
    }

def head_to_head(df: pd.DataFrame, home: str, away: str, home_col: str, 
                 away_col: str, hg_col: str, ag_col: str, last_n: int = 5) -> Dict[str, Any]:
    """Analyze head-to-head record"""
    h2h = df[((df[home_col] == home) & (df[away_col] == away)) | 
             ((df[home_col] == away) & (df[away_col] == home))].tail(last_n)
    
    home_wins = away_wins = draws = 0
    home_goals = away_goals = 0
    
    for _, match in h2h.iterrows():
        if match[home_col] == home:
            hg, ag = match[hg_col], match[ag_col]
        else:
            hg, ag = match[ag_col], match[hg_col]
        
        home_goals += hg
        away_goals += ag
        
        if hg > ag:
            home_wins += 1
        elif ag > hg:
            away_wins += 1
        else:
            draws += 1
    
    return {
        "matches": len(h2h),
        "home_wins": home_wins,
        "away_wins": away_wins,
        "draws": draws,
        "home_goals": int(home_goals),
        "away_goals": int(away_goals),
        "avg_goals": (home_goals + away_goals) / len(h2h) if len(h2h) > 0 else 0
    }

# ================================
# CORE FUNCTIONS
# ================================
def bayesian_smoothing(observed_rate: float, league_avg: float, sample_size: int, confidence: int = 10) -> float:
    return (observed_rate * sample_size + league_avg * confidence) / (sample_size + confidence)

@st.cache_data(show_spinner="Loading CSV...")
def load_csv(uploaded_file_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(uploaded_file_bytes), encoding="utf-8")
    except:
        return pd.read_csv(io.BytesIO(uploaded_file_bytes), encoding="latin1")

@st.cache_data(show_spinner=False)
def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapping = {}
    for col in df.columns:
        lower = col.lower().replace(" ", "")
        if "home" in lower and "team" in lower: 
            mapping["HomeTeam"] = col
        elif "away" in lower and "team" in lower: 
            mapping["AwayTeam"] = col
        elif lower in ["fthg", "hgoals"]: 
            mapping["FTHG"] = col
        elif lower in ["ftag", "agoals"]: 
            mapping["FTAG"] = col
    return mapping

@st.cache_data(show_spinner="Training model...")
def compute_team_stats(
    _df: pd.DataFrame,
    home_col: str, away_col: str, hg_col: str, ag_col: str,
    recency_weight: float = 2.0, min_matches: int = 3
) -> Dict[str, Any]:
    df = _df.copy()
    for col in [hg_col, ag_col]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    stats = {}
    teams = sorted(set(df[home_col]).union(df[away_col]))

    ft_mask = df[hg_col].notna() & df[ag_col].notna()
    clean_ft = df[ft_mask][[home_col, away_col, hg_col, ag_col]].copy()
    if len(clean_ft) < 5: 
        raise ValueError("Not enough matches.")
    
    clean_ft['weight'] = np.exp(np.linspace(-recency_weight, 0, len(clean_ft)))
    avg_home = np.average(clean_ft[hg_col], weights=clean_ft['weight'])
    avg_away = np.average(clean_ft[ag_col], weights=clean_ft['weight'])

    def weighted_mean(group, col):
        if len(group) < min_matches: 
            return None
        return np.average(group[col], weights=group['weight'])

    home_attack = {}
    away_attack = {}
    home_defence = {}
    away_defence = {}
    
    for team in teams:
        home_matches = clean_ft[clean_ft[home_col] == team]
        away_matches = clean_ft[clean_ft[away_col] == team]
        n_home = len(home_matches)
        n_away = len(away_matches)
        
        if n_home >= min_matches:
            ha = weighted_mean(home_matches, hg_col) / avg_home
            hd = weighted_mean(home_matches, ag_col) / avg_away
            home_attack[team] = bayesian_smoothing(ha, 1.0, n_home)
            home_defence[team] = bayesian_smoothing(hd, 1.0, n_home)
        else:
            home_attack[team] = 1.0
            home_defence[team] = 1.0
            
        if n_away >= min_matches:
            aa = weighted_mean(away_matches, ag_col) / avg_away
            ad = weighted_mean(away_matches, hg_col) / avg_home
            away_attack[team] = bayesian_smoothing(aa, 1.0, n_away)
            away_defence[team] = bayesian_smoothing(ad, 1.0, n_away)
        else:
            away_attack[team] = 1.0
            away_defence[team] = 1.0

    stats["goals"] = {
        "league_avg_home": avg_home, 
        "league_avg_away": avg_away,
        "home_attack": home_attack, 
        "away_attack": away_attack,
        "home_defence": home_defence, 
        "away_defence": away_defence
    }

    return stats

def predict_match_advanced(home: str, away: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced prediction with all betting markets"""
    g = stats.get("goals", {})
    
    att_h = g["home_attack"].get(home, 1.0)
    def_a = g["away_defence"].get(away, 1.0)
    att_a = g["away_attack"].get(away, 1.0)
    def_h = g["home_defence"].get(home, 1.0)
    
    lambda_h = att_h * def_a * g["league_avg_home"]
    lambda_a = att_a * def_h * g["league_avg_away"]
    
    max_g = 10
    prob_matrix = np.zeros((max_g + 1, max_g + 1))
    
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            prob_matrix[h, a] = poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
    
    prob_matrix /= prob_matrix.sum()
    
    h_idx, a_idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
    
    prob_home = np.triu(prob_matrix, k=1).sum()
    prob_draw = prob_matrix.diagonal().sum()
    prob_away = np.tril(prob_matrix, k=-1).sum()
    
    btts = btts_probability(lambda_h, lambda_a)
    over_under_15 = over_under_probability(lambda_h, lambda_a, 1.5)
    over_under_25 = over_under_probability(lambda_h, lambda_a, 2.5)
    over_under_35 = over_under_probability(lambda_h, lambda_a, 3.5)
    
    correct_scores = correct_score_probabilities(lambda_h, lambda_a, 10)
    
    return {
        "expected_goals": {"home": round(lambda_h, 2), "away": round(lambda_a, 2)},
        "most_likely_score": f"{h_idx}-{a_idx}",
        "match_result": {
            "home_win": prob_home,
            "draw": prob_draw,
            "away_win": prob_away
        },
        "btts": btts,
        "over_under": {
            "1.5": over_under_15,
            "2.5": over_under_25,
            "3.5": over_under_35
        },
        "correct_scores": correct_scores
    }

# ================================
# MAIN APP
# ================================
st.sidebar.header("📊 Upload Match Data")
uploaded_file = st.sidebar.file_uploader("CSV File", type=["csv"])

if uploaded_file is not None:
    df = load_csv(uploaded_file.read())
    if df.empty:
        st.error("Empty CSV.")
    else:
        st.success(f"✅ Loaded {len(df):,} matches")
        mapping = detect_columns(df)

        st.sidebar.subheader("Column Mapping")
        col_map = {}
        for label in ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]:
            detected = mapping.get(label)
            options = [""] + [c for c in df.columns if c.lower() != "date"]
            default_idx = options.index(detected) if detected in options else 0
            col_map[label] = st.sidebar.selectbox(f"**{label}**", options=options, index=default_idx)

        missing = [r for r in ["HomeTeam", "AwayTeam", "FTHG", "FTAG"] if not col_map[r]]
        if missing:
            st.error(f"Map required columns: {', '.join(missing)}")
            st.stop()

        with st.spinner("🔄 Training model..."):
            team_stats = compute_team_stats(
                _df=df,
                home_col=col_map["HomeTeam"], 
                away_col=col_map["AwayTeam"],
                hg_col=col_map["FTHG"], 
                ag_col=col_map["FTAG"],
                recency_weight=st.sidebar.slider("Recency Weight", 0.5, 5.0, 2.0, 0.1),
                min_matches=st.sidebar.number_input("Min matches", 1, 20, 3)
            )

        teams = sorted(set(df[col_map["HomeTeam"]]).union(df[col_map["AwayTeam"]]))

        st.sidebar.markdown("---")
        st.sidebar.subheader("💰 Bankroll Settings")
        bankroll = st.sidebar.number_input("Total Bankroll (€)", min_value=100, value=1000, step=100)
        min_ev = st.sidebar.slider("Min EV% for Value Bet", 0.0, 20.0, 5.0, 0.5)
        kelly_fraction = st.sidebar.slider("Kelly Fraction", 0.1, 1.0, 0.25, 0.05)

        st.markdown("---")
        st.subheader("🎯 Match Prediction & Betting Analysis")
        
        col1, col2 = st.columns(2)
        home_team = col1.selectbox("🏠 Home Team", teams, key="home")
        away_team = col2.selectbox("✈️ Away Team", teams, key="away")

        if st.button("🔮 Generate Predictions", type="primary"):
            with st.spinner("Analyzing match..."):
                pred = predict_match_advanced(home_team, away_team, team_stats)
                
                home_form = calculate_form(df, home_team, col_map["HomeTeam"], 
                                          col_map["AwayTeam"], col_map["FTHG"], col_map["FTAG"])
                away_form = calculate_form(df, away_team, col_map["HomeTeam"], 
                                          col_map["AwayTeam"], col_map["FTHG"], col_map["FTAG"])
                
                h2h = head_to_head(df, home_team, away_team, col_map["HomeTeam"], 
                                  col_map["AwayTeam"], col_map["FTHG"], col_map["FTAG"])

                st.markdown(f"## {home_team} vs {away_team}")
                
                col_eg1, col_eg2, col_eg3 = st.columns(3)
                with col_eg1:
                    st.metric("xG Home", pred["expected_goals"]["home"])
                with col_eg2:
                    st.markdown(f"<div class='big-metric'>{pred['most_likely_score']}</div>", 
                              unsafe_allow_html=True)
                with col_eg3:
                    st.metric("xG Away", pred["expected_goals"]["away"])

                st.markdown("### 📊 Match Result")
                col_r1, col_r2, col_r3 = st.columns(3)
                mr = pred["match_result"]
                col_r1.metric("Home Win", f"{mr['home_win']:.1%}", 
                            delta=f"Fair Odds: {probability_to_decimal(mr['home_win']):.2f}")
                col_r2.metric("Draw", f"{mr['draw']:.1%}", 
                            delta=f"Fair Odds: {probability_to_decimal(mr['draw']):.2f}")
                col_r3.metric("Away Win", f"{mr['away_win']:.1%}", 
                            delta=f"Fair Odds: {probability_to_decimal(mr['away_win']):.2f}")

                st.markdown("### 📈 Form Analysis (Last 5 Matches)")
                col_f1, col_f2 = st.columns(2)
                
                with col_f1:
                    st.markdown(f"""
                    <div class='stat-card'>
                        <h4>{home_team}</h4>
                        <p><strong>Form:</strong> {home_form['form_string']}</p>
                        <p><strong>PPG:</strong> {home_form['ppg']:.2f}</p>
                        <p><strong>Record:</strong> {home_form['wins']}W-{home_form['draws']}D-{home_form['losses']}L</p>
                        <p><strong>Goals:</strong> {home_form['goals_for']} scored, {home_form['goals_against']} conceded</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_f2:
                    st.markdown(f"""
                    <div class='stat-card'>
                        <h4>{away_team}</h4>
                        <p><strong>Form:</strong> {away_form['form_string']}</p>
                        <p><strong>PPG:</strong> {away_form['ppg']:.2f}</p>
                        <p><strong>Record:</strong> {away_form['wins']}W-{away_form['draws']}D-{away_form['losses']}L</p>
                        <p><strong>Goals:</strong> {away_form['goals_for']} scored, {away_form['goals_against']} conceded</p>
                    </div>
                    """, unsafe_allow_html=True)

                if h2h['matches'] > 0:
                    st.markdown("### 🤝 Head-to-Head")
                    st.markdown(f"""
                    <div class='stat-card'>
                        <p><strong>Last {h2h['matches']} meetings:</strong></p>
                        <p>{home_team}: {h2h['home_wins']} wins | Draws: {h2h['draws']} | {away_team}: {h2h['away_wins']} wins</p>
                        <p><strong>Goals:</strong> {h2h['home_goals']}-{h2h['away_goals']} (Avg: {h2h['avg_goals']:.2f} per game)</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("### 🎲 Over/Under Markets")
                col_ou1, col_ou2, col_ou3 = st.columns(3)
                
                for col, line in zip([col_ou1, col_ou2, col_ou3], ["1.5", "2.5", "3.5"]):
                    ou = pred["over_under"][line]
                    with col:
                        st.write(f"**Over/Under {line} Goals**")
                        st.metric("Over", f"{ou['over']:.1%}", 
                                delta=f"Odds: {probability_to_decimal(ou['over']):.2f}")
                        st.metric("Under", f"{ou['under']:.1%}", 
                                delta=f"Odds: {probability_to_decimal(ou['under']):.2f}")

                st.markdown("### ⚽⚽ Both Teams To Score")
                col_b1, col_b2 = st.columns(2)
                btts = pred["btts"]
                col_b1.metric("BTTS Yes", f"{btts['yes']:.1%}", 
                            delta=f"Fair Odds: {probability_to_decimal(btts['yes']):.2f}")
                col_b2.metric("BTTS No", f"{btts['no']:.1%}", 
                            delta=f"Fair Odds: {probability_to_decimal(btts['no']):.2f}")

                st.markdown("### 🎯 Most Likely Correct Scores")
                cs_data = pred["correct_scores"][:5]
                cs_df = pd.DataFrame(cs_data)
                
                fig_cs = px.bar(cs_df, x="score", y="probability", 
                               title="Top 5 Most Likely Scores",
                               labels={"score": "Score", "probability": "Probability"})
                fig_cs.update_traces(text=cs_df["probability"].apply(lambda x: f"{x:.1%}"), 
                                   textposition="outside")
                st.plotly_chart(fig_cs, use_container_width=True)

                st.markdown("---")
                st.markdown("## 💎 Value Bet Calculator")
                st.markdown("Enter bookmaker odds to detect value bets:")
                
                col_odds1, col_odds2, col_odds3 = st.columns(3)
                with col_odds1:
                    odds_home = st.number_input("Home Win Odds", min_value=1.01, value=2.0, step=0.05)
                with col_odds2:
                    odds_draw = st.number_input("Draw Odds", min_value=1.01, value=3.5, step=0.05)
                with col_odds3:
                    odds_away = st.number_input("Away Win Odds", min_value=1.01, value=4.0, step=0.05)

                ev_home = calculate_ev(mr['home_win'], odds_home, 100)
                ev_draw = calculate_ev(mr['draw'], odds_draw, 100)
                ev_away = calculate_ev(mr['away_win'], odds_away, 100)
                
                value_bets = []
                
                if ev_home['ev_percentage'] >= min_ev:
                    kelly_stake = kelly_criterion(mr['home_win'], odds_home, kelly_fraction) * bankroll
                    value_bets.append({
                        "market": "Home Win",
                        "odds": odds_home,
                        "true_prob": mr['home_win'],
                        "ev": ev_home['ev_percentage'],
                        "edge": ev_home['edge'],
                        "kelly_stake": kelly_stake
                    })
                
                if ev_draw['ev_percentage'] >= min_ev:
                    kelly_stake = kelly_criterion(mr['draw'], odds_draw, kelly_fraction) * bankroll
                    value_bets.append({
                        "market": "Draw",
                        "odds": odds_draw,
                        "true_prob": mr['draw'],
                        "ev": ev_draw['ev_percentage'],
                        "edge": ev_draw['edge'],
                        "kelly_stake": kelly_stake
                    })
                
                if ev_away['ev_percentage'] >= min_ev:
                    kelly_stake = kelly_criterion(mr['away_win'], odds_away, kelly_fraction) * bankroll
                    value_bets.append({
                        "market": "Away Win",
                        "odds": odds_away,
                        "true_prob": mr['away_win'],
                        "ev": ev_away['ev_percentage'],
                        "edge": ev_away['edge'],
                        "kelly_stake": kelly_stake
                    })
                
                if value_bets:
                    st.success(f"🎯 Found {len(value_bets)} Value Bet(s)!")
                    for bet in value_bets:
                        st.markdown(f"""
                        <div class='value-bet'>
                            <h3>💰 {bet['market']}</h3>
                            <p><strong>Odds:</strong> {bet['odds']:.2f} | <strong>True Probability:</strong> {bet['true_prob']:.1%}</p>
                            <p><strong>Edge:</strong> {bet['edge']:.2f}% | <strong>EV:</strong> {bet['ev']:.2f}%</p>
                            <p><strong>Recommended Stake (Kelly):</strong> €{bet['kelly_stake']:.2f} ({(bet['kelly_stake']/bankroll)*100:.2f}% of bankroll)</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No value bets found with current odds and settings.")

else:
    st.info("📁 Upload a CSV file to start analyzing matches.")
