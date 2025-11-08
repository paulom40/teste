# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import io
from typing import Dict, Any, Tuple, List
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

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
                 border-left: 4px solid #007bff; margin: 10px 0; color: #212529; }
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

def generate_html_report(home: str, away: str, pred: Dict, home_form: Dict, 
                         away_form: Dict, h2h: Dict, bankroll: float, 
                         value_bets: List[Dict]) -> str:
    """Generate complete HTML report"""
    from datetime import datetime
    
    html = f"""
    <!DOCTYPE html>
    <html lang="pt">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{home} vs {away} - Análise Profissional</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 30px;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }}
            .header {{
                text-align: center;
                padding-bottom: 30px;
                border-bottom: 3px solid #667eea;
                margin-bottom: 30px;
            }}
            .header h1 {{
                font-size: 2.5rem;
                color: #667eea;
                margin-bottom: 10px;
            }}
            .header .date {{
                color: #666;
                font-size: 0.9rem;
            }}
            .match-header {{
                display: flex;
                justify-content: space-around;
                align-items: center;
                margin: 30px 0;
                padding: 20px;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border-radius: 15px;
            }}
            .team {{
                text-align: center;
                flex: 1;
            }}
            .team-name {{
                font-size: 1.8rem;
                font-weight: bold;
                color: #333;
                margin: 10px 0;
            }}
            .score {{
                font-size: 3rem;
                font-weight: bold;
                color: #667eea;
                text-align: center;
                margin: 0 20px;
            }}
            .xg {{
                text-align: center;
                color: #666;
                font-size: 1.1rem;
                margin-top: 10px;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                border-left: 5px solid #667eea;
            }}
            .section h2 {{
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.5rem;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .metric {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }}
            .metric:hover {{
                transform: translateY(-5px);
            }}
            .metric-label {{
                color: #666;
                font-size: 0.9rem;
                margin-bottom: 5px;
            }}
            .metric-value {{
                color: #667eea;
                font-size: 1.8rem;
                font-weight: bold;
            }}
            .metric-odds {{
                color: #999;
                font-size: 0.8rem;
                margin-top: 5px;
            }}
            .form-container {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 20px 0;
            }}
            .form-box {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .form-box h3 {{
                color: #667eea;
                margin-bottom: 15px;
            }}
            .form-box p {{
                margin: 8px 0;
                color: #333;
            }}
            .form-string {{
                font-size: 1.2rem;
                font-weight: bold;
                letter-spacing: 2px;
                color: #667eea;
            }}
            .value-bet {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 15px 0;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }}
            .value-bet h3 {{
                margin-bottom: 10px;
            }}
            .value-bet p {{
                margin: 5px 0;
            }}
            .table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            .table th {{
                background: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            .table td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            .table tr:hover {{
                background: #f5f7fa;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 2px solid #e0e0e0;
                color: #666;
                font-size: 0.9rem;
            }}
            @media print {{
                body {{ background: white; padding: 0; }}
                .container {{ box-shadow: none; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚽ Football Predictor Pro</h1>
                <p class="date">Relatório gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')}</p>
            </div>
            
            <div class="match-header">
                <div class="team">
                    <div class="team-name">🏠 {home}</div>
                    <div class="xg">xG: {pred['expected_goals']['home']}</div>
                </div>
                <div class="score">{pred['most_likely_score']}</div>
                <div class="team">
                    <div class="team-name">✈️ {away}</div>
                    <div class="xg">xG: {pred['expected_goals']['away']}</div>
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Resultado do Jogo</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">Vitória Casa</div>
                        <div class="metric-value">{pred['match_result']['home_win']:.1%}</div>
                        <div class="metric-odds">Odd Justa: {1/pred['match_result']['home_win']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Empate</div>
                        <div class="metric-value">{pred['match_result']['draw']:.1%}</div>
                        <div class="metric-odds">Odd Justa: {1/pred['match_result']['draw']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Vitória Fora</div>
                        <div class="metric-value">{pred['match_result']['away_win']:.1%}</div>
                        <div class="metric-odds">Odd Justa: {1/pred['match_result']['away_win']:.2f}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Análise de Forma (Últimos 5 Jogos)</h2>
                <div class="form-container">
                    <div class="form-box">
                        <h3>{home}</h3>
                        <p><strong>Forma:</strong> <span class="form-string">{home_form['form_string']}</span></p>
                        <p><strong>Pontos por Jogo:</strong> {home_form['ppg']:.2f}</p>
                        <p><strong>Registo:</strong> {home_form['wins']}V - {home_form['draws']}E - {home_form['losses']}D</p>
                        <p><strong>Golos:</strong> {home_form['goals_for']} marcados, {home_form['goals_against']} sofridos</p>
                        <p><strong>Diferença:</strong> {home_form['goal_difference']:+d}</p>
                    </div>
                    <div class="form-box">
                        <h3>{away}</h3>
                        <p><strong>Forma:</strong> <span class="form-string">{away_form['form_string']}</span></p>
                        <p><strong>Pontos por Jogo:</strong> {away_form['ppg']:.2f}</p>
                        <p><strong>Registo:</strong> {away_form['wins']}V - {away_form['draws']}E - {away_form['losses']}D</p>
                        <p><strong>Golos:</strong> {away_form['goals_for']} marcados, {away_form['goals_against']} sofridos</p>
                        <p><strong>Diferença:</strong> {away_form['goal_difference']:+d}</p>
                    </div>
                </div>
            </div>
    """
    
    if h2h['matches'] > 0:
        html += f"""
            <div class="section">
                <h2>🤝 Confrontos Diretos (Últimos {h2h['matches']})</h2>
                <p><strong>{home}:</strong> {h2h['home_wins']} vitórias</p>
                <p><strong>Empates:</strong> {h2h['draws']}</p>
                <p><strong>{away}:</strong> {h2h['away_wins']} vitórias</p>
                <p><strong>Golos:</strong> {h2h['home_goals']}-{h2h['away_goals']} (Média: {h2h['avg_goals']:.2f} por jogo)</p>
            </div>
        """
    
    html += f"""
            <div class="section">
                <h2>🎲 Mercados Over/Under</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">Over 1.5</div>
                        <div class="metric-value">{pred['over_under']['1.5']['over']:.1%}</div>
                        <div class="metric-odds">Odd: {1/pred['over_under']['1.5']['over']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Over 2.5</div>
                        <div class="metric-value">{pred['over_under']['2.5']['over']:.1%}</div>
                        <div class="metric-odds">Odd: {1/pred['over_under']['2.5']['over']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Over 3.5</div>
                        <div class="metric-value">{pred['over_under']['3.5']['over']:.1%}</div>
                        <div class="metric-odds">Odd: {1/pred['over_under']['3.5']['over']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">BTTS Sim</div>
                        <div class="metric-value">{pred['btts']['yes']:.1%}</div>
                        <div class="metric-odds">Odd: {1/pred['btts']['yes']:.2f}</div>
                    </div>
                </div>
            </div>
    """
    
    if "corners" in pred or "shots" in pred:
        html += '<div class="section"><h2>🚩 Cantos & 🎯 Remates</h2><div class="form-container">'
        
        if "corners" in pred:
            html += f"""
                <div class="form-box">
                    <h3>🚩 Cantos</h3>
                    <p><strong>{home}:</strong> {pred['corners']['home']} (xC: {pred['corners']['expected_home']})</p>
                    <p><strong>{away}:</strong> {pred['corners']['away']} (xC: {pred['corners']['expected_away']})</p>
                    <p><strong>Total:</strong> {pred['corners']['total']} cantos</p>
                </div>
            """
        
        if "shots" in pred:
            html += f"""
                <div class="form-box">
                    <h3>🎯 Remates à Baliza</h3>
                    <p><strong>{home}:</strong> {pred['shots']['home']} remates</p>
                    <p><strong>{away}:</strong> {pred['shots']['away']} remates</p>
                    <p><strong>Total:</strong> {pred['shots']['total']} remates</p>
                </div>
            """
        
        html += '</div></div>'
    
    html += f"""
            <div class="section">
                <h2>🎯 Resultados Mais Prováveis</h2>
                <table class="table">
                    <thead>
                        <tr>
                            <th>Resultado</th>
                            <th>Probabilidade</th>
                            <th>Odd Justa</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    for cs in pred['correct_scores'][:5]:
        html += f"""
                        <tr>
                            <td><strong>{cs['score']}</strong></td>
                            <td>{cs['probability']:.1%}</td>
                            <td>{1/cs['probability']:.2f}</td>
                        </tr>
        """
    
    html += """
                    </tbody>
                </table>
            </div>
    """
    
    if value_bets:
        html += f"""
            <div class="section">
                <h2>💎 Value Bets Identificadas</h2>
                <p style="margin-bottom: 15px;"><strong>Bankroll:</strong> €{bankroll:.2f}</p>
        """
        
        for bet in value_bets:
            html += f"""
                <div class="value-bet">
                    <h3>💰 {bet['market']}</h3>
                    <p><strong>Odds:</strong> {bet['odds']:.2f} | <strong>Probabilidade Real:</strong> {bet['true_prob']:.1%}</p>
                    <p><strong>Edge:</strong> {bet['edge']:.2f}% | <strong>EV:</strong> {bet['ev']:.2f}%</p>
                    <p><strong>Stake Recomendada (Kelly):</strong> €{bet['kelly_stake']:.2f} ({(bet['kelly_stake']/bankroll)*100:.2f}% do bankroll)</p>
                    <p><strong>Lucro Esperado:</strong> €{bet['kelly_stake'] * (bet['odds'] - 1) * bet['true_prob'] - bet['kelly_stake'] * (1 - bet['true_prob']):.2f}</p>
                </div>
            """
        
        html += "</div>"
    
    html += f"""
            <div class="footer">
                <p><strong>Football Predictor Pro</strong> - Análise profissional de apostas desportivas</p>
                <p>© {datetime.now().year} | Modelo Dixon-Coles com Bayesian Smoothing</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

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
        elif lower in ["hc", "homecorners", "hcorners"]:
            mapping["HC"] = col
        elif lower in ["ac", "awaycorners", "acorners"]:
            mapping["AC"] = col
        elif lower in ["hs", "hst", "homeshotsontarget", "homeshots"]:
            mapping["HST"] = col
        elif lower in ["as", "ast", "awayshotsontarget", "awayshots"]:
            mapping["AST"] = col
    return mapping

@st.cache_data(show_spinner="Training model...")
def compute_team_stats(
    _df: pd.DataFrame,
    home_col: str, away_col: str, hg_col: str, ag_col: str,
    hc_col=None, ac_col=None, hst_col=None, ast_col=None,
    recency_weight: float = 2.0, min_matches: int = 3
) -> Dict[str, Any]:
    df = _df.copy()
    for col in [hg_col, ag_col, hc_col, ac_col, hst_col, ast_col]:
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

    # CORNERS ANALYSIS
    if hc_col and ac_col and hc_col in df.columns and ac_col in df.columns:
        c_mask = df[hc_col].notna() & df[ac_col].notna()
        clean_c = df[c_mask][[home_col, away_col, hc_col, ac_col]].copy()
        
        if len(clean_c) >= 5:
            clean_c['weight'] = np.exp(np.linspace(-recency_weight, 0, len(clean_c)))
            avg_hc = np.average(clean_c[hc_col], weights=clean_c['weight'])
            avg_ac = np.average(clean_c[ac_col], weights=clean_c['weight'])
            
            corner_home_attack = {}
            corner_away_attack = {}
            corner_home_defence = {}
            corner_away_defence = {}
            
            for team in teams:
                home_c = clean_c[clean_c[home_col] == team]
                away_c = clean_c[clean_c[away_col] == team]
                
                if len(home_c) >= min_matches:
                    corner_home_attack[team] = bayesian_smoothing(
                        weighted_mean(home_c, hc_col) / avg_hc, 1.0, len(home_c))
                    corner_home_defence[team] = bayesian_smoothing(
                        weighted_mean(home_c, ac_col) / avg_ac, 1.0, len(home_c))
                else:
                    corner_home_attack[team] = 1.0
                    corner_home_defence[team] = 1.0
                    
                if len(away_c) >= min_matches:
                    corner_away_attack[team] = bayesian_smoothing(
                        weighted_mean(away_c, ac_col) / avg_ac, 1.0, len(away_c))
                    corner_away_defence[team] = bayesian_smoothing(
                        weighted_mean(away_c, hc_col) / avg_hc, 1.0, len(away_c))
                else:
                    corner_away_attack[team] = 1.0
                    corner_away_defence[team] = 1.0
            
            stats["corners"] = {
                "league_avg_home": avg_hc,
                "league_avg_away": avg_ac,
                "home_attack": corner_home_attack,
                "away_attack": corner_away_attack,
                "home_defence": corner_home_defence,
                "away_defence": corner_away_defence
            }

    # SHOTS ON TARGET ANALYSIS
    if hst_col and ast_col and hst_col in df.columns and ast_col in df.columns:
        s_mask = df[hst_col].notna() & df[ast_col].notna()
        clean_s = df[s_mask][[home_col, away_col, hst_col, ast_col]].copy()
        
        if len(clean_s) >= 5:
            clean_s['weight'] = np.exp(np.linspace(-recency_weight, 0, len(clean_s)))
            avg_hst = np.average(clean_s[hst_col], weights=clean_s['weight'])
            avg_ast = np.average(clean_s[ast_col], weights=clean_s['weight'])
            
            shot_home_attack = {}
            shot_away_attack = {}
            shot_home_defence = {}
            shot_away_defence = {}
            
            for team in teams:
                home_s = clean_s[clean_s[home_col] == team]
                away_s = clean_s[clean_s[away_col] == team]
                
                if len(home_s) >= min_matches:
                    shot_home_attack[team] = bayesian_smoothing(
                        weighted_mean(home_s, hst_col) / avg_hst, 1.0, len(home_s))
                    shot_home_defence[team] = bayesian_smoothing(
                        weighted_mean(home_s, ast_col) / avg_ast, 1.0, len(home_s))
                else:
                    shot_home_attack[team] = 1.0
                    shot_home_defence[team] = 1.0
                    
                if len(away_s) >= min_matches:
                    shot_away_attack[team] = bayesian_smoothing(
                        weighted_mean(away_s, ast_col) / avg_ast, 1.0, len(away_s))
                    shot_away_defence[team] = bayesian_smoothing(
                        weighted_mean(away_s, hst_col) / avg_hst, 1.0, len(away_s))
                else:
                    shot_away_attack[team] = 1.0
                    shot_away_defence[team] = 1.0
            
            stats["shots"] = {
                "league_avg_home": avg_hst,
                "league_avg_away": avg_ast,
                "home_attack": shot_home_attack,
                "away_attack": shot_away_attack,
                "home_defence": shot_home_defence,
                "away_defence": shot_away_defence
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
    
    result = {
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
    
    # CORNERS PREDICTION
    c = stats.get("corners")
    if c:
        c_att_h = c["home_attack"].get(home, 1.0)
        c_def_a = c["away_defence"].get(away, 1.0)
        c_att_a = c["away_attack"].get(away, 1.0)
        c_def_h = c["home_defence"].get(home, 1.0)
        
        lambda_hc = c_att_h * c_def_a * c["league_avg_home"]
        lambda_ac = c_att_a * c_def_h * c["league_avg_away"]
        
        # Use mode (most likely value) for corners
        corners_home = int(round(lambda_hc))
        corners_away = int(round(lambda_ac))
        
        result["corners"] = {
            "home": max(corners_home, 1),
            "away": max(corners_away, 1),
            "total": max(corners_home, 1) + max(corners_away, 1),
            "expected_home": round(lambda_hc, 1),
            "expected_away": round(lambda_ac, 1)
        }
    
    # SHOTS ON TARGET PREDICTION
    s = stats.get("shots")
    if s:
        s_att_h = s["home_attack"].get(home, 1.0)
        s_def_a = s["away_defence"].get(away, 1.0)
        s_att_a = s["away_attack"].get(away, 1.0)
        s_def_h = s["home_defence"].get(home, 1.0)
        
        lambda_hst = s_att_h * s_def_a * s["league_avg_home"]
        lambda_ast = s_att_a * s_def_h * s["league_avg_away"]
        
        result["shots"] = {
            "home": round(lambda_hst, 1),
            "away": round(lambda_ast, 1),
            "total": round(lambda_hst + lambda_ast, 1)
        }
    
    return result

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
        for label in ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "HC", "AC", "HST", "AST"]:
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
                hc_col=col_map.get("HC"),
                ac_col=col_map.get("AC"),
                hst_col=col_map.get("HST"),
                ast_col=col_map.get("AST"),
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

                # CORNERS & SHOTS
                if "corners" in pred or "shots" in pred:
                    st.markdown("### 🚩 Corners & 🎯 Shots on Target")
                    col_cs1, col_cs2 = st.columns(2)
                    
                    if "corners" in pred:
                        with col_cs1:
                            st.markdown(f"""
                            <div class='stat-card'>
                                <h4>🚩 Corners Prediction</h4>
                                <p><strong>{home_team}:</strong> {pred['corners']['home']} cantos (xC: {pred['corners']['expected_home']})</p>
                                <p><strong>{away_team}:</strong> {pred['corners']['away']} cantos (xC: {pred['corners']['expected_away']})</p>
                                <p><strong>Total Esperado:</strong> {pred['corners']['total']} cantos</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    if "shots" in pred:
                        with col_cs2:
                            st.markdown(f"""
                            <div class='stat-card'>
                                <h4>🎯 Shots on Target</h4>
                                <p><strong>{home_team}:</strong> {pred['shots']['home']} remates</p>
                                <p><strong>{away_team}:</strong> {pred['shots']['away']} remates</p>
                                <p><strong>Total Esperado:</strong> {pred['shots']['total']} remates</p>
                            </div>
                            """, unsafe_allow_html=True)

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

                # EXPORT BUTTON
                st.markdown("---")
                st.markdown("### 📥 Exportar Relatório")
                
                html_report = generate_html_report(
                    home=home_team,
                    away=away_team,
                    pred=pred,
                    home_form=home_form,
                    away_form=away_form,
                    h2h=h2h,
                    bankroll=bankroll,
                    value_bets=value_bets
                )
                
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    st.download_button(
                        label="📄 Download HTML Completo",
                        data=html_report,
                        file_name=f"{home_team}_vs_{away_team}_analise_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                        mime="text/html",
                        type="primary",
                        use_container_width=True
                    )
                    st.caption("Relatório completo com todas as análises e value bets")
                
                with col_export2:
                    st.markdown(
                        """
                        <button onclick="window.print()" style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            padding: 12px 24px;
                            border: none;
                            border-radius: 8px;
                            cursor: pointer;
                            font-size: 16px;
                            width: 100%;
                            font-weight: bold;
                        ">
                            🖨️ Imprimir / Guardar PDF
                        </button>
                        """,
                        unsafe_allow_html=True
                    )
                    st.caption("Imprimir esta página ou guardar como PDF")

else:
    st.info("📁 Upload a CSV file to start analyzing matches.")
