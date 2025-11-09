# app.py - WITH EXCEL & HTML EXPORT
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, nbinom
import io
from typing import Dict, Any, Tuple, List
import requests
from PIL import Image
from io import BytesIO
import plotly.express as px
import re
from datetime import datetime
import base64
import warnings
warnings.filterwarnings('ignore')

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Football Predictor - Last 5 Games Form", layout="wide")
st.title("⚽ Football Predictor Pro - Last 5 Games Form")
st.markdown("""
**Form-Based Analysis**  
- **Last 5 Home Games** for home teams  
- **Last 5 Away Games** for away teams  
- **Excel & HTML Export**  
- **Current Season Focus**  
""")

# ================================
# EXPORT FUNCTIONS
# ================================
def generate_excel_export(pred: Dict[str, Any], home_team: str, away_team: str, 
                         stats: Dict[str, Any], form_data: Dict[str, Any] = None) -> BytesIO:
    """Generate comprehensive Excel export with multiple sheets"""
    
    p = pred["predictions"]
    
    # Create Excel writer
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # ===== MAIN PREDICTION SHEET =====
        main_data = {
            'Metric': [
                'Match', 'Predicted Score', 'Home Win Probability', 'Draw Probability', 
                'Away Win Probability', 'BTTS Probability', 'Over 2.5 Goals Probability',
                'Home Expected Goals (xG)', 'Away Expected Goals (xG)',
                'Home Expected Corners', 'Away Expected Corners', 'Total Expected Corners',
                'Home Form Games Used', 'Away Form Games Used', 'Injury Impact'
            ],
            'Value': [
                f"{home_team} vs {away_team}", p['goals']['score'],
                f"{p['goals']['home_win']:.1%}", f"{p['goals']['draw']:.1%}",
                f"{p['goals']['away_win']:.1%}", f"{p['goals']['btts_yes']:.1%}",
                f"{p['goals']['over_25']:.1%}", f"{p['xg']['home']:.2f}",
                f"{p['xg']['away']:.2f}", str(p['corners']['home']), 
                str(p['corners']['away']), str(p['corners']['total']),
                str(p['games_used']['home']), str(p['games_used']['away']),
                p.get('injury_summary', 'None')
            ]
        }
        df_main = pd.DataFrame(main_data)
        df_main.to_excel(writer, sheet_name='Match Prediction', index=False)
        
        # Format main sheet
        worksheet = writer.sheets['Match Prediction']
        header_format = workbook.add_format({'bold': True, 'bg_color': '#2E86AB', 'font_color': 'white'})
        for col_num, value in enumerate(df_main.columns.values):
            worksheet.write(0, col_num, value, header_format)
        worksheet.set_column('A:B', 25)
        
        # ===== DETAILED PROBABILITIES SHEET =====
        max_goals = 6
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        # Recreate probability matrix for export
        g = stats.get("goals", {})
        if g:
            l_home = g["league_avg_home"]
            l_away = g["league_avg_away"]
            att_h = g["home_attack"].get(home_team, 1.0)
            def_a = g["away_defence"].get(away_team, 1.0)
            att_a = g["away_attack"].get(away_team, 1.0)
            def_h = g["home_defence"].get(home_team, 1.0)
            
            lambda_h = att_h * def_a * l_home
            lambda_a = att_a * def_h * l_away
            
            for h in range(max_goals + 1):
                for a in range(max_goals + 1):
                    prob = poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
                    prob_matrix[h, a] = prob
            prob_matrix /= prob_matrix.sum()
        
        # Create score probability dataframe
        score_probs = []
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob = prob_matrix[h, a]
                if prob > 0.001:  # Only include meaningful probabilities
                    score_probs.append({
                        'Score': f'{h}-{a}',
                        'Probability': f'{prob:.2%}',
                        'Home Goals': h,
                        'Away Goals': a
                    })
        
        df_scores = pd.DataFrame(score_probs)
        df_scores = df_scores.sort_values('Probability', ascending=False)
        df_scores.to_excel(writer, sheet_name='Score Probabilities', index=False)
        
        # Format scores sheet
        worksheet = writer.sheets['Score Probabilities']
        for col_num, value in enumerate(df_scores.columns.values):
            worksheet.write(0, col_num, value, header_format)
        worksheet.set_column('A:D', 15)
        
        # ===== TEAM FORM ANALYSIS SHEET =====
        form_data = []
        g = stats["goals"]
        
        # Home team form
        home_attack = g["home_attack"].get(home_team, 1.0)
        home_defence = g["home_defence"].get(home_team, 1.0)
        form_data.append({
            'Team': home_team,
            'Analysis Type': 'Home Form',
            'Games Used': p['games_used']['home'],
            'Attack Strength': f'{home_attack:.2f}× avg',
            'Defense Strength': f'{1/home_defence:.2f}× avg',
            'Expected Goals': p['xg']['home'],
            'Expected Corners': p['corners']['home']
        })
        
        # Away team form
        away_attack = g["away_attack"].get(away_team, 1.0)
        away_defence = g["away_defence"].get(away_team, 1.0)
        form_data.append({
            'Team': away_team,
            'Analysis Type': 'Away Form',
            'Games Used': p['games_used']['away'],
            'Attack Strength': f'{away_attack:.2f}× avg',
            'Defense Strength': f'{1/away_defence:.2f}× avg',
            'Expected Goals': p['xg']['away'],
            'Expected Corners': p['corners']['away']
        })
        
        df_form = pd.DataFrame(form_data)
        df_form.to_excel(writer, sheet_name='Team Form Analysis', index=False)
        
        # Format form sheet
        worksheet = writer.sheets['Team Form Analysis']
        for col_num, value in enumerate(df_form.columns.values):
            worksheet.write(0, col_num, value, header_format)
        worksheet.set_column('A:G', 18)
        
        # ===== LEAGUE CONTEXT SHEET =====
        league_data = {
            'Metric': [
                'League Average Home Goals',
                'League Average Away Goals', 
                'League Average Home Corners',
                'League Average Away Corners',
                'Total Matches Analyzed',
                'Prediction Model',
                'Export Date'
            ],
            'Value': [
                f"{g['league_avg_home']:.2f}",
                f"{g['league_avg_away']:.2f}",
                f"{stats['corners']['league_avg_home']:.1f}",
                f"{stats['corners']['league_avg_away']:.1f}",
                f"{sum(g['games_used'].values())}",
                'Form-Based Poisson Model',
                datetime.now().strftime('%Y-%m-%d %H:%M')
            ]
        }
        df_league = pd.DataFrame(league_data)
        df_league.to_excel(writer, sheet_name='League Context', index=False)
        
        # Format league sheet
        worksheet = writer.sheets['League Context']
        for col_num, value in enumerate(df_league.columns.values):
            worksheet.write(0, col_num, value, header_format)
        worksheet.set_column('A:B', 25)
    
    output.seek(0)
    return output

def generate_html_export(pred: Dict[str, Any], home_team: str, away_team: str, 
                        stats: Dict[str, Any], logos: Dict[str, str]) -> str:
    """Generate professional HTML export with embedded logos"""
    
    p = pred["predictions"]
    
    # Embed logos as base64
    def embed_logo(team: str) -> str:
        url = logos.get(team)
        if not url:
            return ""
        try:
            img = load_image(url)
            if img:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                b64 = base64.b64encode(buffered.getvalue()).decode()
                return f"data:image/png;base64,{b64}"
        except:
            pass
        return ""
    
    home_logo_b64 = embed_logo(home_team)
    away_logo_b64 = embed_logo(away_team)
    
    # Team form analysis
    g = stats["goals"]
    home_attack = g["home_attack"].get(home_team, 1.0)
    home_defence = g["home_defence"].get(home_team, 1.0)
    away_attack = g["away_attack"].get(away_team, 1.0)
    away_defence = g["away_defence"].get(away_team, 1.0)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{home_team} vs {away_team} - Prediction Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #2c3e50, #34495e);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .teams {{
                display: flex;
                justify-content: space-around;
                align-items: center;
                padding: 30px;
                background: #f8f9fa;
            }}
            .team {{
                text-align: center;
                flex: 1;
            }}
            .logo {{
                width: 100px;
                height: 100px;
                object-fit: contain;
                margin-bottom: 15px;
            }}
            .team-name {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .vs {{
                font-size: 36px;
                font-weight: bold;
                color: #e74c3c;
                margin: 0 40px;
            }}
            .prediction-section {{
                padding: 30px;
                background: white;
            }}
            .score {{
                text-align: center;
                font-size: 48px;
                font-weight: bold;
                color: #2c3e50;
                margin: 20px 0;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .metric-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                border-left: 4px solid #3498db;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                margin: 10px 0;
            }}
            .metric-label {{
                font-size: 14px;
                color: #7f8c8d;
                text-transform: uppercase;
            }}
            .form-analysis {{
                background: #ecf0f1;
                padding: 30px;
                margin: 20px 0;
                border-radius: 10px;
            }}
            .form-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
            }}
            .form-team {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .injury-section {{
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }}
            .footer {{
                background: #2c3e50;
                color: white;
                text-align: center;
                padding: 20px;
                font-size: 14px;
            }}
            .section-title {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Football Match Prediction Report</h1>
                <p>Form-Based Analysis | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            
            <div class="teams">
                <div class="team">
                    {f'<img src="{home_logo_b64}" class="logo" alt="{home_team}">' if home_logo_b64 else ''}
                    <div class="team-name">{home_team}</div>
                    <div>Last {p['games_used']['home']} home games analyzed</div>
                </div>
                <div class="vs">VS</div>
                <div class="team">
                    {f'<img src="{away_logo_b64}" class="logo" alt="{away_team}">' if away_logo_b64 else ''}
                    <div class="team-name">{away_team}</div>
                    <div>Last {p['games_used']['away']} away games analyzed</div>
                </div>
            </div>
            
            <div class="prediction-section">
                <div class="score">{p['goals']['score']}</div>
                <div style="text-align: center; color: #7f8c8d; margin-bottom: 30px;">
                    Most Likely Score Based on Recent Form
                </div>
                
                <h2 class="section-title">Match Probabilities</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Home Win</div>
                        <div class="metric-value">{p['goals']['home_win']:.1%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Draw</div>
                        <div class="metric-value">{p['goals']['draw']:.1%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Away Win</div>
                        <div class="metric-value">{p['goals']['away_win']:.1%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Both Teams Score</div>
                        <div class="metric-value">{p['goals']['btts_yes']:.1%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Over 2.5 Goals</div>
                        <div class="metric-value">{p['goals']['over_25']:.1%}</div>
                    </div>
                </div>
                
                <h2 class="section-title">Expected Match Statistics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Home xG</div>
                        <div class="metric-value">{p['xg']['home']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Away xG</div>
                        <div class="metric-value">{p['xg']['away']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Home Corners</div>
                        <div class="metric-value">{p['corners']['home']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Away Corners</div>
                        <div class="metric-value">{p['corners']['away']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total Corners</div>
                        <div class="metric-value">{p['corners']['total']}</div>
                    </div>
                </div>
                
                <div class="form-analysis">
                    <h2 class="section-title">Form Analysis</h2>
                    <div class="form-grid">
                        <div class="form-team">
                            <h3>{home_team} - Home Form</h3>
                            <p><strong>Attack Strength:</strong> {home_attack:.2f}× league average</p>
                            <p><strong>Defense Strength:</strong> {1/home_defence:.2f}× league average</p>
                            <p><strong>Games Analyzed:</strong> {p['games_used']['home']} recent home games</p>
                        </div>
                        <div class="form-team">
                            <h3>{away_team} - Away Form</h3>
                            <p><strong>Attack Strength:</strong> {away_attack:.2f}× league average</p>
                            <p><strong>Defense Strength:</strong> {1/away_defence:.2f}× league average</p>
                            <p><strong>Games Analyzed:</strong> {p['games_used']['away']} recent away games</p>
                        </div>
                    </div>
                </div>
                
                {f'''
                <div class="injury-section">
                    <h2 class="section-title">Injury Impact</h2>
                    <p><strong>{p['injury_summary']}</strong></p>
                </div>
                ''' if p.get('injury_summary') else ''}
            </div>
            
            <div class="footer">
                <p>Generated by Football Predictor Pro | Form-Based Analysis</p>
                <p>© {datetime.now().year} - All predictions based on statistical models</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

# ================================
# ORIGINAL FUNCTIONS (KEEP THESE FROM BEFORE)
# ================================
@st.cache_data(ttl=3600)
def get_team_logo(team_name: str) -> str:
    team_clean = team_name.strip().lower().replace(" ", "_").replace(".", "").replace("'", "")
    replacements = {
        "man_utd": "Manchester_United_F.C.", "man_city": "Manchester_City_F.C.",
        "arsenal": "Arsenal_F.C.", "chelsea": "Chelsea_F.C.", "liverpool": "Liverpool_F.C.",
        "nottm_forest": "Nottingham_Forest_F.C.", "leeds": "Leeds_United_F.C.",
        "spurs": "Tottenham_Hotspur_F.C.", "newcastle": "Newcastle_United_F.C.",
        "brighton": "Brighton_&_Hove_Albion_F.C.", "west_ham": "West_Ham_United_F.C."
    }
    wiki_name = replacements.get(team_clean, team_name.replace(" ", "_") + "_F.C.")
    url = f"https://en.wikipedia.org/wiki/File:{wiki_name}_logo.svg"
    try:
        if requests.head(url, timeout=5).status_code == 200:
            return url
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def load_image(url: str):
    try:
        response = requests.get(url, timeout=10)
        return Image.open(BytesIO(response.content)).convert("RGBA")
    except:
        return None

@st.cache_data(show_spinner="Loading CSV...")
def load_csv(uploaded_file_bytes: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(uploaded_file_bytes), encoding="utf-8")
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        return df
    except:
        return pd.read_csv(io.BytesIO(uploaded_file_bytes), encoding="latin1")

@st.cache_data(show_spinner=False)
def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapping = {}
    for col in df.columns:
        lower = col.lower().replace(" ", "")
        if "home" in lower and "team" in lower: mapping["HomeTeam"] = col
        elif "away" in lower and "team" in lower: mapping["AwayTeam"] = col
        elif lower in ["fthg", "hgoals"]: mapping["FTHG"] = col
        elif lower in ["ftag", "agoals"]: mapping["FTAG"] = col
        elif lower in ["hc", "homecorners"]: mapping["HC"] = col
        elif lower in ["ac", "awaycorners"]: mapping["AC"] = col
        elif lower in ["hs", "homeshotsontarget"]: mapping["HS"] = col
        elif lower in ["as", "awayshotsontarget"]: mapping["AS"] = col
        elif lower in ["hxg", "home_xg"]: mapping["HxG"] = col
        elif lower in ["axg", "away_xg"]: mapping["AxG"] = col
        elif "date" in lower: mapping["Date"] = col
    return mapping

def parse_injuries(injury_str: str) -> Dict[str, Dict[str, float]]:
    injuries = {}
    if not injury_str.strip(): return injuries
    for line in injury_str.split('\n'):
        match = re.match(r'(\w+):\s*(\w+)\s*\(role:(\w+),\s*impact:(\d+)%\)', line.strip())
        if match:
            team, player, role, impact = match.groups()
            impact = float(impact) / 100
            if team not in injuries: injuries[team] = {}
            injuries[team][player] = {"role": role, "impact": impact}
    return injuries

def get_last_n_home_games(df: pd.DataFrame, team: str, home_col: str, n: int = 5) -> pd.DataFrame:
    home_games = df[df[home_col] == team].copy()
    if 'Date' in home_games.columns:
        home_games = home_games.sort_values('Date', ascending=False)
    return home_games.head(n)

def get_last_n_away_games(df: pd.DataFrame, team: str, away_col: str, n: int = 5) -> pd.DataFrame:
    away_games = df[df[away_col] == team].copy()
    if 'Date' in away_games.columns:
        away_games = away_games.sort_values('Date', ascending=False)
    return away_games.head(n)

def calculate_team_form(df: pd.DataFrame, home_col: str, away_col: str, hg_col: str, ag_col: str, 
                       teams: List[str], n_games: int = 5) -> Dict[str, Any]:
    form_stats = {
        'home_attack': {}, 'home_defence': {}, 'away_attack': {}, 'away_defence': {},
        'home_games_used': {}, 'away_games_used': {}
    }
    
    recent_home_goals = []
    recent_away_goals = []
    
    for team in teams:
        home_games = get_last_n_home_games(df, team, home_col, n_games)
        form_stats['home_games_used'][team] = len(home_games)
        
        if len(home_games) > 0:
            home_goals_scored = home_games[hg_col].mean()
            home_goals_conceded = home_games[ag_col].mean()
            recent_home_goals.extend(home_games[hg_col].tolist())
            form_stats['home_attack'][team] = home_goals_scored
            form_stats['home_defence'][team] = home_goals_conceded
        else:
            form_stats['home_attack'][team] = 1.0
            form_stats['home_defence'][team] = 1.0
        
        away_games = get_last_n_away_games(df, team, away_col, n_games)
        form_stats['away_games_used'][team] = len(away_games)
        
        if len(away_games) > 0:
            away_goals_scored = away_games[ag_col].mean()
            away_goals_conceded = away_games[hg_col].mean()
            recent_away_goals.extend(away_games[ag_col].tolist())
            form_stats['away_attack'][team] = away_goals_scored
            form_stats['away_defence'][team] = away_goals_conceded
        else:
            form_stats['away_attack'][team] = 1.0
            form_stats['away_defence'][team] = 1.0
    
    form_stats['league_avg_home'] = np.mean(recent_home_goals) if recent_home_goals else 1.5
    form_stats['league_avg_away'] = np.mean(recent_away_goals) if recent_away_goals else 1.2
    
    return form_stats

@st.cache_data(show_spinner="Analyzing last 5 games form...")
def compute_form_based_stats(_df: pd.DataFrame, home_col: str, away_col: str, hg_col: str, ag_col: str,
                           hc_col=None, ac_col=None, hs_col=None, as_col=None, n_games: int = 5) -> Dict[str, Any]:
    df = _df.copy()
    for col in [hg_col, ag_col, hc_col, ac_col, hs_col, as_col]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    stats = {}
    teams = sorted(set(df[home_col]).union(df[away_col]))
    form_stats = calculate_team_form(df, home_col, away_col, hg_col, ag_col, teams, n_games)
    
    home_attack = {}; home_defence = {}; away_attack = {}; away_defence = {}
    
    for team in teams:
        home_attack[team] = form_stats['home_attack'][team] / form_stats['league_avg_home']
        home_defence[team] = form_stats['home_defence'][team] / form_stats['league_avg_away']
        away_attack[team] = form_stats['away_attack'][team] / form_stats['league_avg_away']
        away_defence[team] = form_stats['away_defence'][team] / form_stats['league_avg_home']

    stats["goals"] = {
        "league_avg_home": form_stats['league_avg_home'], "league_avg_away": form_stats['league_avg_away'],
        "home_attack": home_attack, "away_attack": away_attack, "home_defence": home_defence, "away_defence": away_defence,
        "games_used": form_stats['home_games_used'], "away_games_used": form_stats['away_games_used']
    }

    if hc_col and ac_col and hc_col in df.columns and ac_col in df.columns:
        corner_stats = calculate_team_form(df, home_col, away_col, hc_col, ac_col, teams, n_games)
        stats["corners"] = {
            "league_avg_home": corner_stats['league_avg_home'], "league_avg_away": corner_stats['league_avg_away'],
            "home_attack": {t: corner_stats['home_attack'][t] / corner_stats['league_avg_home'] for t in teams},
            "away_attack": {t: corner_stats['away_attack'][t] / corner_stats['league_avg_away'] for t in teams},
            "home_defence": {t: corner_stats['home_defence'][t] / corner_stats['league_avg_away'] for t in teams},
            "away_defence": {t: corner_stats['away_defence'][t] / corner_stats['league_avg_home'] for t in teams}
        }
    else:
        stats["corners"] = {
            "league_avg_home": 5.5, "league_avg_away": 4.8,
            "home_attack": {t: 1.0 for t in teams}, "away_attack": {t: 1.0 for t in teams},
            "home_defence": {t: 1.0 for t in teams}, "away_defence": {t: 1.0 for t in teams},
        }

    return stats

@st.cache_data(show_spinner=False)
def predict_form_based_match(home: str, away: str, stats: Dict[str, Any], injuries: Dict = None) -> Dict[str, Any]:
    injury_summary = apply_injury_adjustment(stats, injuries) if injuries else ""

    predictions = {
        "goals": {"score": "N/A", "home_win": 0, "draw": 0, "away_win": 0, "btts_yes": 0, "over_25": 0},
        "xg": {"home": 0.0, "away": 0.0}, "corners": {"home": 0, "away": 0, "total": 0},
        "form_based": True, "injury_summary": injury_summary,
        "games_used": {"home": stats["goals"]["games_used"].get(home, 0), "away": stats["goals"]["away_games_used"].get(away, 0)}
    }

    g = stats.get("goals", {})
    if g:
        l_home = g["league_avg_home"]; l_away = g["league_avg_away"]
        att_h = g["home_attack"].get(home, 1.0); def_a = g["away_defence"].get(away, 1.0)
        att_a = g["away_attack"].get(away, 1.0); def_h = g["home_defence"].get(home, 1.0)
        lambda_h = att_h * def_a * l_home; lambda_a = att_a * def_h * l_away

        max_g = 8
        prob_matrix = np.zeros((max_g + 1, max_g + 1))
        for h in range(max_g + 1):
            for a in range(max_g + 1):
                p = poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
                prob_matrix[h, a] = p
        prob_matrix /= prob_matrix.sum()
        
        h_idx, a_idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
        predictions["goals"]["score"] = f"{h_idx}–{a_idx}"
        predictions["goals"]["home_win"] = (prob_matrix[1:, :].sum() - prob_matrix.diagonal()[1:].sum())
        predictions["goals"]["away_win"] = (prob_matrix[:, 1:].sum() - prob_matrix.diagonal()[1:].sum())
        predictions["goals"]["draw"] = prob_matrix.diagonal().sum()
        predictions["goals"]["btts_yes"] = (prob_matrix[1:, 1:]).sum()
        predictions["goals"]["over_25"] = (prob_matrix[3:, :].sum() + prob_matrix[:, 3:].sum() - prob_matrix[3:, 3:].sum())
        predictions["xg"]["home"] = max(round(lambda_h, 2), 0.1)
        predictions["xg"]["away"] = max(round(lambda_a, 2), 0.1)

    c = stats.get("corners")
    if c:
        mu_hc = c["home_attack"].get(home, 1.0) * c["away_defence"].get(away, 1.0) * c["league_avg_home"]
        mu_ac = c["away_attack"].get(away, 1.0) * c["home_defence"].get(home, 1.0) * c["league_avg_away"]
        predictions["corners"]["home"] = max(int(np.round(mu_hc)), 1)
        predictions["corners"]["away"] = max(int(np.round(mu_ac)), 1)
        predictions["corners"]["total"] = predictions["corners"]["home"] + predictions["corners"]["away"]

    return {"predictions": predictions}

def apply_injury_adjustment(stats: Dict[str, Any], injuries: Dict[str, Dict[str, float]]) -> str:
    summary = ""
    for team, players in injuries.items():
        attack_reduction = defence_reduction = 0
        for p, data in players.items():
            if data["role"] in ["forward", "midfielder", "winger", "striker"]:
                attack_reduction += data["impact"]
            elif data["role"] in ["defender", "goalkeeper"]:
                defence_reduction += data["impact"]
        attack_reduction = min(attack_reduction, 0.3); defence_reduction = min(defence_reduction, 0.3)
        if attack_reduction > 0: summary += f"{team} Attack -{attack_reduction*100:.0f}% | "
        if defence_reduction > 0: summary += f"{team} Defence -{defence_reduction*100:.0f}% | "
    return summary.strip(" | ")

def display_form_based_predictions(pred: Dict[str, Any], home_team: str, away_team: str, stats: Dict[str, Any]):
    p = pred["predictions"]
    st.markdown(f"### **{home_team} vs {away_team}**")
    st.markdown("#### 🎯 Last 5 Games Form Analysis")
    
    logos = {home_team: get_team_logo(home_team), away_team: get_team_logo(away_team)}
    colA, colB, colC = st.columns([1,2,1])
    with colA:
        if logos[home_team]: 
            img = load_image(logos[home_team])
            if img: st.image(img, width=80)
        st.write(f"**{home_team}**"); st.caption(f"Last {p['games_used']['home']} home games")
    with colC:
        if logos[away_team]: 
            img = load_image(logos[away_team])
            if img: st.image(img, width=80)
        st.write(f"**{away_team}**"); st.caption(f"Last {p['games_used']['away']} away games")
    with colB:
        st.markdown(f"<h2 style='text-align:center'>{p['goals']['score']}</h2>", unsafe_allow_html=True)
        st.caption("Most likely score based on recent form")

    colW1, colW2, colW3 = st.columns(3)
    colW1.metric("Home Win", f"{p['goals']['home_win']:.1%}"); colW2.metric("Draw", f"{p['goals']['draw']:.1%}"); colW3.metric("Away Win", f"{p['goals']['away_win']:.1%}")
    colB1, colB2 = st.columns(2)
    colB1.metric("Both Teams to Score", f"{p['goals']['btts_yes']:.1%}"); colB2.metric("Over 2.5 Goals", f"{p['goals']['over_25']:.1%}")

    st.markdown("#### ⚽ Expected Match Stats")
    colX1, colX2 = st.columns(2)
    with colX1: st.write(f"**Expected Goals (xG)**\n{home_team}: **{p['xg']['home']}**\n{away_team}: **{p['xg']['away']}**")
    with colX2: st.write(f"**Expected Corners**\n{home_team}: **{p['corners']['home']}**\n{away_team}: **{p['corners']['away']}**\n**Total**: **{p['corners']['total']}**")

    # ===== EXPORT BUTTONS =====
    st.markdown("---")
    st.markdown("#### 📤 Export Prediction")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        # Excel Export
        excel_data = generate_excel_export(pred, home_team, away_team, stats)
        st.download_button(
            label="📊 Download Excel Report",
            data=excel_data,
            file_name=f"{home_team}_vs_{away_team}_prediction.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        st.caption("Comprehensive report with multiple sheets")
    
    with col_exp2:
        # HTML Export
        html_content = generate_html_export(pred, home_team, away_team, stats, logos)
        st.download_button(
            label="🌐 Download HTML Report", 
            data=html_content,
            file_name=f"{home_team}_vs_{away_team}_prediction.html",
            mime="text/html",
            use_container_width=True
        )
        st.caption("Professional report with team logos")

    # Form Analysis
    st.markdown("#### 📈 Recent Form Analysis")
    g = stats["goals"]
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{home_team} Home Form**")
        home_attack = g["home_attack"].get(home_team, 1.0); home_defence = g["home_defence"].get(home_team, 1.0)
        st.write(f"Attack: {home_attack:.2f}× avg"); st.write(f"Defence: {1/home_defence:.2f}× avg")
    with col2:
        st.write(f"**{away_team} Away Form**")
        away_attack = g["away_attack"].get(away_team, 1.0); away_defence = g["away_defence"].get(away_team, 1.0)
        st.write(f"Attack: {away_attack:.2f}× avg"); st.write(f"Defence: {1/away_defence:.2f}× avg")

    if p["injury_summary"]:
        st.markdown(f"#### 🏥 Injury Impact")
        st.markdown(f"<span style='color:red'>{p['injury_summary']}</span>", unsafe_allow_html=True)

# ================================
# MAIN APP
# ================================
st.sidebar.header("📁 Upload Match Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV File", type=["csv"])

if uploaded_file is not None:
    df = load_csv(uploaded_file.read())
    if df.empty:
        st.error("Empty CSV file.")
    else:
        st.success(f"✅ Loaded {len(df):,} matches")
        with st.expander("📊 Data Preview"): st.dataframe(df.head(8))
        
        mapping = detect_columns(df)
        st.sidebar.subheader("🔧 Column Mapping")
        col_map = {}
        for label in ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "HC", "AC", "Date"]:
            detected = mapping.get(label)
            options = [""] + list(df.columns)
            default_idx = options.index(detected) if detected in options else 0
            col_map[label] = st.sidebar.selectbox(f"**{label}**", options=options, index=default_idx)

        missing = [r for r in ["HomeTeam", "AwayTeam", "FTHG", "FTAG"] if not col_map[r]]
        if missing: st.error(f"❌ Map required fields: {', '.join(missing)}"); st.stop()

        st.sidebar.subheader("⚙️ Form Analysis Settings")
        n_games = st.sidebar.slider("Number of games for form analysis", 3, 10, 5)
        require_dates = st.sidebar.toggle("Require date column", value=True)

        if require_dates and not col_map.get("Date"):
            st.warning("⚠️ Date column not mapped. Form analysis may be less accurate.")
            df = df.sort_index(ascending=False)

        with st.spinner(f"🔄 Analyzing last {n_games} home/away games form..."):
            team_stats = compute_form_based_stats(_df=df, home_col=col_map["HomeTeam"], away_col=col_map["AwayTeam"],
                                                hg_col=col_map["FTHG"], ag_col=col_map["FTAG"], hc_col=col_map.get("HC"), 
                                                ac_col=col_map.get("AC"), n_games=n_games)

        teams = sorted(set(df[col_map["HomeTeam"]]).union(df[col_map["AwayTeam"]]))
        st.sidebar.subheader("🏥 Current Injuries")
        injury_input = st.sidebar.text_area("Injured Players", placeholder="Arsenal: Saka (role:forward, impact:15%)", height=100)
        injuries = parse_injuries(injury_input)

        st.markdown("---")
        st.subheader("🔮 Form-Based Match Prediction")
        col1, col2 = st.columns(2)
        home_team = col1.selectbox("Home Team", teams, key="home_select")
        away_team = col2.selectbox("Away Team", teams, key="away_select")

        if st.button(f"🎯 Predict Based on Last {n_games} Games", type="primary", use_container_width=True):
            with st.spinner("Analyzing recent form..."):
                pred = predict_form_based_match(home_team, away_team, team_stats, injuries)
                display_form_based_predictions(pred, home_team, away_team, team_stats)

else:
    st.info("📁 Please upload CSV data to get started")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Form Analysis")
st.sidebar.info(f"Using last {n_games if 'n_games' in locals() else 5} home/away games")
