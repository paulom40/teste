import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from io import BytesIO
import math
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Football Betting Model", layout="wide", page_icon="⚽")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
    }
    .value-bet {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .download-btn {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .team-stats-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">⚽ Professional Football Betting Model</h1>', unsafe_allow_html=True)

# Load and process data
@st.cache_data
def load_data(source='default', uploaded_file=None):
    """Load data from default URL or uploaded file"""
    if source == 'upload' and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
        df = pd.read_csv(url)
    
    # Try different date formats
    date_formats = ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y']
    for fmt in date_formats:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format=fmt)
            break
        except:
            continue
    
    # If all formats fail, use automatic parsing
    if df['Date'].dtype != 'datetime64[ns]':
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Validate required columns
    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        st.info("Please ensure your CSV has these columns: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, B365H, B365D, B365A")
        return None
    
    return df

def calculate_team_stats(df):
    """Calculate comprehensive team statistics"""
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    stats = {}
    
    for team in teams:
        home_games = df[df['HomeTeam'] == team]
        away_games = df[df['AwayTeam'] == team]
        
        # Overall stats
        total_games = len(home_games) + len(away_games)
        
        # Home stats
        home_wins = (home_games['FTR'] == 'H').sum()
        home_draws = (home_games['FTR'] == 'D').sum()
        home_goals = home_games['FTHG'].sum()
        home_conceded = home_games['FTAG'].sum()
        
        # Away stats
        away_wins = (away_games['FTR'] == 'A').sum()
        away_draws = (away_games['FTR'] == 'D').sum()
        away_goals = away_games['FTAG'].sum()
        away_conceded = away_games['FTHG'].sum()
        
        # Total stats
        total_wins = home_wins + away_wins
        total_goals = home_goals + away_goals
        total_conceded = home_conceded + away_conceded
        
        # Calculate strength metrics
        stats[team] = {
            'games': total_games,
            'wins': total_wins,
            'win_rate': total_wins / total_games if total_games > 0 else 0,
            'home_win_rate': home_wins / len(home_games) if len(home_games) > 0 else 0,
            'away_win_rate': away_wins / len(away_games) if len(away_games) > 0 else 0,
            'goals_per_game': total_goals / total_games if total_games > 0 else 0,
            'conceded_per_game': total_conceded / total_games if total_games > 0 else 0,
            'home_goals_avg': home_goals / len(home_games) if len(home_games) > 0 else 0,
            'away_goals_avg': away_goals / len(away_games) if len(away_games) > 0 else 0,
            'goal_difference': total_goals - total_conceded
        }
    
    return stats

def generate_excel_special_markets(df):
    """Generate Excel file with special markets analysis"""
    
    # Create Excel writer object
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#667eea',
            'font_color': 'white',
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        title_format = workbook.add_format({
            'bold': True,
            'font_size': 14,
            'bg_color': '#764ba2',
            'font_color': 'white',
            'align': 'center',
            'valign': 'vcenter'
        })
        
        data_format = workbook.add_format({
            'border': 1,
            'align': 'center'
        })
        
        percent_format = workbook.add_format({
            'border': 1,
            'align': 'center',
            'num_format': '0.0%'
        })
        
        # Calculate totals
        if 'HST' in df.columns and 'AST' in df.columns:
            df['TotalSOT'] = df['HST'] + df['AST']
        else:
            df['TotalSOT'] = 0
        
        if 'HC' in df.columns and 'AC' in df.columns:
            df['TotalCorners'] = df['HC'] + df['AC']
        else:
            df['TotalCorners'] = 0
        
        df['TotalGoals'] = df['FTHG'] + df['FTAG']
        
        # Sheet 1: Match by Match Analysis
        match_analysis = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'TotalGoals', 
                             'HST', 'AST', 'TotalSOT', 'HC', 'AC', 'TotalCorners']].copy()
        match_analysis['Date'] = pd.to_datetime(match_analysis['Date']).dt.strftime('%Y-%m-%d')
        match_analysis.to_excel(writer, sheet_name='Match Analysis', index=False, startrow=2)
        
        worksheet1 = writer.sheets['Match Analysis']
        worksheet1.merge_range('A1:L1', '⚽ MATCH BY MATCH SPECIAL MARKETS ANALYSIS', title_format)
        
        for col_num, value in enumerate(match_analysis.columns.values):
            worksheet1.write(2, col_num, value, header_format)
        
        worksheet1.set_column('A:A', 12)
        worksheet1.set_column('B:C', 15)
        worksheet1.set_column('D:L', 12)
        
        # Sheet 2: Goals Market Summary
        goals_summary = pd.DataFrame({
            'Market Line': ['Over 0.5', 'Over 1.5', 'Over 2.5', 'Over 3.5', 'Over 4.5'],
            'Hit Rate': [
                (df['TotalGoals'] > 0.5).sum() / len(df),
                (df['TotalGoals'] > 1.5).sum() / len(df),
                (df['TotalGoals'] > 2.5).sum() / len(df),
                (df['TotalGoals'] > 3.5).sum() / len(df),
                (df['TotalGoals'] > 4.5).sum() / len(df)
            ],
            'Total Matches': [
                (df['TotalGoals'] > 0.5).sum(),
                (df['TotalGoals'] > 1.5).sum(),
                (df['TotalGoals'] > 2.5).sum(),
                (df['TotalGoals'] > 3.5).sum(),
                (df['TotalGoals'] > 4.5).sum()
            ],
            'Average Goals': df['TotalGoals'].mean()
        })
        
        goals_summary.to_excel(writer, sheet_name='Goals Market', index=False, startrow=2)
        
        worksheet2 = writer.sheets['Goals Market']
        worksheet2.merge_range('A1:D1', '⚽ GOALS MARKET ANALYSIS', title_format)
        
        for col_num, value in enumerate(goals_summary.columns.values):
            worksheet2.write(2, col_num, value, header_format)
        
        for row_num in range(len(goals_summary)):
            worksheet2.write(row_num + 3, 1, goals_summary.iloc[row_num, 1], percent_format)
        
        worksheet2.set_column('A:A', 15)
        worksheet2.set_column('B:D', 15)
        
        # Sheet 3: Shots on Target Market
        if df['TotalSOT'].sum() > 0:
            sot_summary = pd.DataFrame({
                'Market Line': ['Over 6.5', 'Over 8.5', 'Over 10.5', 'Over 12.5', 'Over 14.5'],
                'Hit Rate': [
                    (df['TotalSOT'] > 6.5).sum() / len(df),
                    (df['TotalSOT'] > 8.5).sum() / len(df),
                    (df['TotalSOT'] > 10.5).sum() / len(df),
                    (df['TotalSOT'] > 12.5).sum() / len(df),
                    (df['TotalSOT'] > 14.5).sum() / len(df)
                ],
                'Total Matches': [
                    (df['TotalSOT'] > 6.5).sum(),
                    (df['TotalSOT'] > 8.5).sum(),
                    (df['TotalSOT'] > 10.5).sum(),
                    (df['TotalSOT'] > 12.5).sum(),
                    (df['TotalSOT'] > 14.5).sum()
                ],
                'Average SOT': df['TotalSOT'].mean()
            })
            
            sot_summary.to_excel(writer, sheet_name='Shots on Target', index=False, startrow=2)
            
            worksheet3 = writer.sheets['Shots on Target']
            worksheet3.merge_range('A1:D1', '🎯 SHOTS ON TARGET MARKET ANALYSIS', title_format)
            
            for col_num, value in enumerate(sot_summary.columns.values):
                worksheet3.write(2, col_num, value, header_format)
            
            for row_num in range(len(sot_summary)):
                worksheet3.write(row_num + 3, 1, sot_summary.iloc[row_num, 1], percent_format)
            
            worksheet3.set_column('A:A', 15)
            worksheet3.set_column('B:D', 15)
        
        # Sheet 4: Corners Market
        if df['TotalCorners'].sum() > 0:
            corners_summary = pd.DataFrame({
                'Market Line': ['Over 6.5', 'Over 8.5', 'Over 10.5', 'Over 12.5', 'Over 14.5'],
                'Hit Rate': [
                    (df['TotalCorners'] > 6.5).sum() / len(df),
                    (df['TotalCorners'] > 8.5).sum() / len(df),
                    (df['TotalCorners'] > 10.5).sum() / len(df),
                    (df['TotalCorners'] > 12.5).sum() / len(df),
                    (df['TotalCorners'] > 14.5).sum() / len(df)
                ],
                'Total Matches': [
                    (df['TotalCorners'] > 6.5).sum(),
                    (df['TotalCorners'] > 8.5).sum(),
                    (df['TotalCorners'] > 10.5).sum(),
                    (df['TotalCorners'] > 12.5).sum(),
                    (df['TotalCorners'] > 14.5).sum()
                ],
                'Average Corners': df['TotalCorners'].mean()
            })
            
            corners_summary.to_excel(writer, sheet_name='Corners Market', index=False, startrow=2)
            
            worksheet4 = writer.sheets['Corners Market']
            worksheet4.merge_range('A1:D1', '🚩 CORNERS MARKET ANALYSIS', title_format)
            
            for col_num, value in enumerate(corners_summary.columns.values):
                worksheet4.write(2, col_num, value, header_format)
            
            for row_num in range(len(corners_summary)):
                worksheet4.write(row_num + 3, 1, corners_summary.iloc[row_num, 1], percent_format)
            
            worksheet4.set_column('A:A', 15)
            worksheet4.set_column('B:D', 15)
        
        # Sheet 5: Team Statistics
        team_stats_data = []
        for team in df['HomeTeam'].unique():
            home_games = df[df['HomeTeam'] == team]
            away_games = df[df['AwayTeam'] == team]
            
            team_stats_data.append({
                'Team': team,
                'Matches': len(home_games) + len(away_games),
                'Avg Goals Scored': (home_games['FTHG'].sum() + away_games['FTAG'].sum()) / (len(home_games) + len(away_games)),
                'Avg Goals Conceded': (home_games['FTAG'].sum() + away_games['FTHG'].sum()) / (len(home_games) + len(away_games)),
                'Avg SOT For': (home_games['HST'].sum() + away_games['AST'].sum()) / (len(home_games) + len(away_games)) if 'HST' in df.columns else 0,
                'Avg SOT Against': (home_games['AST'].sum() + away_games['HST'].sum()) / (len(home_games) + len(away_games)) if 'AST' in df.columns else 0,
                'Avg Corners For': (home_games['HC'].sum() + away_games['AC'].sum()) / (len(home_games) + len(away_games)) if 'HC' in df.columns else 0,
                'Avg Corners Against': (home_games['AC'].sum() + away_games['HC'].sum()) / (len(home_games) + len(away_games)) if 'AC' in df.columns else 0
            })
        
        team_stats_df = pd.DataFrame(team_stats_data)
        team_stats_df = team_stats_df.sort_values('Avg Goals Scored', ascending=False)
        team_stats_df.to_excel(writer, sheet_name='Team Statistics', index=False, startrow=2)
        
        worksheet5 = writer.sheets['Team Statistics']
        worksheet5.merge_range('A1:H1', '📊 TEAM SPECIAL MARKETS STATISTICS', title_format)
        
        for col_num, value in enumerate(team_stats_df.columns.values):
            worksheet5.write(2, col_num, value, header_format)
        
        worksheet5.set_column('A:A', 20)
        worksheet5.set_column('B:H', 15)
        
        # Sheet 6: Summary Dashboard
        summary_data = pd.DataFrame({
            'Metric': [
                'Total Matches',
                'Average Goals per Match',
                'Average SOT per Match',
                'Average Corners per Match',
                'Over 2.5 Goals %',
                'Over 10.5 SOT %',
                'Over 10.5 Corners %',
                'Highest Scoring Match',
                'Most Corners Match'
            ],
            'Value': [
                len(df),
                f"{df['TotalGoals'].mean():.2f}",
                f"{df['TotalSOT'].mean():.2f}" if df['TotalSOT'].sum() > 0 else 'N/A',
                f"{df['TotalCorners'].mean():.2f}" if df['TotalCorners'].sum() > 0 else 'N/A',
                f"{((df['TotalGoals'] > 2.5).sum() / len(df) * 100):.1f}%",
                f"{((df['TotalSOT'] > 10.5).sum() / len(df) * 100):.1f}%" if df['TotalSOT'].sum() > 0 else 'N/A',
                f"{((df['TotalCorners'] > 10.5).sum() / len(df) * 100):.1f}%" if df['TotalCorners'].sum() > 0 else 'N/A',
                f"{df['TotalGoals'].max():.0f} goals",
                f"{df['TotalCorners'].max():.0f} corners" if df['TotalCorners'].sum() > 0 else 'N/A'
            ]
        })
        
        summary_data.to_excel(writer, sheet_name='Summary Dashboard', index=False, startrow=2)
        
        worksheet6 = writer.sheets['Summary Dashboard']
        worksheet6.merge_range('A1:B1', '📈 SPECIAL MARKETS SUMMARY DASHBOARD', title_format)
        
        for col_num, value in enumerate(summary_data.columns.values):
            worksheet6.write(2, col_num, value, header_format)
        
        worksheet6.set_column('A:A', 30)
        worksheet6.set_column('B:B', 20)
    
    output.seek(0)
    return output

def generate_special_markets_summary(df):
    """Generate a summary DataFrame for special markets"""
    # Calculate totals
    if 'HST' in df.columns and 'AST' in df.columns:
        df['TotalSOT'] = df['HST'] + df['AST']
    else:
        df['TotalSOT'] = 0
    
    if 'HC' in df.columns and 'AC' in df.columns:
        df['TotalCorners'] = df['HC'] + df['AC']
    else:
        df['TotalCorners'] = 0
    
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    
    # Goals Market Summary
    goals_summary = pd.DataFrame({
        'Market': ['Goals Market'] * 5,
        'Line': ['Over 0.5', 'Over 1.5', 'Over 2.5', 'Over 3.5', 'Over 4.5'],
        'Hit Rate': [
            (df['TotalGoals'] > 0.5).sum() / len(df),
            (df['TotalGoals'] > 1.5).sum() / len(df),
            (df['TotalGoals'] > 2.5).sum() / len(df),
            (df['TotalGoals'] > 3.5).sum() / len(df),
            (df['TotalGoals'] > 4.5).sum() / len(df)
        ],
        'Hit Count': [
            (df['TotalGoals'] > 0.5).sum(),
            (df['TotalGoals'] > 1.5).sum(),
            (df['TotalGoals'] > 2.5).sum(),
            (df['TotalGoals'] > 3.5).sum(),
            (df['TotalGoals'] > 4.5).sum()
        ],
        'Average': df['TotalGoals'].mean()
    })
    
    # Shots on Target Summary
    sot_summary = pd.DataFrame({
        'Market': ['Shots on Target'] * 5,
        'Line': ['Over 6.5', 'Over 8.5', 'Over 10.5', 'Over 12.5', 'Over 14.5'],
        'Hit Rate': [
            (df['TotalSOT'] > 6.5).sum() / len(df) if df['TotalSOT'].sum() > 0 else 0,
            (df['TotalSOT'] > 8.5).sum() / len(df) if df['TotalSOT'].sum() > 0 else 0,
            (df['TotalSOT'] > 10.5).sum() / len(df) if df['TotalSOT'].sum() > 0 else 0,
            (df['TotalSOT'] > 12.5).sum() / len(df) if df['TotalSOT'].sum() > 0 else 0,
            (df['TotalSOT'] > 14.5).sum() / len(df) if df['TotalSOT'].sum() > 0 else 0
        ],
        'Hit Count': [
            (df['TotalSOT'] > 6.5).sum(),
            (df['TotalSOT'] > 8.5).sum(),
            (df['TotalSOT'] > 10.5).sum(),
            (df['TotalSOT'] > 12.5).sum(),
            (df['TotalSOT'] > 14.5).sum()
        ],
        'Average': df['TotalSOT'].mean() if df['TotalSOT'].sum() > 0 else 0
    })
    
    # Corners Summary
    corners_summary = pd.DataFrame({
        'Market': ['Corners'] * 5,
        'Line': ['Over 6.5', 'Over 8.5', 'Over 10.5', 'Over 12.5', 'Over 14.5'],
        'Hit Rate': [
            (df['TotalCorners'] > 6.5).sum() / len(df) if df['TotalCorners'].sum() > 0 else 0,
            (df['TotalCorners'] > 8.5).sum() / len(df) if df['TotalCorners'].sum() > 0 else 0,
            (df['TotalCorners'] > 10.5).sum() / len(df) if df['TotalCorners'].sum() > 0 else 0,
            (df['TotalCorners'] > 12.5).sum() / len(df) if df['TotalCorners'].sum() > 0 else 0,
            (df['TotalCorners'] > 14.5).sum() / len(df) if df['TotalCorners'].sum() > 0 else 0
        ],
        'Hit Count': [
            (df['TotalCorners'] > 6.5).sum(),
            (df['TotalCorners'] > 8.5).sum(),
            (df['TotalCorners'] > 10.5).sum(),
            (df['TotalCorners'] > 12.5).sum(),
            (df['TotalCorners'] > 14.5).sum()
        ],
        'Average': df['TotalCorners'].mean() if df['TotalCorners'].sum() > 0 else 0
    })
    
    # Combine all summaries
    summary_df = pd.concat([goals_summary, sot_summary, corners_summary], ignore_index=True)
    
    # Format percentages
    summary_df['Hit Rate'] = summary_df['Hit Rate'].apply(lambda x: f"{x*100:.1f}%")
    
    return summary_df

# ... [REST OF THE FUNCTIONS REMAIN THE SAME UNTIL THE SPECIAL MARKETS TAB] ...

with tab5:
    st.header("🎯 Special Markets Analysis")
    
    # Add download button at the top
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col2:
        # Generate and download Excel button
        download_clicked = st.button("📥 Download Excel Report", type="secondary", use_container_width=True)
    
    with col3:
        # Generate simple summary CSV
        csv_clicked = st.button("📊 Download Summary CSV", type="secondary", use_container_width=True)
    
    st.markdown("---")
    
    # Model comparison table
    with st.expander("📊 Model Comparison Guide", expanded=False):
        model_comparison = pd.DataFrame({
            'Model': ['Statistical', 'Poisson', 'Dixon-Coles', 'Negative Binomial', 'Ensemble'],
            'Speed': ['⚡⚡⚡', '⚡⚡⚡', '⚡⚡', '⚡⚡', '⚡'],
            'Accuracy': ['⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'],
            'Best Use Case': [
                'General purpose, special markets',
                'Fast baseline predictions',
                'Low-scoring leagues, draws',
                'High-scoring, unpredictable',
                'Maximum reliability'
            ]
        })
        
        st.dataframe(
            model_comparison,
            width='stretch',
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "Speed": st.column_config.TextColumn("Speed", width="small"),
                "Accuracy": st.column_config.TextColumn("Accuracy", width="medium"),
                "Best Use Case": st.column_config.TextColumn("Best Use Case", width="large")
            }
        )
    
    st.info("📊 Comprehensive analysis of Goals, Shots on Target, and Corners markets across all matches")
    
    # Calculate market statistics
    if 'HST' in df.columns and 'AST' in df.columns:
        df['TotalSOT'] = df['HST'] + df['AST']
    else:
        df['TotalSOT'] = 0
    
    if 'HC' in df.columns and 'AC' in df.columns:
        df['TotalCorners'] = df['HC'] + df['AC']
    else:
        df['TotalCorners'] = 0
    
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    
    # Market Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_goals = df['TotalGoals'].mean()
        over_25_pct = (df['TotalGoals'] > 2.5).sum() / len(df) * 100
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
            <h3 style='margin: 0;'>⚽ Goals Market</h3>
            <h1 style='margin: 1rem 0;'>{avg_goals:.2f}</h1>
            <p style='margin: 0;'>Avg Total Goals</p>
            <p style='margin: 0.5rem 0; font-size: 1.2rem;'>{over_25_pct:.1f}% Over 2.5</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_sot = df['TotalSOT'].mean()
        over_10_sot_pct = (df['TotalSOT'] > 10.5).sum() / len(df) * 100 if df['TotalSOT'].sum() > 0 else 0
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
            <h3 style='margin: 0;'>🎯 Shots on Target</h3>
            <h1 style='margin: 1rem 0;'>{avg_sot:.2f}</h1>
            <p style='margin: 0;'>Avg Total SOT</p>
            <p style='margin: 0.5rem 0; font-size: 1.2rem;'>{over_10_sot_pct:.1f}% Over 10.5</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_corners = df['TotalCorners'].mean()
        over_10_corners_pct = (df['TotalCorners'] > 10.5).sum() / len(df) * 100 if df['TotalCorners'].sum() > 0 else 0
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
            <h3 style='margin: 0;'>🚩 Corners</h3>
            <h1 style='margin: 1rem 0;'>{avg_corners:.2f}</h1>
            <p style='margin: 0;'>Avg Total Corners</p>
            <p style='margin: 0.5rem 0; font-size: 1.2rem;'>{over_10_corners_pct:.1f}% Over 10.5</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # NEW SECTION: Team-by-Team Special Markets Analysis
    st.markdown("### 🏆 Team-by-Team Special Markets Analysis")
    
    # Calculate team-specific special markets stats
    team_special_stats = []
    for team in df['HomeTeam'].unique():
        home_games = df[df['HomeTeam'] == team]
        away_games = df[df['AwayTeam'] == team]
        all_games = pd.concat([home_games, away_games])
        
        if len(all_games) > 0:
            # Calculate team's performance in special markets
            team_special_stats.append({
                'Team': team,
                'Matches': len(all_games),
                'Avg Goals For': (home_games['FTHG'].sum() + away_games['FTAG'].sum()) / len(all_games),
                'Avg Goals Against': (home_games['FTAG'].sum() + away_games['FTHG'].sum()) / len(all_games),
                'Avg Total Goals': all_games['TotalGoals'].mean(),
                'Over 2.5 %': (all_games['TotalGoals'] > 2.5).sum() / len(all_games) * 100,
                'Over 3.5 %': (all_games['TotalGoals'] > 3.5).sum() / len(all_games) * 100,
                'Avg SOT For': (home_games['HST'].sum() + away_games['AST'].sum()) / len(all_games) if 'HST' in df.columns else 0,
                'Avg SOT Against': (home_games['AST'].sum() + away_games['HST'].sum()) / len(all_games) if 'AST' in df.columns else 0,
                'Avg Total SOT': all_games['TotalSOT'].mean() if 'HST' in df.columns else 0,
                'Over 10.5 SOT %': (all_games['TotalSOT'] > 10.5).sum() / len(all_games) * 100 if 'HST' in df.columns else 0,
                'Avg Corners For': (home_games['HC'].sum() + away_games['AC'].sum()) / len(all_games) if 'HC' in df.columns else 0,
                'Avg Corners Against': (home_games['AC'].sum() + away_games['HC'].sum()) / len(all_games) if 'AC' in df.columns else 0,
                'Avg Total Corners': all_games['TotalCorners'].mean() if 'HC' in df.columns else 0,
                'Over 10.5 Corners %': (all_games['TotalCorners'] > 10.5).sum() / len(all_games) * 100 if 'HC' in df.columns else 0
            })
    
    team_special_df = pd.DataFrame(team_special_stats)
    
    # Display all teams with their stats
    st.markdown("#### All Teams Special Markets Statistics")
    
    # Format the dataframe for display
    display_df = team_special_df.copy()
    
    # Format percentages
    display_df['Over 2.5 %'] = display_df['Over 2.5 %'].apply(lambda x: f"{x:.1f}%")
    display_df['Over 3.5 %'] = display_df['Over 3.5 %'].apply(lambda x: f"{x:.1f}%")
    
    if 'HST' in df.columns:
        display_df['Over 10.5 SOT %'] = display_df['Over 10.5 SOT %'].apply(lambda x: f"{x:.1f}%" if x > 0 else "N/A")
    
    if 'HC' in df.columns:
        display_df['Over 10.5 Corners %'] = display_df['Over 10.5 Corners %'].apply(lambda x: f"{x:.1f}%" if x > 0 else "N/A")
    
    # Allow sorting by different metrics
    sort_option = st.selectbox(
        "Sort teams by:",
        ["Team Name", "Avg Total Goals", "Over 2.5 %", "Avg Total SOT", "Avg Total Corners"],
        index=1
    )
    
    if sort_option == "Team Name":
        display_df = display_df.sort_values('Team')
    elif sort_option == "Avg Total Goals":
        display_df = display_df.sort_values('Avg Total Goals', ascending=False)
    elif sort_option == "Over 2.5 %":
        display_df = display_df.sort_values('Over 2.5 %', ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
    elif sort_option == "Avg Total SOT":
        display_df = display_df.sort_values('Avg Total SOT', ascending=False)
    elif sort_option == "Avg Total Corners":
        display_df = display_df.sort_values('Avg Total Corners', ascending=False)
    
    st.dataframe(
        display_df[[
            'Team', 'Matches', 'Avg Total Goals', 'Over 2.5 %', 'Over 3.5 %',
            'Avg Total SOT' if 'HST' in df.columns else None,
            'Over 10.5 SOT %' if 'HST' in df.columns else None,
            'Avg Total Corners' if 'HC' in df.columns else None,
            'Over 10.5 Corners %' if 'HC' in df.columns else None
        ]].dropna(axis=1),
        width='stretch',
        hide_index=True,
        height=400
    )
    
    st.markdown("---")
    
    # Team Leaders Section
    st.markdown("### 🏅 Team Leaders in Special Markets")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**🎯 Top 5 Scoring Teams**")
        top_scoring = team_special_df.nlargest(5, 'Avg Total Goals')[['Team', 'Avg Total Goals', 'Over 2.5 %']]
        top_scoring['Over 2.5 %'] = top_scoring['Over 2.5 %'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(top_scoring, width='stretch', hide_index=True)
    
    with col2:
        st.markdown("**🔥 Highest Over 2.5% Teams**")
        top_over25 = team_special_df.nlargest(5, 'Over 2.5 %')[['Team', 'Over 2.5 %', 'Avg Total Goals']]
        top_over25['Over 2.5 %'] = top_over25['Over 2.5 %'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(top_over25, width='stretch', hide_index=True)
    
    with col3:
        if 'HST' in df.columns:
            st.markdown("**🎯 Top 5 SOT Teams**")
            top_sot = team_special_df.nlargest(5, 'Avg Total SOT')[['Team', 'Avg Total SOT', 'Over 10.5 SOT %']]
            top_sot['Over 10.5 SOT %'] = top_sot['Over 10.5 SOT %'].apply(lambda x: f"{x:.1f}%" if x > 0 else "N/A")
            st.dataframe(top_sot, width='stretch', hide_index=True)
        else:
            st.markdown("**📊 No SOT Data**")
            st.info("Shots on target data not available")
    
    with col4:
        if 'HC' in df.columns:
            st.markdown("**🚩 Top 5 Corners Teams**")
            top_corners = team_special_df.nlargest(5, 'Avg Total Corners')[['Team', 'Avg Total Corners', 'Over 10.5 Corners %']]
            top_corners['Over 10.5 Corners %'] = top_corners['Over 10.5 Corners %'].apply(lambda x: f"{x:.1f}%" if x > 0 else "N/A")
            st.dataframe(top_corners, width='stretch', hide_index=True)
        else:
            st.markdown("**📊 No Corners Data**")
            st.info("Corners data not available")
    
    st.markdown("---")
    
    # Distribution Charts
    st.markdown("### 📊 Market Distributions")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚽ Goals Distribution")
        
        goals_dist = df['TotalGoals'].value_counts().sort_index()
        fig = go.Figure(data=[
            go.Bar(x=goals_dist.index, y=goals_dist.values,
                   marker_color='rgb(102, 126, 234)',
                   text=goals_dist.values,
                   textposition='auto')
        ])
        fig.update_layout(
            xaxis_title="Total Goals",
            yaxis_title="Frequency",
            height=300,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, width='stretch')
        
        # Goals market performance
        goal_lines = [1.5, 2.5, 3.5, 4.5]
        over_pct = [(df['TotalGoals'] > line).sum() / len(df) * 100 for line in goal_lines]
        
        market_df = pd.DataFrame({
            'Line': [f'Over {line}' for line in goal_lines],
            'Hit Rate (%)': over_pct,
            'Under Rate (%)': [100 - pct for pct in over_pct]
        })
        
        st.dataframe(market_df.style.background_gradient(subset=['Hit Rate (%)'], cmap='RdYlGn'),
                    width='stretch', hide_index=True)
    
    with col2:
        st.markdown("#### 🎯 Shots on Target Distribution")
        
        if df['TotalSOT'].sum() > 0:
            # Create bins for SOT
            sot_bins = pd.cut(df['TotalSOT'], bins=[0, 6, 8, 10, 12, 14, 100])
            sot_counts = sot_bins.value_counts().sort_index()
            
            fig = go.Figure(data=[
                go.Bar(x=[str(x) for x in sot_counts.index], y=sot_counts.values,
                       marker_color='rgb(79, 172, 254)',
                       text=sot_counts.values,
                       textposition='auto')
            ])
            fig.update_layout(
                xaxis_title="Total SOT Range",
                yaxis_title="Frequency",
                height=300,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, width='stretch')
            
            # SOT market performance
            sot_lines = [8.5, 10.5, 12.5, 14.5]
            over_sot_pct = [(df['TotalSOT'] > line).sum() / len(df) * 100 for line in sot_lines]
            
            sot_market_df = pd.DataFrame({
                'Line': [f'Over {line}' for line in sot_lines],
                'Hit Rate (%)': over_sot_pct,
                'Under Rate (%)': [100 - pct for pct in over_sot_pct]
            })
            
            st.dataframe(sot_market_df.style.background_gradient(subset=['Hit Rate (%)'], cmap='RdYlGn'),
                        width='stretch', hide_index=True)
        else:
            st.warning("No shots on target data available")
    
    st.markdown("---")
    
    # Corners Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚩 Corners Distribution")
        
        if df['TotalCorners'].sum() > 0:
            # Create bins for corners
            corners_bins = pd.cut(df['TotalCorners'], bins=[0, 6, 8, 10, 12, 14, 100])
            corners_counts = corners_bins.value_counts().sort_index()
            
            fig = go.Figure(data=[
                go.Bar(x=[str(x) for x in corners_counts.index], y=corners_counts.values,
                       marker_color='rgb(72, 187, 120)',
                       text=corners_counts.values,
                       textposition='auto')
            ])
            fig.update_layout(
                xaxis_title="Total Corners Range",
                yaxis_title="Frequency",
                height=300,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, width='stretch')
            
            # Corners market performance
            corners_lines = [8.5, 10.5, 12.5, 14.5]
            over_corners_pct = [(df['TotalCorners'] > line).sum() / len(df) * 100 for line in corners_lines]
            
            corners_market_df = pd.DataFrame({
                'Line': [f'Over {line}' for line in corners_lines],
                'Hit Rate (%)': over_corners_pct,
                'Under Rate (%)': [100 - pct for pct in over_corners_pct]
            })
            
            st.dataframe(corners_market_df.style.background_gradient(subset=['Hit Rate (%)'], cmap='RdYlGn'),
                        width='stretch', hide_index=True)
        else:
            st.warning("No corners data available")
    
    with col2:
        st.markdown("#### 🔗 Market Correlation")
        
        # Correlation heatmap
        if df['TotalSOT'].sum() > 0 and df['TotalCorners'].sum() > 0:
            corr_data = df[['TotalGoals', 'TotalSOT', 'TotalCorners']].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=['Goals', 'SOT', 'Corners'],
                y=['Goals', 'SOT', 'Corners'],
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_data.values, 2),
                texttemplate='%{text}',
                textfont={"size": 16},
                colorbar=dict(title="Correlation")
            ))
            fig.update_layout(
                height=300,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, width='stretch')
            
            st.info("💡 **Insights:**\n\n"
                   f"- Goals ↔ SOT correlation: **{corr_data.loc['TotalGoals', 'TotalSOT']:.2f}**\n"
                   f"- Goals ↔ Corners correlation: **{corr_data.loc['TotalGoals', 'TotalCorners']:.2f}**\n"
                   f"- SOT ↔ Corners correlation: **{corr_data.loc['TotalSOT', 'TotalCorners']:.2f}**")
        else:
            st.warning("Insufficient data for correlation analysis")
    
    # Top Matches by Market
    st.markdown("---")
    st.markdown("### ⭐ Top Matches by Market")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**⚽ Highest Scoring Matches**")
        top_goals = df.nlargest(5, 'TotalGoals')[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'TotalGoals']]
        top_goals['Date'] = pd.to_datetime(top_goals['Date']).dt.strftime('%Y-%m-%d')
        top_goals['Score'] = top_goals['FTHG'].astype(str) + '-' + top_goals['FTAG'].astype(str)
        top_goals['Match'] = top_goals['HomeTeam'] + ' vs ' + top_goals['AwayTeam']
        st.dataframe(top_goals[['Date', 'Match', 'Score', 'TotalGoals']], width='stretch', hide_index=True)
    
    with col2:
        if df['TotalSOT'].sum() > 0:
            st.markdown("**🎯 Most Shots on Target**")
            top_sot = df.nlargest(5, 'TotalSOT')[['Date', 'HomeTeam', 'AwayTeam', 'HST', 'AST', 'TotalSOT']]
            top_sot['Date'] = pd.to_datetime(top_sot['Date']).dt.strftime('%Y-%m-%d')
            top_sot['SOT'] = top_sot['HST'].astype(str) + '-' + top_sot['AST'].astype(str)
            top_sot['Match'] = top_sot['HomeTeam'] + ' vs ' + top_sot['AwayTeam']
            st.dataframe(top_sot[['Date', 'Match', 'SOT', 'TotalSOT']], width='stretch', hide_index=True)
        else:
            st.warning("No SOT data")
    
    with col3:
        if df['TotalCorners'].sum() > 0:
            st.markdown("**🚩 Most Corners**")
            top_corners = df.nlargest(5, 'TotalCorners')[['Date', 'HomeTeam', 'AwayTeam', 'HC', 'AC', 'TotalCorners']]
            top_corners['Date'] = pd.to_datetime(top_corners['Date']).dt.strftime('%Y-%m-%d')
            top_corners['Corners'] = top_corners['HC'].astype(str) + '-' + top_corners['AC'].astype(str)
            top_corners['Match'] = top_corners['HomeTeam'] + ' vs ' + top_corners['AwayTeam']
            st.dataframe(top_corners[['Date', 'Match', 'Corners', 'TotalCorners']], width='stretch', hide_index=True)
        else:
            st.warning("No corners data")
    
    # Handle download buttons
    if download_clicked:
        try:
            with st.spinner("Generating Excel report..."):
                excel_file = generate_excel_special_markets(df)
                
                st.download_button(
                    label="⬇️ Download Excel",
                    data=excel_file,
                    file_name=f"special_markets_{league_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="excel_download"
                )
        except Exception as e:
            st.error(f"Error generating Excel file: {str(e)}")
            st.info("Please make sure xlsxwriter is installed. Run: pip install xlsxwriter")
    
    if csv_clicked:
        summary_df = generate_special_markets_summary(df)
        csv = summary_df.to_csv(index=False)
        
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"special_markets_summary_{league_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="csv_download"
        )

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray;'>
    <p>⚽ Professional Football Betting Model | {league_name}</p>
    <p>📊 Statistical Model using Team Strength, Form & Expected Goals</p>
    <p>⚠️ For educational purposes only. Always gamble responsibly.</p>
</div>
""", unsafe_allow_html=True)
