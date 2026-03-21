import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import requests
from io import BytesIO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ATP/WTA Competitive Matches", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🎾 ATP/WTA COMPETITIVE MATCHES ANALYZER")
st.markdown("Find the most competitive ATP & WTA matches expected to go 22-23 games")
st.markdown("---")

# LOAD DATA
st.markdown("## 📥 LOADING DATA")

@st.cache_data
def load_wta_data():
    try:
        url = "https://github.com/paulom40/teste/raw/main/wta_data.xlsx"
        response = requests.get(url, timeout=10)
        df = pd.read_excel(BytesIO(response.content))
        df['Tour'] = 'WTA'
        st.success("✅ WTA data loaded")
        return df
    except:
        st.warning("Could not load WTA data")
        return None

@st.cache_data
def load_atp_data():
    try:
        # Try multiple ATP data sources
        urls = [
            "https://github.com/JeffSackmann/tennis_atp/raw/master/atp_matches_2024.csv",
            "https://github.com/JeffSackmann/tennis_atp/raw/master/atp_matches_2023.csv",
            "https://github.com/JeffSackmann/tennis_atp/raw/master/atp_matches_2022.csv",
        ]
        
        dfs = []
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                df = pd.read_csv(BytesIO(response.content))
                df['Tour'] = 'ATP'
                dfs.append(df)
            except:
                continue
        
        if dfs:
            atp_df = pd.concat(dfs, ignore_index=True)
            st.success(f"✅ ATP data loaded ({len(atp_df)} matches)")
            return atp_df
        else:
            st.warning("Could not load ATP data from GitHub")
            return None
    except:
        st.warning("Could not load ATP data")
        return None

# Load both datasets
wta_df = load_wta_data()
atp_df = load_atp_data()

# Combine datasets
dfs_to_combine = []
if wta_df is not None:
    dfs_to_combine.append(wta_df)
if atp_df is not None:
    dfs_to_combine.append(atp_df)

if not dfs_to_combine:
    st.error("Could not load any data")
    st.stop()

# Merge datasets - handle different column names
if len(dfs_to_combine) == 2:
    # Standardize ATP columns to match WTA format
    atp_df = dfs_to_combine[1]
    atp_df = atp_df.rename(columns={
        'winner_name': 'Winner',
        'loser_name': 'Loser',
        'surface': 'Surface',
        'tourney_date': 'Date'
    })
    
    # Calculate games from score
    def extract_games_from_atp_score(score_str):
        try:
            sets = str(score_str).split()
            w1 = int(sets[0].split('-')[0]) if len(sets) > 0 else 0
            l1 = int(sets[0].split('-')[1]) if len(sets) > 0 else 0
            w2 = int(sets[1].split('-')[0]) if len(sets) > 1 else 0
            l2 = int(sets[1].split('-')[1]) if len(sets) > 1 else 0
            w3 = int(sets[2].split('-')[0]) if len(sets) > 2 else 0
            l3 = int(sets[2].split('-')[1]) if len(sets) > 2 else 0
            
            atp_df.loc[atp_df['score'] == score_str, 'W1'] = w1
            atp_df.loc[atp_df['score'] == score_str, 'L1'] = l1
            atp_df.loc[atp_df['score'] == score_str, 'W2'] = w2
            atp_df.loc[atp_df['score'] == score_str, 'L2'] = l2
            atp_df.loc[atp_df['score'] == score_str, 'W3'] = w3
            atp_df.loc[atp_df['score'] == score_str, 'L3'] = l3
        except:
            pass
    
    if 'score' in atp_df.columns:
        for score in atp_df['score'].unique():
            extract_games_from_atp_score(score)
    
    df = pd.concat(dfs_to_combine, ignore_index=True)
else:
    df = dfs_to_combine[0]

st.info(f"📊 Total matches in database: {len(df)}")
if 'Tour' in df.columns:
    wta_count = len(df[df['Tour'] == 'WTA'])
    atp_count = len(df[df['Tour'] == 'ATP'])
    st.info(f"WTA: {wta_count} matches | ATP: {atp_count} matches")

st.markdown("---")

def calculate_total_games(row):
    total = 0
    for i in range(1, 6):
        w = row.get(f'W{i}', 0)
        l = row.get(f'L{i}', 0)
        if pd.notna(w) and pd.notna(l) and w > 0 and l > 0:
            total += int(w) + int(l)
    return total if total > 0 else None

def predict_competitiveness(row):
    """Calculate competitiveness score (0-1, where 1 is most competitive)"""
    try:
        w1 = int(row.get('W1', 0)) if pd.notna(row.get('W1')) else 0
        l1 = int(row.get('L1', 0)) if pd.notna(row.get('L1')) else 0
        w2 = int(row.get('W2', 0)) if pd.notna(row.get('W2')) else 0
        l2 = int(row.get('L2', 0)) if pd.notna(row.get('L2')) else 0
        
        # Games should be close in both sets
        set1_closeness = 1 - (abs(w1 - l1) / max(w1 + l1, 1)) if (w1 + l1) > 0 else 0
        set2_closeness = 1 - (abs(w2 - l2) / max(w2 + l2, 1)) if (w2 + l2) > 0 else 0
        
        # Average closeness
        competitiveness = (set1_closeness + set2_closeness) / 2
        return max(0, min(1, competitiveness))
    except:
        return 0

# ANALYZE ALL MATCHES
df_analysis = df.copy()
df_analysis['Total_Games'] = df_analysis.apply(calculate_total_games, axis=1)
df_analysis['Competitiveness'] = df_analysis.apply(predict_competitiveness, axis=1)
df_analysis = df_analysis.dropna(subset=['Total_Games'])

# Filter for competitive matches around 22-23 games
competitive_matches = df_analysis[
    (df_analysis['Total_Games'] >= 20) & 
    (df_analysis['Total_Games'] <= 26) &
    (df_analysis['Competitiveness'] >= 0.5)
].copy()

competitive_matches = competitive_matches.sort_values('Competitiveness', ascending=False)

st.markdown("## 🎯 COMPETITIVE MATCHES ANALYSIS")
st.markdown(f"Found **{len(competitive_matches)} matches** in the 20-26 games range with high competitiveness")
st.markdown("---")

# DISPLAY OPTIONS
st.markdown("### 🔍 FILTER & ANALYZE")

col1, col2, col3, col4 = st.columns(4)

with col1:
    min_games = st.slider("Minimum Games", 20, 26, 21, key="min_games")

with col2:
    max_games = st.slider("Maximum Games", 20, 26, 23, key="max_games")

with col3:
    min_competitiveness = st.slider("Min Competitiveness", 0.0, 1.0, 0.5, 0.05, key="comp")

with col4:
    tour_filter = st.multiselect("Tours", ['ATP', 'WTA'], default=['ATP', 'WTA'], key="tour")

# Filter based on selections
filtered_matches = competitive_matches[
    (competitive_matches['Total_Games'] >= min_games) &
    (competitive_matches['Total_Games'] <= max_games) &
    (competitive_matches['Competitiveness'] >= min_competitiveness) &
    (competitive_matches['Tour'].isin(tour_filter))
].copy()

st.markdown(f"### 📊 Showing {len(filtered_matches)} matches matching criteria")

# DISPLAY MATCHES
if len(filtered_matches) > 0:
    # Get top matches
    top_n = st.slider("Show top N matches", 5, 100, 25, key="top_n")
    top_matches = filtered_matches.head(top_n)
    
    # Create display dataframe
    display_df = pd.DataFrame({
        'Tour': top_matches['Tour'].values,
        'Rank': range(1, len(top_matches) + 1),
        'Player 1': top_matches['Winner'].values,
        'Player 2': top_matches['Loser'].values,
        'Surface': top_matches['Surface'].values,
        'Score': [f"{int(row.get('W1', 0))}-{int(row.get('L1', 0))} {int(row.get('W2', 0))}-{int(row.get('L2', 0))}" 
                  if pd.notna(row.get('W1')) else 'N/A' for _, row in top_matches.iterrows()],
        'Total Games': top_matches['Total_Games'].values.astype(int),
        'Competitiveness': (top_matches['Competitiveness'] * 100).round(1).astype(str) + '%',
        'Date': top_matches['Date'].values
    })
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("## 📥 GENERATE DETAILED REPORT")
    
    if st.button("📥 Generate Full Report (HTML)", use_container_width=True, key="generate_report"):
        # Generate comprehensive HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ATP/WTA Competitive Matches Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px;
                    margin: 0;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 15px;
                    overflow: hidden;
                    box-shadow: 0 10px 50px rgba(0,0,0,0.3);
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    font-size: 1.1em;
                }}
                .content {{
                    padding: 40px;
                }}
                .filters {{
                    background: #f9f9f9;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    border-left: 5px solid #667eea;
                }}
                .filters h3 {{
                    color: #667eea;
                    margin-top: 0;
                }}
                .match-card {{
                    background: white;
                    border: 2px solid #667eea;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 10px;
                    page-break-inside: avoid;
                }}
                .match-card.atp {{
                    border-color: #4CAF50;
                }}
                .match-card.wta {{
                    border-color: #FF6B6B;
                }}
                .tour-badge {{
                    display: inline-block;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-weight: bold;
                    color: white;
                    font-size: 0.9em;
                    margin-bottom: 10px;
                }}
                .tour-badge.atp {{
                    background: #4CAF50;
                }}
                .tour-badge.wta {{
                    background: #FF6B6B;
                }}
                .match-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 15px;
                    margin-bottom: 15px;
                }}
                .match-rank {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #667eea;
                    background: #e3f2fd;
                    padding: 10px 15px;
                    border-radius: 5px;
                }}
                .players {{
                    font-size: 1.3em;
                    font-weight: bold;
                    color: #333;
                    flex: 1;
                    margin: 0 20px;
                }}
                .score {{
                    background: #764ba2;
                    color: white;
                    padding: 10px 15px;
                    border-radius: 5px;
                    font-weight: bold;
                    text-align: center;
                }}
                .match-stats {{
                    display: grid;
                    grid-template-columns: 1fr 1fr 1fr 1fr;
                    gap: 15px;
                }}
                .stat {{
                    background: #f0f0f0;
                    padding: 15px;
                    border-radius: 5px;
                    text-align: center;
                }}
                .stat-label {{
                    font-size: 0.9em;
                    color: #666;
                    margin-bottom: 5px;
                }}
                .stat-value {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #667eea;
                }}
                .footer {{
                    background: #f9f9f9;
                    padding: 20px;
                    text-align: center;
                    border-top: 1px solid #eee;
                    color: #666;
                    font-size: 0.9em;
                }}
                .summary {{
                    background: #e3f2fd;
                    border-left: 4px solid #667eea;
                    padding: 20px;
                    margin-bottom: 30px;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                table th {{
                    background: #667eea;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }}
                table td {{
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                }}
                table tr:hover {{
                    background: #f5f5f5;
                }}
                .atp-row {{
                    background: #f0f8f5;
                }}
                .wta-row {{
                    background: #fff5f5;
                }}
                @media print {{
                    body {{
                        background: white;
                    }}
                    .container {{
                        box-shadow: none;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎾 ATP/WTA COMPETITIVE MATCHES REPORT</h1>
                    <p>Most Competitive Matches Analysis (20-26 Games Range)</p>
                </div>
                
                <div class="content">
                    <div class="summary">
                        <h2 style="margin-top: 0; color: #667eea;">📊 Report Summary</h2>
                        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                        <p><strong>Total Matches Analyzed:</strong> {len(df_analysis)}</p>
                        <p><strong>Competitive Matches Found:</strong> {len(filtered_matches)}</p>
                        <p><strong>Games Range:</strong> {min_games}-{max_games}</p>
                        <p><strong>Min Competitiveness:</strong> {min_competitiveness*100:.0f}%</p>
                        <p><strong>Tours Included:</strong> {', '.join(tour_filter)}</p>
                    </div>
                    
                    <h2 style="color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px;">🏆 TOP {len(top_matches)} MOST COMPETITIVE MATCHES</h2>
                    
                    <table>
                        <tr>
                            <th>Tour</th>
                            <th>Rank</th>
                            <th>Player 1</th>
                            <th>Player 2</th>
                            <th>Surface</th>
                            <th>Score</th>
                            <th>Total Games</th>
                            <th>Competitiveness</th>
                            <th>Date</th>
                        </tr>
        """
        
        for idx, (_, match) in enumerate(top_matches.iterrows(), 1):
            tour = match.get('Tour', 'Unknown')
            row_class = f"{tour.lower()}-row"
            w1 = int(match.get('W1', 0)) if pd.notna(match.get('W1')) else 0
            l1 = int(match.get('L1', 0)) if pd.notna(match.get('L1')) else 0
            w2 = int(match.get('W2', 0)) if pd.notna(match.get('W2')) else 0
            l2 = int(match.get('L2', 0)) if pd.notna(match.get('L2')) else 0
            
            html_content += f"""
                        <tr class="{row_class}">
                            <td><strong>{tour}</strong></td>
                            <td><strong>#{idx}</strong></td>
                            <td>{match.get('Winner', 'N/A')}</td>
                            <td>{match.get('Loser', 'N/A')}</td>
                            <td>{match.get('Surface', 'N/A')}</td>
                            <td>{w1}-{l1} {w2}-{l2}</td>
                            <td><strong>{int(match['Total_Games'])}</strong></td>
                            <td>{match['Competitiveness']*100:.1f}%</td>
                            <td>{match.get('Date', 'N/A')}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                    
                    <h2 style="color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px;">📈 DETAILED MATCH ANALYSIS</h2>
        """
        
        for idx, (_, match) in enumerate(top_matches.iterrows(), 1):
            tour = match.get('Tour', 'Unknown')
            card_class = f"match-card {tour.lower()}"
            w1 = int(match.get('W1', 0)) if pd.notna(match.get('W1')) else 0
            l1 = int(match.get('L1', 0)) if pd.notna(match.get('L1')) else 0
            w2 = int(match.get('W2', 0)) if pd.notna(match.get('W2')) else 0
            l2 = int(match.get('L2', 0)) if pd.notna(match.get('L2')) else 0
            w3 = int(match.get('W3', 0)) if pd.notna(match.get('W3')) else 0
            l3 = int(match.get('L3', 0)) if pd.notna(match.get('L3')) else 0
            badge_class = f"tour-badge {tour.lower()}"
            
            html_content += f"""
                    <div class="{card_class}">
                        <span class="{badge_class}">{tour}</span>
                        
                        <div class="match-header">
                            <div class="match-rank">#{idx}</div>
                            <div class="players">{match.get('Winner', 'N/A')} vs {match.get('Loser', 'N/A')}</div>
                            <div class="score">{int(match['Total_Games'])} Games</div>
                        </div>
                        
                        <div class="match-stats">
                            <div class="stat">
                                <div class="stat-label">Set 1</div>
                                <div class="stat-value">{w1}-{l1}</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Set 2</div>
                                <div class="stat-value">{w2}-{l2}</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Set 3</div>
                                <div class="stat-value">{'N/A' if w3 == 0 else f'{w3}-{l3}'}</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Competitiveness</div>
                                <div class="stat-value">{match['Competitiveness']*100:.1f}%</div>
                            </div>
                        </div>
                        
                        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee;">
                            <p><strong>Surface:</strong> {match.get('Surface', 'N/A')}</p>
                            <p><strong>Date:</strong> {match.get('Date', 'N/A')}</p>
                        </div>
                    </div>
            """
        
        html_content += """
                </div>
                
                <div class="footer">
                    <p><strong>ATP/WTA Competitive Matches Analyzer</strong></p>
                    <p>Report generated automatically - All statistics based on match data</p>
                    <p>These matches show high competitiveness in the 20-26 games range</p>
                    <p>🟢 ATP | 🔴 WTA</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        st.download_button(
            label="📥 Download HTML Report",
            data=html_content,
            file_name=f"ATP_WTA_Competitive_Matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )
        st.success("✅ Report ready to download!")

else:
    st.warning("No matches found with selected criteria. Try adjusting filters.")

st.markdown("---")
st.markdown("### 📊 STATISTICS")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Matches", len(df_analysis))
col2.metric("Competitive Matches", len(competitive_matches))
col3.metric("Avg Games", f"{competitive_matches['Total_Games'].mean():.1f}")
col4.metric("Avg Competitiveness", f"{competitive_matches['Competitiveness'].mean()*100:.1f}%")

if 'Tour' in competitive_matches.columns:
    atp_comp = len(competitive_matches[competitive_matches['Tour'] == 'ATP'])
    col5.metric("ATP Competitive", atp_comp)
