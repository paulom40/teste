import streamlit as st
import pandas as pd
import numpy as np
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
st.markdown("## 📥 LOADING DATA FROM GITHUB")

@st.cache_data
def load_data():
    """Load both ATP and WTA data from GitHub"""
    
    # URLs to GitHub raw files
    wta_url = "https://github.com/paulom40/teste/raw/main/wta_data.xlsx"
    atp_url = "https://github.com/paulom40/teste/raw/main/atp_data.xlsx"
    
    dfs = []
    
    # Load WTA data
    try:
        response = requests.get(wta_url, timeout=15)
        response.raise_for_status()
        wta_df = pd.read_excel(BytesIO(response.content))
        wta_df['Tour'] = 'WTA'
        st.success("✅ WTA data loaded")
        dfs.append(wta_df)
    except Exception as e:
        st.error(f"❌ Could not load WTA data: {str(e)}")
    
    # Load ATP data
    try:
        response = requests.get(atp_url, timeout=15)
        response.raise_for_status()
        atp_df = pd.read_excel(BytesIO(response.content))
        atp_df['Tour'] = 'ATP'
        st.success("✅ ATP data loaded")
        dfs.append(atp_df)
    except Exception as e:
        st.error(f"❌ Could not load ATP data: {str(e)}")
    
    if not dfs:
        st.error("❌ Could not load any data from GitHub")
        return None
    
    # Combine datasets
    df = pd.concat(dfs, ignore_index=True)
    return df

# Load data
df = load_data()

if df is None:
    st.stop()

st.info(f"📊 Total matches in database: {len(df)}")
if 'Tour' in df.columns:
    wta_count = len(df[df['Tour'] == 'WTA'])
    atp_count = len(df[df['Tour'] == 'ATP'])
    st.info(f"✅ WTA: {wta_count:,} matches | ATP: {atp_count:,} matches")

st.markdown("---")

def calculate_total_games(row):
    """Calculate total games in match"""
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
with st.spinner("Analyzing all matches..."):
    df_analysis = df.copy()
    df_analysis['Total_Games'] = df_analysis.apply(calculate_total_games, axis=1)
    df_analysis['Competitiveness'] = df_analysis.apply(predict_competitiveness, axis=1)
    df_analysis = df_analysis.dropna(subset=['Total_Games'])

# Filter for competitive matches
competitive_matches = df_analysis[
    (df_analysis['Total_Games'] >= 20) & 
    (df_analysis['Total_Games'] <= 26) &
    (df_analysis['Competitiveness'] >= 0.5)
].copy()

competitive_matches = competitive_matches.sort_values('Competitiveness', ascending=False)

st.markdown("## 🎯 COMPETITIVE MATCHES ANALYSIS")
st.markdown(f"Found **{len(competitive_matches):,} matches** in the 20-26 games range with high competitiveness (50%+)")
st.markdown("---")

# DISPLAY OPTIONS
st.markdown("### 🔍 FILTER & ANALYZE")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    min_games = st.slider("Minimum Games", 20, 26, 21, key="min_games")

with col2:
    max_games = st.slider("Maximum Games", 20, 26, 23, key="max_games")

with col3:
    min_competitiveness = st.slider("Min Competitiveness %", 0, 100, 50, 5, key="comp")

with col4:
    tour_filter = st.multiselect("Tours", ['ATP', 'WTA'], default=['ATP', 'WTA'], key="tour")

with col5:
    st.markdown("**📅 Date**")
    selected_date = st.date_input("Select match date", value=datetime.now().date(), key="date", label_visibility="collapsed")

# Filter based on selections
filtered_matches = competitive_matches[
    (competitive_matches['Total_Games'] >= min_games) &
    (competitive_matches['Total_Games'] <= max_games) &
    (competitive_matches['Competitiveness'] >= min_competitiveness/100) &
    (competitive_matches['Tour'].isin(tour_filter))
].copy()

# Apply date filter if date column exists
date_filtered = False
if 'Date' in filtered_matches.columns:
    try:
        filtered_matches['Date_parsed'] = pd.to_datetime(filtered_matches['Date'], errors='coerce')
        filtered_matches_by_date = filtered_matches[filtered_matches['Date_parsed'].dt.date == selected_date]
        
        if len(filtered_matches_by_date) > 0:
            filtered_matches = filtered_matches_by_date
            date_filtered = True
            st.info(f"📅 Showing matches from {selected_date.strftime('%d/%m/%Y')}")
        else:
            st.warning(f"⚠️ No matches found for {selected_date.strftime('%d/%m/%Y')}. Showing all matches matching other criteria.")
    except Exception as e:
        st.info("ℹ️ Could not filter by date. Showing all matches.")
else:
    st.info("ℹ️ Date column not found in data. Showing all matches.")

st.markdown(f"### 📊 Showing {len(filtered_matches):,} matches matching criteria")

# DISPLAY MATCHES
if len(filtered_matches) > 0:
    # Get top matches
    max_show = min(100, len(filtered_matches))
    top_n = st.slider("Show top N matches", 5, max_show, min(25, max_show), key="top_n")
    top_matches = filtered_matches.head(top_n)
    
    # Create display dataframe
    display_data = []
    for idx, (_, match) in enumerate(top_matches.iterrows(), 1):
        w1 = int(match.get('W1', 0)) if pd.notna(match.get('W1')) else 0
        l1 = int(match.get('L1', 0)) if pd.notna(match.get('L1')) else 0
        w2 = int(match.get('W2', 0)) if pd.notna(match.get('W2')) else 0
        l2 = int(match.get('L2', 0)) if pd.notna(match.get('L2')) else 0
        
        display_data.append({
            '🏆': match['Tour'],
            'Rank': idx,
            'Player 1': match['Winner'],
            'Player 2': match['Loser'],
            'Surface': match['Surface'],
            'Score': f"{w1}-{l1} {w2}-{l2}",
            'Games': int(match['Total_Games']),
            'Competitiveness': f"{match['Competitiveness']*100:.1f}%",
            'Date': str(match.get('Date', 'N/A'))
        })
    
    display_df = pd.DataFrame(display_data)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("## 📥 GENERATE DETAILED REPORT")
    
    if st.button("📥 Generate Full Report (HTML)", use_container_width=True, key="generate_report"):
        with st.spinner("Generating HTML report..."):
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
                    .content {{
                        padding: 40px;
                    }}
                    .summary {{
                        background: #e3f2fd;
                        border-left: 4px solid #667eea;
                        padding: 20px;
                        margin-bottom: 30px;
                        border-radius: 5px;
                    }}
                    .match-card {{
                        background: white;
                        border: 3px solid;
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
                        padding: 8px 16px;
                        border-radius: 20px;
                        font-weight: bold;
                        color: white;
                        margin-bottom: 10px;
                        font-size: 0.9em;
                    }}
                    .tour-badge.atp {{
                        background: #4CAF50;
                    }}
                    .tour-badge.wta {{
                        background: #FF6B6B;
                    }}
                    .players {{
                        font-size: 1.3em;
                        font-weight: bold;
                        color: #333;
                        margin: 10px 0;
                    }}
                    .match-stats {{
                        display: grid;
                        grid-template-columns: 1fr 1fr 1fr 1fr;
                        gap: 15px;
                        margin: 20px 0;
                    }}
                    .stat {{
                        background: #f0f0f0;
                        padding: 15px;
                        border-radius: 5px;
                        text-align: center;
                    }}
                    .stat-value {{
                        font-size: 1.5em;
                        font-weight: bold;
                        color: #667eea;
                        margin-top: 5px;
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
                    table tr:nth-child(even) {{
                        background: #f9f9f9;
                    }}
                    .footer {{
                        background: #f9f9f9;
                        padding: 20px;
                        text-align: center;
                        border-top: 1px solid #eee;
                        color: #666;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🎾 ATP/WTA COMPETITIVE MATCHES REPORT</h1>
                        <p>Most Competitive Matches (20-26 Games Range)</p>
                    </div>
                    
                    <div class="content">
                        <div class="summary">
                            <h2 style="margin-top: 0; color: #667eea;">📊 Report Summary</h2>
                            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                            <p><strong>Total Matches Analyzed:</strong> {len(df_analysis):,}</p>
                            <p><strong>Competitive Matches Found:</strong> {len(competitive_matches):,}</p>
                            <p><strong>Filtered Results:</strong> {len(filtered_matches):,}</p>
                            <p><strong>Games Range:</strong> {min_games}-{max_games}</p>
                            <p><strong>Min Competitiveness:</strong> {min_competitiveness}%</p>
                            <p><strong>Tours:</strong> {', '.join(tour_filter)}</p>
                            <p><strong>Date Selected:</strong> {selected_date.strftime("%d/%m/%Y")}</p>
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
                                <th>Games</th>
                                <th>Competitiveness</th>
                                <th>Date</th>
                            </tr>
            """
            
            for idx, (_, match) in enumerate(top_matches.iterrows(), 1):
                tour = match.get('Tour', 'Unknown')
                w1 = int(match.get('W1', 0)) if pd.notna(match.get('W1')) else 0
                l1 = int(match.get('L1', 0)) if pd.notna(match.get('L1')) else 0
                w2 = int(match.get('W2', 0)) if pd.notna(match.get('W2')) else 0
                l2 = int(match.get('L2', 0)) if pd.notna(match.get('L2')) else 0
                
                html_content += f"""
                            <tr>
                                <td><strong>{tour}</strong></td>
                                <td>#{idx}</td>
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
                badge_class = f"tour-badge {tour.lower()}"
                w1 = int(match.get('W1', 0)) if pd.notna(match.get('W1')) else 0
                l1 = int(match.get('L1', 0)) if pd.notna(match.get('L1')) else 0
                w2 = int(match.get('W2', 0)) if pd.notna(match.get('W2')) else 0
                l2 = int(match.get('L2', 0)) if pd.notna(match.get('L2')) else 0
                w3 = int(match.get('W3', 0)) if pd.notna(match.get('W3')) else 0
                l3 = int(match.get('L3', 0)) if pd.notna(match.get('L3')) else 0
                
                html_content += f"""
                        <div class="{card_class}">
                            <span class="{badge_class}">{tour}</span>
                            <div style="color: #667eea; font-weight: bold; margin: 10px 0;">Match #{idx}</div>
                            
                            <div class="players">{match.get('Winner', 'N/A')} vs {match.get('Loser', 'N/A')}</div>
                            
                            <div class="match-stats">
                                <div class="stat">
                                    <div>Set 1</div>
                                    <div class="stat-value">{w1}-{l1}</div>
                                </div>
                                <div class="stat">
                                    <div>Set 2</div>
                                    <div class="stat-value">{w2}-{l2}</div>
                                </div>
                                <div class="stat">
                                    <div>Set 3</div>
                                    <div class="stat-value">{'N/A' if w3 == 0 else f'{w3}-{l3}'}</div>
                                </div>
                                <div class="stat">
                                    <div>Competitiveness</div>
                                    <div class="stat-value">{match['Competitiveness']*100:.1f}%</div>
                                </div>
                            </div>
                            
                            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee;">
                                <p><strong>Total Games:</strong> {int(match['Total_Games'])}</p>
                                <p><strong>Surface:</strong> {match.get('Surface', 'N/A')}</p>
                                <p><strong>Date:</strong> {match.get('Date', 'N/A')}</p>
                            </div>
                        </div>
                """
            
            html_content += """
                    </div>
                    
                    <div class="footer">
                        <p><strong>ATP/WTA Competitive Matches Analyzer</strong></p>
                        <p>Professional Tennis Match Analysis Report</p>
                        <p>🟢 ATP Matches | 🔴 WTA Matches</p>
                    </div>
                </div>
            </body>
            </html>
            """
        
        st.download_button(
            label="📥 Download HTML Report",
            data=html_content,
            file_name=f"ATP_WTA_Competitive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )
        st.success("✅ Report ready to download!")

else:
    st.warning("No matches found with selected criteria. Try adjusting filters.")

st.markdown("---")
st.markdown("### 📊 STATISTICS")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Analyzed", f"{len(df_analysis):,}")
col2.metric("Competitive Found", f"{len(competitive_matches):,}")
col3.metric("Avg Games", f"{competitive_matches['Total_Games'].mean():.1f}")
col4.metric("Avg Competitiveness", f"{competitive_matches['Competitiveness'].mean()*100:.1f}%")

if 'Tour' in competitive_matches.columns:
    atp_count = len(competitive_matches[competitive_matches['Tour'] == 'ATP'])
    col5.metric("ATP Competitive", f"{atp_count:,}")
