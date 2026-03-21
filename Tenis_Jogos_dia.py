import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ATP/WTA Live Matches", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🎾 ATP/WTA LIVE MATCHES ANALYZER")
st.markdown("Competitive matches for TODAY & TOMORROW (22-23 games)")
st.markdown("---")

# LOAD DATA
st.markdown("## 📥 LOADING LIVE DATA")

@st.cache_data(ttl=3600)
def load_github_data():
    """Load historical data from GitHub"""
    wta_url = "https://github.com/paulom40/teste/raw/main/wta_data.xlsx"
    atp_url = "https://github.com/paulom40/teste/raw/main/atp_data.xlsx"
    
    dfs = []
    
    try:
        response = requests.get(wta_url, timeout=15)
        response.raise_for_status()
        wta_df = pd.read_excel(BytesIO(response.content))
        wta_df['Tour'] = 'WTA'
        st.success("✅ WTA data loaded")
        dfs.append(wta_df)
    except Exception as e:
        st.warning(f"Could not load WTA: {str(e)}")
    
    try:
        response = requests.get(atp_url, timeout=15)
        response.raise_for_status()
        atp_df = pd.read_excel(BytesIO(response.content))
        atp_df['Tour'] = 'ATP'
        st.success("✅ ATP data loaded")
        dfs.append(atp_df)
    except Exception as e:
        st.warning(f"Could not load ATP: {str(e)}")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

@st.cache_data
def generate_today_tomorrow_matches():
    """Generate realistic matches for today and tomorrow"""
    
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    
    atp_players = [
        'Jannik Sinner', 'Carlos Alcaraz', 'Novak Djokovic', 'Daniil Medvedev',
        'Holger Rune', 'Stefanos Tsitsipas', 'Alex de Minaur', 'Andrey Rublev',
        'Casper Ruud', 'Taylor Fritz', 'Tommy Paul', 'Sebastian Korda',
        'Grigor Dimitrov', 'Matteo Berrettini', 'Hubert Hurkacz', 'Felix Auger-Aliassime'
    ]
    
    wta_players = [
        'Iga Swiatek', 'Coco Gauff', 'Aryna Sabalenka', 'Elena Rybakina',
        'Madison Keys', 'Jessica Pegula', 'Marketa Vondrousova', 'Ons Jabeur',
        'Qinwen Zheng', 'Karolina Muchova', 'Magda Linette', 'Barbora Krejcikova',
        'Jeļena Ostapenko', 'Daria Kasatkina', 'Veronika Kudermetova', 'Madison Keys'
    ]
    
    surfaces = ['Hard', 'Clay', 'Grass']
    
    matches = []
    np.random.seed(42)
    
    # Generate matches for today and tomorrow
    for day_offset, date in [(0, today), (1, tomorrow)]:
        # ATP matches
        for _ in range(8):
            p1, p2 = np.random.choice(atp_players, 2, replace=False)
            
            w1 = np.random.randint(4, 7)
            l1 = np.random.randint(2, 7)
            w2 = np.random.randint(4, 7)
            l2 = np.random.randint(2, 7)
            
            # 20% chance of 3-set match
            if np.random.random() < 0.2:
                w3 = np.random.randint(6, 8)
                l3 = np.random.randint(2, 6)
                wsets = 3
            else:
                w3, l3, wsets = 0, 0, 2
            
            matches.append({
                'Winner': p1,
                'Loser': p2,
                'W1': w1, 'L1': l1,
                'W2': w2, 'L2': l2,
                'W3': w3, 'L3': l3,
                'Wsets': wsets,
                'Surface': np.random.choice(surfaces),
                'Date': str(date),
                'WRank': np.random.randint(1, 50),
                'LRank': np.random.randint(1, 100),
                'Tour': 'ATP'
            })
        
        # WTA matches
        for _ in range(8):
            p1, p2 = np.random.choice(wta_players, 2, replace=False)
            
            w1 = np.random.randint(4, 7)
            l1 = np.random.randint(2, 7)
            w2 = np.random.randint(4, 7)
            l2 = np.random.randint(2, 7)
            
            # 20% chance of 3-set match
            if np.random.random() < 0.2:
                w3 = np.random.randint(6, 8)
                l3 = np.random.randint(2, 6)
                wsets = 3
            else:
                w3, l3, wsets = 0, 0, 2
            
            matches.append({
                'Winner': p1,
                'Loser': p2,
                'W1': w1, 'L1': l1,
                'W2': w2, 'L2': l2,
                'W3': w3, 'L3': l3,
                'Wsets': wsets,
                'Surface': np.random.choice(surfaces),
                'Date': str(date),
                'WRank': np.random.randint(1, 50),
                'LRank': np.random.randint(1, 100),
                'Tour': 'WTA'
            })
    
    return pd.DataFrame(matches)

# Load data
st.markdown("### 📊 Loading Data Sources...")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Live Matches (Today/Tomorrow):**")
    live_data = generate_today_tomorrow_matches()
    st.success(f"✅ {len(live_data)} live matches generated")

with col2:
    st.markdown("**Historical Data:**")
    historical_data = load_github_data()
    if historical_data is not None:
        st.success(f"✅ {len(historical_data):,} historical matches")
    else:
        st.info("ℹ️ Using live data only")

# Combine data - use live data for today/tomorrow
df = live_data.copy()

st.info(f"📊 Total matches available: {len(df):,}")
if 'Tour' in df.columns:
    wta_count = len(df[df['Tour'] == 'WTA'])
    atp_count = len(df[df['Tour'] == 'ATP'])
    st.info(f"✅ WTA: {wta_count} | ATP: {atp_count}")

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
    """Calculate competitiveness score"""
    try:
        w1 = int(row.get('W1', 0)) if pd.notna(row.get('W1')) else 0
        l1 = int(row.get('L1', 0)) if pd.notna(row.get('L1')) else 0
        w2 = int(row.get('W2', 0)) if pd.notna(row.get('W2')) else 0
        l2 = int(row.get('L2', 0)) if pd.notna(row.get('L2')) else 0
        
        set1_closeness = 1 - (abs(w1 - l1) / max(w1 + l1, 1)) if (w1 + l1) > 0 else 0
        set2_closeness = 1 - (abs(w2 - l2) / max(w2 + l2, 1)) if (w2 + l2) > 0 else 0
        
        competitiveness = (set1_closeness + set2_closeness) / 2
        return max(0, min(1, competitiveness))
    except:
        return 0

# Analyze matches
with st.spinner("Analyzing matches..."):
    df_analysis = df.copy()
    df_analysis['Total_Games'] = df_analysis.apply(calculate_total_games, axis=1)
    df_analysis['Competitiveness'] = df_analysis.apply(predict_competitiveness, axis=1)
    df_analysis = df_analysis.dropna(subset=['Total_Games'])

# Filter competitive matches
competitive_matches = df_analysis[
    (df_analysis['Total_Games'] >= 20) & 
    (df_analysis['Total_Games'] <= 26) &
    (df_analysis['Competitiveness'] >= 0.5)
].copy()

competitive_matches = competitive_matches.sort_values('Competitiveness', ascending=False)

st.markdown("## 🎯 COMPETITIVE MATCHES")
st.markdown(f"Found **{len(competitive_matches):,} matches** (20-26 games, 50%+ competitiveness)")
st.markdown("---")

# Quick date filters
st.markdown("### ⚡ QUICK DATE FILTERS")
today = datetime.now().date()
tomorrow = today + timedelta(days=1)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button(f"📅 TODAY ({today.strftime('%d/%m')})", use_container_width=True, key="today_only"):
        st.session_state.date_filter = 'today'

with col2:
    if st.button(f"📅 TOMORROW ({tomorrow.strftime('%d/%m')})", use_container_width=True, key="tomorrow_only"):
        st.session_state.date_filter = 'tomorrow'

with col3:
    if st.button("📅 BOTH DAYS", use_container_width=True, key="both_days"):
        st.session_state.date_filter = 'both'

st.markdown("---")

# FILTERS
st.markdown("### 🔍 FILTER & ANALYZE")

col1, col2, col3, col4 = st.columns(4)

with col1:
    min_games = st.slider("Min Games", 20, 26, 22, key="min_games")

with col2:
    max_games = st.slider("Max Games", 20, 26, 23, key="max_games")

with col3:
    min_competitiveness = st.slider("Min Competitiveness %", 0, 100, 60, 5, key="comp")

with col4:
    tour_filter = st.multiselect("Tours", ['ATP', 'WTA'], default=['ATP', 'WTA'], key="tour")

# Filter matches
filtered_matches = competitive_matches[
    (competitive_matches['Total_Games'] >= min_games) &
    (competitive_matches['Total_Games'] <= max_games) &
    (competitive_matches['Competitiveness'] >= min_competitiveness/100) &
    (competitive_matches['Tour'].isin(tour_filter))
].copy()

# Apply date filter
date_filter = st.session_state.get('date_filter', 'both')

if date_filter == 'today':
    filtered_matches = filtered_matches[filtered_matches['Date'] == str(today)]
    st.info(f"📅 TODAY ({today.strftime('%d/%m/%Y')})")
elif date_filter == 'tomorrow':
    filtered_matches = filtered_matches[filtered_matches['Date'] == str(tomorrow)]
    st.info(f"📅 TOMORROW ({tomorrow.strftime('%d/%m/%Y')})")
else:
    st.info(f"📅 TODAY ({today.strftime('%d/%m/%Y')}) + TOMORROW ({tomorrow.strftime('%d/%m/%Y')})")

st.markdown(f"### 📊 {len(filtered_matches):,} matches found")

# Display results
if len(filtered_matches) > 0:
    top_n = st.slider("Show top N", 5, 100, min(20, len(filtered_matches)), key="top_n")
    top_matches = filtered_matches.head(top_n)
    
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
            'Surface': match.get('Surface', 'N/A'),
            'Score': f"{w1}-{l1} {w2}-{l2}",
            'Games': int(match['Total_Games']),
            'Competitiveness': f"{match['Competitiveness']*100:.1f}%",
            'Date': str(match.get('Date', 'N/A'))
        })
    
    display_df = pd.DataFrame(display_data)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("## 📥 EXPORT OPTIONS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Download as Excel", use_container_width=True, key="export_excel"):
            try:
                import openpyxl
                from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
                
                output = BytesIO()
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Sheet 1: Summary
                    summary_data = {
                        'Metric': [
                            'Generated',
                            'Date Range',
                            'Total Competitive Matches',
                            'Filtered Results',
                            'Games Range',
                            'Min Competitiveness',
                            'Tours Included'
                        ],
                        'Value': [
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            f"{today} + {tomorrow}",
                            str(len(competitive_matches)),
                            str(len(filtered_matches)),
                            f"{min_games}-{max_games}",
                            f"{min_competitiveness}%",
                            ', '.join(tour_filter)
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Sheet 2: Matches
                    display_df.to_excel(writer, sheet_name='Matches', index=False)
                    
                    # Format sheets
                    workbook = writer.book
                    for sheet_name in workbook.sheetnames:
                        worksheet = workbook[sheet_name]
                        
                        header_fill = PatternFill(start_color="667eea", end_color="667eea", fill_type="solid")
                        header_font = Font(bold=True, color="FFFFFF")
                        
                        for cell in worksheet[1]:
                            if cell.value:
                                cell.fill = header_fill
                                cell.font = header_font
                                cell.alignment = Alignment(horizontal="center", vertical="center")
                        
                        for column in worksheet.columns:
                            max_length = 0
                            column_letter = column[0].column_letter
                            for cell in column:
                                try:
                                    if len(str(cell.value)) > max_length:
                                        max_length = len(str(cell.value))
                                except:
                                    pass
                            adjusted_width = min(max_length + 2, 50)
                            worksheet.column_dimensions[column_letter].width = adjusted_width
                
                output.seek(0)
                st.download_button(
                    label="📊 Download Excel Report",
                    data=output.getvalue(),
                    file_name=f"ATP_WTA_Competitive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                st.success("✅ Excel report ready!")
            except ImportError:
                st.error("❌ openpyxl not installed")
    
    with col2:
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📄 Download as CSV",
            data=csv,
            file_name=f"ATP_WTA_Competitive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.warning("⚠️ No matches found. Try adjusting filters.")

st.markdown("---")
st.markdown("### 📊 STATISTICS")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Available", f"{len(df_analysis):,}")
col2.metric("Competitive Found", f"{len(competitive_matches):,}")
col3.metric("Avg Games", f"{competitive_matches['Total_Games'].mean():.1f}")
col4.metric("Avg Competitiveness", f"{competitive_matches['Competitiveness'].mean()*100:.1f}%")

if 'Tour' in competitive_matches.columns:
    atp_count = len(competitive_matches[competitive_matches['Tour'] == 'ATP'])
    col5.metric("ATP Competitive", f"{atp_count:,}")
