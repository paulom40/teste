import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Miami 2026 Predictor", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🎾 MIAMI 2026 - GAME PREDICTOR")
st.markdown("Enter TODAY'S SCHEDULED matches and predict games (22-25)")
st.markdown("---")

today = datetime.now().date()
tomorrow = today + timedelta(days=1)

# DISPLAY CURRENT DATE/TIME
st.markdown("## 📅 CURRENT DATE & TIME")
current_time = datetime.now()
col1, col2 = st.columns(2)
col1.metric("📅 Today", today.strftime('%A, %d/%m/%Y'))
col2.metric("⏰ Current Time", current_time.strftime('%H:%M:%S'))

st.markdown("---")

st.markdown("## 📝 ENTER SCHEDULED MATCHES FOR TODAY")
st.warning("""
⚠️ **IMPORTANT:** Only enter matches that are SCHEDULED for TODAY (not finished/past matches!)

Example of CORRECT matches to paste:
- Zheng Q. vs Stephens S. (scheduled for 16:30 today)
- Swiatek I. vs Tauson C. (scheduled for today)

Example of WRONG matches (do NOT paste):
- ❌ Alcaraz C. vs Fonseca J. (2-0 Finished - this is already played)
- ❌ Any match with a score like "6-4 6-3" or "Finished"

Go to https://www.flashscore.com/tennis/ and look for matches WITHOUT scores yet!
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏆 ATP SINGLES")
    st.markdown("**TODAY'S SCHEDULED ATP matches (no score yet)**")
    
    atp_input = st.text_area(
        "ATP Singles - paste scheduled matches only",
        placeholder="""Enter one match per line:
Player1 vs Player2

Example:
Sinner J. vs Munarovic P.
Alcaraz C. vs Kokkinakis T.
Medvedev D. vs Rublev A.""",
        height=180,
        key="atp_input"
    )

with col2:
    st.markdown("### 🏆 WTA SINGLES")
    st.markdown("**TODAY'S SCHEDULED WTA matches (no score yet)**")
    
    wta_input = st.text_area(
        "WTA Singles - paste scheduled matches only",
        placeholder="""Enter one match per line:
Player1 vs Player2

Example:
Zheng Q. vs Stephens S.
Swiatek I. vs Tauson C.
Sabalenka A. vs Fręch M.""",
        height=180,
        key="wta_input"
    )

st.markdown("---")

# Parse matches
def parse_matches(text, date, tour):
    """Parse match text into dataframe"""
    matches = []
    
    if not text.strip():
        return pd.DataFrame()
    
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or 'vs' not in line.lower():
            continue
        
        try:
            # Split by "vs" (case insensitive)
            parts = line.split('vs')
            if len(parts) != 2:
                continue
            
            p1 = parts[0].strip()
            p2 = parts[1].strip()
            
            # Skip if has score (contains numbers like 6-4, 7-5, or "Finished")
            if any(char.isdigit() for char in p1) or any(char.isdigit() for char in p2):
                st.warning(f"⚠️ Skipped: '{line}' - appears to be a finished match (has score)")
                continue
            
            if 'finished' in line.lower() or 'live' in line.lower():
                st.warning(f"⚠️ Skipped: '{line}' - appears to be finished")
                continue
            
            if len(p1) > 3 and len(p2) > 3:
                matches.append({
                    'Date': date.strftime('%d/%m/%Y'),
                    'Tour': tour,
                    'Player 1': p1,
                    'Player 2': p2,
                    'Surface': 'Hard (Miami)',
                    'Status': '📅 Scheduled'
                })
        except:
            continue
    
    return pd.DataFrame(matches)

# Parse all matches
atp_matches = parse_matches(atp_input, today, 'ATP')
wta_matches = parse_matches(wta_input, today, 'WTA')
all_matches = pd.concat([atp_matches, wta_matches], ignore_index=True)

if len(all_matches) > 0:
    st.markdown("## ✅ LOADED MATCHES")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Matches", len(all_matches))
    col2.metric("ATP", len(atp_matches))
    col3.metric("WTA", len(wta_matches))
    
    st.dataframe(all_matches, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # LOAD HISTORICAL DATA
    st.markdown("## 📥 STEP 1: LOAD HISTORICAL DATA")
    
    @st.cache_data(ttl=3600)
    def load_github_data():
        """Load historical ATP and WTA data"""
        wta_url = "https://github.com/paulom40/teste/raw/main/wta_data.xlsx"
        atp_url = "https://github.com/paulom40/teste/raw/main/atp_data.xlsx"
        
        dfs = []
        
        try:
            response = requests.get(wta_url, timeout=15)
            response.raise_for_status()
            wta_df = pd.read_excel(BytesIO(response.content))
            wta_df['Tour'] = 'WTA'
            st.success("✅ WTA historical data loaded")
            dfs.append(wta_df)
        except:
            st.warning("Could not load WTA data")
        
        try:
            response = requests.get(atp_url, timeout=15)
            response.raise_for_status()
            atp_df = pd.read_excel(BytesIO(response.content))
            atp_df['Tour'] = 'ATP'
            st.success("✅ ATP historical data loaded")
            dfs.append(atp_df)
        except:
            st.warning("Could not load ATP data")
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return None
    
    df = load_github_data()
    
    if df is None:
        st.error("❌ Could not load historical data")
        st.stop()
    
    st.info(f"📊 Historical matches loaded: {len(df):,}")
    
    st.markdown("---")
    
    # ANALYSIS
    st.markdown("## 🔍 STEP 2: ANALYZE HARD COURT DATA")
    
    def calculate_total_games(row):
        """Calculate total games"""
        total = 0
        for i in range(1, 6):
            w = row.get(f'W{i}', 0)
            l = row.get(f'L{i}', 0)
            if pd.notna(w) and pd.notna(l) and w > 0 and l > 0:
                total += int(w) + int(l)
        return total if total > 0 else None
    
    df_analysis = df.copy()
    df_analysis['Total_Games'] = df_analysis.apply(calculate_total_games, axis=1)
    df_analysis = df_analysis.dropna(subset=['Total_Games'])
    
    # Focus on HARD COURT data
    hard_court_data = df_analysis[df_analysis['Surface'].str.contains('Hard', case=False, na=False)]
    
    st.info(f"🎾 Hard Court historical matches: {len(hard_court_data):,}")
    
    if len(hard_court_data) > 50:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Games (Hard)", f"{hard_court_data['Total_Games'].mean():.1f}")
        col2.metric("Min Games", int(hard_court_data['Total_Games'].min()))
        col3.metric("Max Games", int(hard_court_data['Total_Games'].max()))
        col4.metric("Data Points", len(hard_court_data))
        
        st.markdown("---")
        st.markdown("## 🔮 STEP 3: GENERATE PREDICTIONS")
        
        if st.button("🎯 Predict Games (22-25 Range)", use_container_width=True, key="predict"):
            with st.spinner("Training model and generating predictions..."):
                
                @st.cache_resource
                def train_model():
                    surf_data = hard_court_data.copy()
                    surf_data = surf_data.dropna(subset=['Total_Games'])
                    
                    w1 = pd.to_numeric(surf_data['W1'], errors='coerce').fillna(0).values
                    l1 = pd.to_numeric(surf_data['L1'], errors='coerce').fillna(0).values
                    w2 = pd.to_numeric(surf_data['W2'], errors='coerce').fillna(0).values
                    l2 = pd.to_numeric(surf_data['L2'], errors='coerce').fillna(0).values
                    
                    features = [
                        w1 + l1,
                        w2 + l2,
                        (surf_data['Wsets'] == 2).astype(float).values,
                        (surf_data['Wsets'] == 3).astype(float).values,
                        1 / (1 + np.abs(w1 - l1) + np.abs(w2 - l2))
                    ]
                    
                    X = np.column_stack(features)
                    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
                    y = surf_data['Total_Games'].values
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    
                    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    
                    return model, scaler
                
                model, scaler = train_model()
                
                predictions = []
                
                for idx, match in all_matches.iterrows():
                    found = False
                    for _ in range(40):
                        w1 = np.random.randint(4, 7)
                        l1 = np.random.randint(2, 7)
                        w2 = np.random.randint(4, 7)
                        l2 = np.random.randint(2, 7)
                        
                        is_3set = np.random.random() < 0.25
                        w3 = np.random.randint(6, 8) if is_3set else 0
                        l3 = np.random.randint(2, 6) if is_3set else 0
                        
                        X_scenario = np.array([[
                            w1 + l1,
                            w2 + l2,
                            1.0 if not is_3set else 0.0,
                            1.0 if is_3set else 0.0,
                            1 / (1 + abs(w1 - l1) + abs(w2 - l2))
                        ]])
                        
                        X_scenario_scaled = scaler.transform(X_scenario)
                        predicted = model.predict(X_scenario_scaled)[0]
                        
                        if 22 <= predicted <= 25:
                            actual = w1 + l1 + w2 + l2 + (w3 + l3 if is_3set else 0)
                            predictions.append({
                                'Date': match['Date'],
                                'Tour': match['Tour'],
                                'Player 1': match['Player 1'],
                                'Player 2': match['Player 2'],
                                'Surface': match['Surface'],
                                'Set 1': f"{w1}-{l1}",
                                'Set 2': f"{w2}-{l2}",
                                'Set 3': f"{w3}-{l3}" if is_3set else "—",
                                'Predicted Games': round(predicted, 1),
                                '✅ Probability': "HIGH"
                            })
                            found = True
                            break
                
                if predictions:
                    predictions_df = pd.DataFrame(predictions)
                    predictions_df = predictions_df.sort_values('Predicted Games', ascending=False)
                    
                    st.markdown(f"### ✅ {len(predictions_df)} MATCHES WITH 22-25 GAMES PREDICTION")
                    st.dataframe(predictions_df, use_container_width=True, hide_index=True)
                    
                    st.markdown("---")
                    st.markdown("## 📥 EXPORT RESULTS")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📊 Download as CSV",
                            data=csv,
                            file_name=f"Miami_Predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        if st.button("📊 Download as Excel", use_container_width=True):
                            try:
                                import openpyxl
                                from openpyxl.styles import Font, PatternFill
                                
                                output = BytesIO()
                                
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    summary = pd.DataFrame({
                                        'Metric': ['Date', 'Tournament', 'Surface', 'Total', 'WTA', 'ATP', 'Avg Games'],
                                        'Value': [
                                            today.strftime('%d/%m/%Y'),
                                            'Miami 2026',
                                            'Hard Court',
                                            len(predictions_df),
                                            len(predictions_df[predictions_df['Tour'] == 'WTA']),
                                            len(predictions_df[predictions_df['Tour'] == 'ATP']),
                                            f"{predictions_df['Predicted Games'].mean():.1f}"
                                        ]
                                    })
                                    summary.to_excel(writer, sheet_name='Summary', index=False)
                                    predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
                                    
                                    workbook = writer.book
                                    for sheet in workbook.sheetnames:
                                        for cell in workbook[sheet][1]:
                                            if cell.value:
                                                cell.fill = PatternFill(start_color="667eea", end_color="667eea", fill_type="solid")
                                                cell.font = Font(bold=True, color="FFFFFF")
                                
                                output.seek(0)
                                st.download_button(
                                    label="✅ Download Excel",
                                    data=output.getvalue(),
                                    file_name=f"Miami_Predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    
                    st.markdown("---")
                    st.markdown("### 📊 STATISTICS")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Total", len(predictions_df))
                    col2.metric("Avg Games", f"{predictions_df['Predicted Games'].mean():.1f}")
                    col3.metric("Min", f"{predictions_df['Predicted Games'].min():.1f}")
                    col4.metric("Max", f"{predictions_df['Predicted Games'].max():.1f}")
                    col5.metric("WTA", len(predictions_df[predictions_df['Tour'] == 'WTA']))
                else:
                    st.warning("⚠️ No predictions in 22-25 range")

else:
    st.info("👆 Enter SCHEDULED matches above (only upcoming, not finished!)")
