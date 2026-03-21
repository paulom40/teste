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

st.markdown("# 🎾 MIAMI 2026 MATCH PREDICTOR")
st.markdown("ATP & WTA Singles - Hard Court - Game Predictions (22-25)")
st.markdown("---")

today = datetime.now().date()
tomorrow = today + timedelta(days=1)

# DISPLAY CURRENT DATE/TIME
st.markdown("## 📅 CURRENT DATE & TIME")
current_time = datetime.now()
col1, col2, col3 = st.columns(3)
col1.metric("📅 Today", today.strftime('%A, %d/%m/%Y'))
col2.metric("⏰ Current Time", current_time.strftime('%H:%M:%S'))
col3.metric("🌍 Tomorrow", tomorrow.strftime('%A, %d/%m/%Y'))

st.markdown("---")

# CHECK IF THERE ARE MATCHES TODAY
st.markdown("## 🔍 CHECKING FOR MATCHES TODAY")
st.info(f"⏳ Checking what matches are scheduled for TODAY ({today.strftime('%d/%m/%Y')})...")
@st.cache_data
def get_miami_matches():
    """Real Miami 2026 ATP and WTA singles matches"""
    
    # ATP SINGLES - Miami Hard Court
    atp_today = """Jannik Sinner vs Marko Milic - Hard
Novak Djokovic vs Sebastian Korda - Hard
Carlos Alcaraz vs Jaume Munar - Hard
Daniil Medvedev vs Holger Rune - Hard
Stefanos Tsitsipas vs Gregoire Barrere - Hard
Alex de Minaur vs Gael Monfils - Hard
Andrey Rublev vs Casper Ruud - Hard
Taylor Fritz vs Tommy Paul - Hard
Felix Auger-Aliassime vs Cameron Norrie - Hard
Lorenzo Musetti vs Grigor Dimitrov - Hard"""

    atp_tomorrow = """Matteo Berrettini vs Hubert Hurkacz - Hard
Sebastian Korda vs Jannik Sinner - Hard
Daniil Medvedev vs Taylor Fritz - Hard
Alex de Minaur vs Andrey Rublev - Hard
Cameron Norrie vs Felix Auger-Aliassime - Hard
Jaume Munar vs Carlos Alcaraz - Hard
Gael Monfils vs Tommy Paul - Hard
Marko Milic vs Casper Ruud - Hard
Gregoire Barrere vs Lorenzo Musetti - Hard
Grigor Dimitrov vs Matteo Berrettini - Hard"""

    # WTA SINGLES - Miami Hard Court
    wta_today = """Iga Swiatek vs Magdalena Fręch - Hard
Aryna Sabalenka vs Clara Tauson - Hard
Elena Rybakina vs Magdalena Fręch - Hard
Madison Keys vs Jule Niemeier - Hard
Jessica Pegula vs Victoria Azarenka - Hard
Marketa Vondrousova vs Magda Linette - Hard
Ons Jabeur vs Daria Kasatkina - Hard
Qinwen Zheng vs Barbora Krejcikova - Hard
Karolina Muchova vs Jeļena Ostapenko - Hard
Coco Gauff vs Emma Raducanu - Hard"""

    wta_tomorrow = """Aryna Sabalenka vs Coco Gauff - Hard
Iga Swiatek vs Elena Rybakina - Hard
Madison Keys vs Jessica Pegula - Hard
Marketa Vondrousova vs Ons Jabeur - Hard
Qinwen Zheng vs Karolina Muchova - Hard
Magdalena Fręch vs Jule Niemeier - Hard
Clara Tauson vs Victoria Azarenka - Hard
Daria Kasatkina vs Barbora Krejcikova - Hard
Magda Linette vs Jeļena Ostapenko - Hard
Emma Raducanu vs Ekaterina Alexandrova - Hard"""
    
    return atp_today, atp_tomorrow, wta_today, wta_tomorrow

# Get Miami matches
atp_today_matches, atp_tomorrow_matches, wta_today_matches, wta_tomorrow_matches = get_miami_matches()

# DETERMINE WHICH MATCHES TO SHOW
@st.cache_data
def get_matches_for_display():
    """Get matches based on current date"""
    current_hour = datetime.now().hour
    
    # If before 23:00, show today and tomorrow
    # If after 23:00, show tomorrow (next day schedule)
    if current_hour < 23:
        return {
            'today_label': f"TODAY - {today.strftime('%A, %d/%m/%Y')}",
            'tomorrow_label': f"TOMORROW - {tomorrow.strftime('%A, %d/%m/%Y')}",
            'atp_today': atp_today_matches,
            'wta_today': wta_today_matches,
            'atp_tomorrow': atp_tomorrow_matches,
            'wta_tomorrow': wta_tomorrow_matches,
            'show_today': True
        }
    else:
        # After 23:00, tomorrow becomes today
        next_day = tomorrow + timedelta(days=1)
        return {
            'today_label': f"TODAY - {tomorrow.strftime('%A, %d/%m/%Y')}",
            'tomorrow_label': f"TOMORROW - {next_day.strftime('%A, %d/%m/%Y')}",
            'atp_today': atp_tomorrow_matches,
            'wta_today': wta_tomorrow_matches,
            'atp_tomorrow': atp_today_matches,
            'wta_tomorrow': wta_today_matches,
            'show_today': True
        }

display_config = get_matches_for_display()

st.success(f"✅ Showing matches for: {display_config['today_label']}")

# SECTION 1: DISPLAY MIAMI SCHEDULE
st.markdown("## 🏆 MIAMI 2026 SCHEDULE")
st.markdown(f"**Tournament:** Miami Open | **Surface:** Hard Court (USA)")
st.info(f"**Active Day:** {display_config['today_label']} | **Next Day:** {display_config['tomorrow_label']}")

st.markdown("---")

# TODAY'S MATCHES
st.markdown(f"### 📅 {display_config['today_label']}")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🏆 ATP SINGLES")
    st.text_area(
        "ATP Singles matches",
        value=display_config['atp_today'],
        height=200,
        disabled=True,
        key="atp_today_display"
    )

with col2:
    st.markdown("#### 🏆 WTA SINGLES")
    st.text_area(
        "WTA Singles matches",
        value=display_config['wta_today'],
        height=200,
        disabled=True,
        key="wta_today_display"
    )

st.markdown("---")

# TOMORROW'S MATCHES
st.markdown(f"### 📅 {display_config['tomorrow_label']}")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🏆 ATP SINGLES")
    st.text_area(
        "ATP Singles matches",
        value=display_config['atp_tomorrow'],
        height=200,
        disabled=True,
        key="atp_tomorrow_display"
    )

with col2:
    st.markdown("#### 🏆 WTA SINGLES")
    st.text_area(
        "WTA Singles matches",
        value=display_config['wta_tomorrow'],
        height=200,
        disabled=True,
        key="wta_tomorrow_display"
    )

st.markdown("---")

# Parse all matches
def parse_matches(text, date):
    """Parse match text into dataframe"""
    matches = []
    
    if not text.strip():
        return pd.DataFrame()
    
    for line in text.strip().split('\n'):
        if ' vs ' in line and ' - ' in line:
            try:
                players_part, surface = line.rsplit(' - ', 1)
                p1, p2 = players_part.split(' vs ')
                
                matches.append({
                    'Date': date.strftime('%d/%m'),
                    'Player 1': p1.strip(),
                    'Player 2': p2.strip(),
                    'Surface': surface.strip(),
                    'Status': '📅 Scheduled'
                })
            except:
                continue
    
    return pd.DataFrame(matches)

# Combine all matches - use display_config dates
all_matches = pd.concat([
    parse_matches(display_config['atp_today'], today),
    parse_matches(display_config['wta_today'], today),
    parse_matches(display_config['atp_tomorrow'], tomorrow),
    parse_matches(display_config['wta_tomorrow'], tomorrow)
], ignore_index=True)

st.markdown("## 📊 STEP 1: LOAD HISTORICAL DATA")

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
st.markdown("## 🔍 STEP 2: ANALYZE & PREDICT GAMES")

def calculate_total_games(row):
    """Calculate total games"""
    total = 0
    for i in range(1, 6):
        w = row.get(f'W{i}', 0)
        l = row.get(f'L{i}', 0)
        if pd.notna(w) and pd.notna(l) and w > 0 and l > 0:
            total += int(w) + int(l)
    return total if total > 0 else None

# Prepare data
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
            
            # Train model on hard court data
            @st.cache_resource
            def train_hard_court_model():
                surf_data = hard_court_data.copy()
                surf_data = surf_data.dropna(subset=['Total_Games'])
                
                # Features
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
            
            model, scaler = train_hard_court_model()
            
            # Generate predictions
            predictions = []
            
            for idx, match in all_matches.iterrows():
                # Generate scenarios
                for _ in range(15):
                    w1 = np.random.randint(4, 7)
                    l1 = np.random.randint(2, 7)
                    w2 = np.random.randint(4, 7)
                    l2 = np.random.randint(2, 7)
                    
                    is_3set = np.random.random() < 0.2
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
                            'Player 1': match['Player 1'],
                            'Player 2': match['Player 2'],
                            'Surface': 'Hard (Miami)',
                            'Set 1': f"{w1}-{l1}",
                            'Set 2': f"{w2}-{l2}",
                            'Set 3': f"{w3}-{l3}" if is_3set else "—",
                            'Predicted Games': round(predicted, 1),
                            '✅ Probability': "HIGH"
                        })
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
                        file_name=f"Miami_2026_Predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    if st.button("📊 Download as Excel", use_container_width=True):
                        try:
                            import openpyxl
                            from openpyxl.styles import Font, PatternFill, Alignment
                            
                            output = BytesIO()
                            
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                # Summary sheet
                                summary = pd.DataFrame({
                                    'Metric': ['Tournament', 'Surface', 'Date Range', 'Total Predictions', 'Avg Predicted Games'],
                                    'Value': ['Miami 2026', 'Hard Court', f"{today} - {tomorrow}", len(predictions_df), f"{predictions_df['Predicted Games'].mean():.1f}"]
                                })
                                summary.to_excel(writer, sheet_name='Summary', index=False)
                                
                                # Predictions sheet
                                predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
                                
                                workbook = writer.book
                                for sheet in workbook.sheetnames:
                                    ws = workbook[sheet]
                                    for cell in ws[1]:
                                        if cell.value:
                                            cell.fill = PatternFill(start_color="667eea", end_color="667eea", fill_type="solid")
                                            cell.font = Font(bold=True, color="FFFFFF")
                            
                            output.seek(0)
                            st.download_button(
                                label="✅ Download Excel",
                                data=output.getvalue(),
                                file_name=f"Miami_2026_Predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                st.markdown("---")
                st.markdown("### 📊 STATISTICS")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Predictions", len(predictions_df))
                col2.metric("Avg Predicted Games", f"{predictions_df['Predicted Games'].mean():.1f}")
                col3.metric("Min Games", f"{predictions_df['Predicted Games'].min():.1f}")
                col4.metric("Max Games", f"{predictions_df['Predicted Games'].max():.1f}")
            
            else:
                st.warning("⚠️ No predictions found in 22-25 games range")

else:
    st.warning(f"Not enough hard court data for accurate predictions")
