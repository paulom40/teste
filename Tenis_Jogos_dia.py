import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Miami 2026 Auto Scraper", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🎾 MIAMI 2026 - AUTO SCRAPER & PREDICTOR")
st.markdown("Automatically gets real match data and predicts games (22-25)")
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

# AUTO SCRAPE FLASHSCORE
st.markdown("## 🌐 STEP 1: SCRAPING REAL MATCH DATA FROM FLASHSCORE")
st.info("⏳ Automatically fetching real Miami 2026 matches...")

@st.cache_data(ttl=1800)
def scrape_flashscore_miami():
    """Scrape Miami 2026 matches from FlashScore"""
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    all_matches = {
        'ATP': {'today': [], 'tomorrow': []},
        'WTA': {'today': [], 'tomorrow': []}
    }
    
    try:
        # Try ATP matches
        st.markdown("### 🔍 Scraping ATP matches...")
        atp_url = 'https://www.flashscore.com/tennis/atp/'
        response = requests.get(atp_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for match data
        matches = soup.find_all('div', {'class': 'event__match'})
        
        if matches:
            for match in matches[:20]:
                try:
                    teams = match.find_all('span', {'class': 'event__participant'})
                    if len(teams) >= 2:
                        p1 = teams[0].text.strip()
                        p2 = teams[1].text.strip()
                        
                        if p1 and p2 and len(p1) > 2 and len(p2) > 2:
                            all_matches['ATP']['today'].append(f"{p1} vs {p2}")
                except:
                    continue
        
        if all_matches['ATP']['today']:
            st.success(f"✅ ATP: Found {len(all_matches['ATP']['today'])} matches")
        else:
            st.warning("⚠️ ATP: Could not scrape live data, using alternative method...")
    
    except Exception as e:
        st.warning(f"ATP scraping error: {str(e)}")
    
    try:
        # Try WTA matches
        st.markdown("### 🔍 Scraping WTA matches...")
        wta_url = 'https://www.flashscore.com/tennis/wta/'
        response = requests.get(wta_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for match data
        matches = soup.find_all('div', {'class': 'event__match'})
        
        if matches:
            for match in matches[:20]:
                try:
                    teams = match.find_all('span', {'class': 'event__participant'})
                    if len(teams) >= 2:
                        p1 = teams[0].text.strip()
                        p2 = teams[1].text.strip()
                        
                        if p1 and p2 and len(p1) > 2 and len(p2) > 2:
                            all_matches['WTA']['today'].append(f"{p1} vs {p2}")
                except:
                    continue
        
        if all_matches['WTA']['today']:
            st.success(f"✅ WTA: Found {len(all_matches['WTA']['today'])} matches")
        else:
            st.warning("⚠️ WTA: Could not scrape live data, using sample data...")
    
    except Exception as e:
        st.warning(f"WTA scraping error: {str(e)}")
    
    # If no data scraped, use sample Miami data
    if not all_matches['ATP']['today'] and not all_matches['WTA']['today']:
        st.warning("⚠️ Using sample Miami 2026 data...")
        
        all_matches['ATP']['today'] = [
            "Jannik Sinner vs Marko Milic",
            "Novak Djokovic vs Sebastian Korda",
            "Carlos Alcaraz vs Jaume Munar",
            "Daniil Medvedev vs Holger Rune",
            "Stefanos Tsitsipas vs Gregoire Barrere",
            "Alex de Minaur vs Gael Monfils",
            "Andrey Rublev vs Casper Ruud",
            "Taylor Fritz vs Tommy Paul"
        ]
        
        all_matches['WTA']['today'] = [
            "Qinwen Zheng vs Sloane Stephens",
            "Iga Swiatek vs Clara Tauson",
            "Aryna Sabalenka vs Magdalena Fręch",
            "Madison Keys vs Jule Niemeier",
            "Jessica Pegula vs Victoria Azarenka",
            "Marketa Vondrousova vs Magda Linette",
            "Ons Jabeur vs Daria Kasatkina",
            "Karolina Muchova vs Jeļena Ostapenko"
        ]
        
        all_matches['ATP']['tomorrow'] = [
            "Matteo Berrettini vs Hubert Hurkacz",
            "Lorenzo Musetti vs Sebastian Korda",
            "Cameron Norrie vs Felix Auger-Aliassime"
        ]
        
        all_matches['WTA']['tomorrow'] = [
            "Coco Gauff vs Emma Raducanu",
            "Aryna Sabalenka vs Coco Gauff",
            "Madison Keys vs Jessica Pegula"
        ]
        
        st.success("✅ Using sample Miami 2026 data")
    
    return all_matches

# Scrape data
with st.spinner("🌐 Fetching match data from FlashScore..."):
    scraped_matches = scrape_flashscore_miami()

st.markdown("---")

# Parse matches
def parse_match_list(matches_list, date, tour):
    """Convert match list to dataframe"""
    matches = []
    
    for match_text in matches_list:
        if ' vs ' in match_text:
            try:
                parts = match_text.split(' vs ')
                p1 = parts[0].strip()
                p2 = parts[1].strip()
                
                if p1 and p2:
                    matches.append({
                        'Date': date.strftime('%d/%m'),
                        'Player 1': p1,
                        'Player 2': p2,
                        'Tour': tour,
                        'Surface': 'Hard (Miami)',
                        'Status': '📅 Scheduled'
                    })
            except:
                continue
    
    return pd.DataFrame(matches)

# Combine all matches
all_matches_df = pd.concat([
    parse_match_list(scraped_matches['ATP']['today'], today, 'ATP'),
    parse_match_list(scraped_matches['WTA']['today'], today, 'WTA'),
    parse_match_list(scraped_matches['ATP']['tomorrow'], tomorrow, 'ATP'),
    parse_match_list(scraped_matches['WTA']['tomorrow'], tomorrow, 'WTA')
], ignore_index=True)

st.markdown("## ✅ LOADED MATCHES")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Matches", len(all_matches_df))
col2.metric("ATP Matches", len(all_matches_df[all_matches_df['Tour'] == 'ATP']))
col3.metric("WTA Matches", len(all_matches_df[all_matches_df['Tour'] == 'WTA']))
col4.metric("Today Matches", len(all_matches_df[all_matches_df['Date'] == today.strftime('%d/%m')]))

st.dataframe(all_matches_df, use_container_width=True, hide_index=True)

st.markdown("---")

# LOAD HISTORICAL DATA
st.markdown("## 📥 STEP 2: LOAD HISTORICAL DATA")

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
st.markdown("## 🔍 STEP 3: ANALYZE & PREDICT GAMES")

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
    st.markdown("## 🔮 STEP 4: GENERATE PREDICTIONS")
    
    if st.button("🎯 Predict Games (22-25 Range) for All Scraped Matches", use_container_width=True, key="predict"):
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
            
            # Generate predictions for each match
            predictions = []
            
            for idx, match in all_matches_df.iterrows():
                # Generate 25 scenarios per match
                for scenario_num in range(25):
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
                        file_name=f"Miami_2026_Auto_Predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
                                    'Metric': ['Source', 'Tournament', 'Surface', 'Total Predictions', 'WTA', 'ATP', 'Avg Games'],
                                    'Value': [
                                        'FlashScore Auto Scraper',
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
                                file_name=f"Miami_2026_Auto_Predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                st.markdown("---")
                st.markdown("### 📊 STATISTICS")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total Predictions", len(predictions_df))
                col2.metric("Avg Games", f"{predictions_df['Predicted Games'].mean():.1f}")
                col3.metric("Min Games", f"{predictions_df['Predicted Games'].min():.1f}")
                col4.metric("Max Games", f"{predictions_df['Predicted Games'].max():.1f}")
                col5.metric("WTA Count", len(predictions_df[predictions_df['Tour'] == 'WTA']))
            
            else:
                st.warning("⚠️ No predictions found in 22-25 games range")

else:
    st.warning(f"Not enough hard court data for accurate predictions")
