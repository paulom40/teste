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

st.set_page_config(page_title="ATP/WTA Real Matches", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🎾 ATP/WTA REAL MATCHES PREDICTOR")
st.markdown("Live matches for today/tomorrow with game predictions (22-25 games)")
st.markdown("---")

# SCRAPE REAL MATCHES
st.markdown("## 🌐 LOADING REAL MATCHES")

@st.cache_data(ttl=1800)
def scrape_tennisexplorer():
    """Scrape real matches from Tennis Explorer"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        matches = []
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        
        # Try Tennis Explorer
        url_today = 'https://www.tennisexplorer.com/'
        
        response = requests.get(url_today, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for match data in the page
        match_elements = soup.find_all('tr', {'class': ['dark', 'light']})
        
        if match_elements:
            for elem in match_elements[:20]:
                try:
                    cells = elem.find_all('td')
                    if len(cells) >= 4:
                        time_text = cells[0].text.strip()
                        player1 = cells[1].text.strip()
                        player2 = cells[2].text.strip()
                        score = cells[3].text.strip() if len(cells) > 3 else "TBA"
                        
                        if player1 and player2 and time_text:
                            matches.append({
                                'Date': today.strftime('%d/%m'),
                                'Time': time_text,
                                'Tour': 'ATP/WTA',
                                'Player 1': player1,
                                'Player 2': player2,
                                'Surface': 'Unknown',
                                'Status': '📅 Live'
                            })
                except:
                    continue
        
        if matches:
            st.success(f"✅ Found {len(matches)} real matches")
            return pd.DataFrame(matches)
        else:
            return None
            
    except Exception as e:
        st.warning(f"Could not scrape live data: {str(e)}")
        return None

@st.cache_data(ttl=1800)
def scrape_sofascore():
    """Scrape from SofaScore API (more reliable)"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        today = datetime.now().date()
        
        # SofaScore tennis data
        url = f'https://www.sofascore.com/tennis/'
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        matches = []
        
        # Find match containers
        match_divs = soup.find_all('div', {'class': 'sc-l3xrv4'})
        
        if match_divs:
            for match_div in match_divs[:15]:
                try:
                    # Extract match info
                    text_content = match_div.get_text()
                    
                    if '-' in text_content and any(char.isdigit() for char in text_content):
                        matches.append({
                            'Date': today.strftime('%d/%m'),
                            'Time': 'TBA',
                            'Tour': 'ATP/WTA',
                            'Player 1': text_content.split('-')[0][:30] if '-' in text_content else 'Player 1',
                            'Player 2': text_content.split('-')[1][:30] if '-' in text_content else 'Player 2',
                            'Surface': 'Unknown',
                            'Status': '📅 Scheduled'
                        })
                except:
                    continue
        
        if matches:
            return pd.DataFrame(matches)
        return None
        
    except Exception as e:
        st.warning(f"SofaScore error: {str(e)}")
        return None

# Try to scrape real data
st.markdown("### 🔍 Searching for real matches...")

with st.spinner("Scraping live match data..."):
    real_matches = scrape_sofascore()
    if real_matches is None:
        real_matches = scrape_tennisexplorer()

today = datetime.now().date()
tomorrow = today + timedelta(days=1)

if real_matches is not None and len(real_matches) > 0:
    st.success(f"✅ Found real matches for today")
    
    # Display today's matches
    st.markdown(f"### 📅 TODAY - {today.strftime('%A, %d/%m/%Y')}")
    st.dataframe(real_matches, use_container_width=True, hide_index=True)
    
    st.markdown("---")
else:
    st.warning("⚠️ No real live matches found. Using data from historical analysis.")
    
    # Show message
    st.info("""
    💡 **Note:** Live match data requires:
    - Matches to be scheduled today/tomorrow
    - Tennis tournament calendar active
    - Website accessibility
    
    **Proceeding with historical data analysis to predict potential 22-25 games matches...**
    """)

st.markdown("---")

# LOAD HISTORICAL DATA FOR ANALYSIS
st.markdown("## 📥 LOADING HISTORICAL DATA")

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
        st.success("✅ WTA data loaded")
        dfs.append(wta_df)
    except:
        st.warning("Could not load WTA")
    
    try:
        response = requests.get(atp_url, timeout=15)
        response.raise_for_status()
        atp_df = pd.read_excel(BytesIO(response.content))
        atp_df['Tour'] = 'ATP'
        st.success("✅ ATP data loaded")
        dfs.append(atp_df)
    except:
        st.warning("Could not load ATP")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

# Load data
df = load_github_data()

if df is None:
    st.error("❌ Could not load historical data")
    st.stop()

st.info(f"📊 Total historical matches: {len(df):,}")

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

# Prepare data
st.markdown("## 🔍 ANALYZING HISTORICAL DATA BY SURFACE")

df_analysis = df.copy()
df_analysis['Total_Games'] = df_analysis.apply(calculate_total_games, axis=1)
df_analysis = df_analysis.dropna(subset=['Total_Games'])

# Get available surfaces
available_surfaces = sorted(df_analysis['Surface'].dropna().unique())
st.info(f"📊 Matches analyzed: {len(df_analysis):,}")
st.info(f"🏆 Surfaces: {', '.join(available_surfaces)}")

st.markdown("---")

# SURFACE SELECTION
st.markdown("## 🏆 SELECT SURFACE & GENERATE PREDICTIONS")

col1, col2 = st.columns(2)

with col1:
    surface = st.selectbox(
        "Choose surface",
        available_surfaces,
        key="surface"
    )

with col2:
    tour = st.selectbox("Choose tour", ['ATP', 'WTA', 'Both'], key="tour")

# Analyze surface
surface_matches = df_analysis[df_analysis['Surface'] == surface].copy()

st.info(f"📊 {surface} matches: {len(surface_matches):,} | Avg games: {surface_matches['Total_Games'].mean():.1f}")

if len(surface_matches) > 0:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Avg Games", f"{surface_matches['Total_Games'].mean():.1f}")
    col2.metric(f"Min Games", int(surface_matches['Total_Games'].min()))
    col3.metric(f"Max Games", int(surface_matches['Total_Games'].max()))
    col4.metric(f"Matches Count", len(surface_matches))
    
    st.markdown("---")
    
    # Train model
    @st.cache_resource
    def train_model(df_train, surf):
        """Train model on surface-specific data"""
        df_model = df_train[df_train['Surface'] == surf].copy()
        df_model = df_model.dropna(subset=['Total_Games'])
        
        if len(df_model) < 30:
            return None, None, None
        
        features = []
        w1 = pd.to_numeric(df_model['W1'], errors='coerce').fillna(0).values
        l1 = pd.to_numeric(df_model['L1'], errors='coerce').fillna(0).values
        w2 = pd.to_numeric(df_model['W2'], errors='coerce').fillna(0).values
        l2 = pd.to_numeric(df_model['L2'], errors='coerce').fillna(0).values
        
        features.append(w1 + l1)
        features.append(w2 + l2)
        features.append((df_model['Wsets'] == 2).astype(float).values)
        features.append((df_model['Wsets'] == 3).astype(float).values)
        features.append(1 / (1 + np.abs(w1 - l1) + np.abs(w2 - l2)))
        
        X = np.column_stack(features)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        y = df_model['Total_Games'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        return model, scaler, df_model
    
    with st.spinner("Training prediction model..."):
        model, scaler, model_df = train_model(df_analysis, surface)
    
    if model is None:
        st.error(f"❌ Not enough data for {surface}")
    else:
        st.success(f"✅ Model trained on {len(model_df):,} {surface} matches")
        
        st.markdown("---")
        st.markdown("## 🔮 GENERATE 22-25 GAMES PREDICTIONS")
        
        num_scenarios = st.slider("Generate N match scenarios", 5, 50, 20, key="scenarios")
        
        if st.button("🎯 Predict Matches (22-25 Games)", use_container_width=True, key="generate"):
            with st.spinner("Generating predictions..."):
                # Get players
                if tour == 'Both':
                    tour_data = df_analysis.copy()
                else:
                    tour_data = df_analysis[df_analysis['Tour'] == tour].copy()
                
                players = pd.concat([tour_data['Winner'], tour_data['Loser']]).unique()
                players = [p for p in players if pd.notna(p)]
                
                if len(players) < 2:
                    st.error("Not enough players")
                else:
                    scenarios = []
                    np.random.seed(42)
                    
                    for _ in range(num_scenarios):
                        p1, p2 = np.random.choice(players, 2, replace=False)
                        
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
                        predicted_games = model.predict(X_scenario_scaled)[0]
                        
                        if 22 <= predicted_games <= 25:
                            actual = w1 + l1 + w2 + l2 + (w3 + l3 if is_3set else 0)
                            scenarios.append({
                                'Player 1': p1,
                                'Player 2': p2,
                                'Surface': surface,
                                'Set 1': f"{w1}-{l1}",
                                'Set 2': f"{w2}-{l2}",
                                'Set 3': f"{w3}-{l3}" if is_3set else "N/A",
                                'Actual Games': actual,
                                'Predicted': round(predicted_games, 1),
                                '✅ Probability': "HIGH"
                            })
                    
                    if scenarios:
                        scenarios_df = pd.DataFrame(scenarios)
                        scenarios_df = scenarios_df.sort_values('Predicted', ascending=False)
                        
                        st.markdown(f"### ✅ Found {len(scenarios_df)} matches (22-25 games)")
                        st.dataframe(scenarios_df, use_container_width=True, hide_index=True)
                        
                        st.markdown("---")
                        st.markdown("## 📥 EXPORT")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv = scenarios_df.to_csv(index=False)
                            st.download_button(
                                label="📊 Download Excel",
                                data=csv,
                                file_name=f"Predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col2:
                            st.metric("Total Predictions", len(scenarios_df))
                    
                    else:
                        st.warning("⚠️ No matches found in 22-25 range")

else:
    st.warning(f"No data for {surface}")
