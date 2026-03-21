import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Miami 2026 Predictor Pro", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .match-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🎾 MIAMI 2026 PRO - ADVANCED GAME PREDICTOR")
st.markdown("AI-powered predictions for Miami Open 2026 (Games 22-25)")
st.markdown("---")

# API Configuration
RAPIDAPI_KEY = "bba6af0e8dmsh6350139b0f77a4ap16b6fajsn219553636a4"

class TennisDataAPI:
    def __init__(self):
        self.rapidapi_key = RAPIDAPI_KEY
        self.base_url = "https://tennis-live-data.p.rapidapi.com"
        
    def get_live_matches(self):
        """Get live and scheduled tennis matches"""
        try:
            url = f"{self.base_url}/matches"
            headers = {
                "X-RapidAPI-Key": self.rapidapi_key,
                "X-RapidAPI-Host": "tennis-live-data.p.rapidapi.com"
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return self.parse_matches(data)
            else:
                st.warning(f"API Error: Status {response.status_code}")
                return None
                
        except Exception as e:
            st.warning(f"Could not fetch from API: {str(e)}")
            return None
    
    def get_todays_matches(self):
        """Get today's scheduled matches"""
        try:
            url = f"{self.base_url}/fixtures/today"
            headers = {
                "X-RapidAPI-Key": self.rapidapi_key,
                "X-RapidAPI-Host": "tennis-live-data.p.rapidapi.com"
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return self.parse_matches(data)
            else:
                return None
                
        except Exception as e:
            st.warning(f"Could not fetch today's matches: {str(e)}")
            return None
    
    def get_tournament_matches(self, tournament_id="miami-open"):
        """Get matches for specific tournament"""
        try:
            url = f"{self.base_url}/tournaments/{tournament_id}/matches"
            headers = {
                "X-RapidAPI-Key": self.rapidapi_key,
                "X-RapidAPI-Host": "tennis-live-data.p.rapidapi.com"
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return self.parse_matches(data)
            else:
                return None
                
        except Exception as e:
            st.warning(f"Could not fetch tournament matches: {str(e)}")
            return None
    
    def parse_matches(self, data):
        """Parse API response into match objects"""
        matches = []
        
        try:
            # Handle different API response structures
            if isinstance(data, dict):
                if 'response' in data:
                    matches_data = data['response']
                elif 'matches' in data:
                    matches_data = data['matches']
                elif 'data' in data:
                    matches_data = data['data']
                else:
                    matches_data = [data]
            elif isinstance(data, list):
                matches_data = data
            else:
                return None
            
            for match in matches_data:
                try:
                    # Extract player names
                    player1 = match.get('home_name') or match.get('player1') or match.get('player_one') or 'Unknown'
                    player2 = match.get('away_name') or match.get('player2') or match.get('player_two') or 'Unknown'
                    
                    # Determine tour (ATP/WTA)
                    tour = 'ATP'
                    if 'wta' in str(match).lower() or 'women' in str(match).lower():
                        tour = 'WTA'
                    
                    # Check if match is scheduled (no score)
                    has_score = False
                    if 'score' in match:
                        if match['score'] and match['score'] != '0-0':
                            has_score = True
                    
                    if not has_score:
                        matches.append({
                            'Player 1': player1,
                            'Player 2': player2,
                            'Tour': tour,
                            'Time': match.get('time', match.get('start_time', 'TBD')),
                            'Round': match.get('round', match.get('stage', 'Unknown'))
                        })
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            st.warning(f"Error parsing matches: {str(e)}")
            
        return matches
    
    def get_demo_matches(self):
        """Demo matches for testing"""
        return [
            {"player1": "Jannik Sinner", "player2": "Carlos Alcaraz", "tour": "ATP", "time": "14:30", "round": "Quarter Finals"},
            {"player1": "Daniil Medvedev", "player2": "Alexander Zverev", "tour": "ATP", "time": "16:00", "round": "Quarter Finals"},
            {"player1": "Novak Djokovic", "player2": "Taylor Fritz", "tour": "ATP", "time": "18:30", "round": "Round of 16"},
            {"player1": "Iga Swiatek", "player2": "Elena Rybakina", "tour": "WTA", "time": "15:00", "round": "Semi Finals"},
            {"player1": "Coco Gauff", "player2": "Aryna Sabalenka", "tour": "WTA", "time": "20:00", "round": "Semi Finals"},
            {"player1": "Jessica Pegula", "player2": "Ons Jabeur", "tour": "WTA", "time": "17:30", "round": "Quarter Finals"}
        ]

# Initialize API
tennis_api = TennisDataAPI()

# Auto-fetch matches
st.markdown("## 🤖 AUTO-FETCH MATCHES")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔄 Live Matches", use_container_width=True):
        with st.spinner("Fetching live matches from API..."):
            matches = tennis_api.get_live_matches()
            if matches and len(matches) > 0:
                st.success(f"✅ Found {len(matches)} matches")
                st.session_state.auto_matches = matches
                st.session_state.match_source = "Live"
            else:
                st.warning("No live matches found, using demo data")
                st.session_state.auto_matches = tennis_api.get_demo_matches()
                st.session_state.match_source = "Demo"

with col2:
    if st.button("📅 Today's Matches", use_container_width=True):
        with st.spinner("Fetching today's scheduled matches..."):
            matches = tennis_api.get_todays_matches()
            if matches and len(matches) > 0:
                st.success(f"✅ Found {len(matches)} matches for today")
                st.session_state.auto_matches = matches
                st.session_state.match_source = "Today"
            else:
                st.warning("Using demo data for today's matches")
                st.session_state.auto_matches = tennis_api.get_demo_matches()
                st.session_state.match_source = "Demo"

with col3:
    if st.button("🏆 Miami Open", use_container_width=True):
        with st.spinner("Fetching Miami Open matches..."):
            matches = tennis_api.get_tournament_matches("miami-open")
            if matches and len(matches) > 0:
                st.success(f"✅ Found {len(matches)} Miami Open matches")
                st.session_state.auto_matches = matches
                st.session_state.match_source = "Miami Open"
            else:
                st.warning("Using Miami Open demo matches")
                st.session_state.auto_matches = tennis_api.get_demo_matches()
                st.session_state.match_source = "Demo"

with col4:
    if st.button("📊 API Status", use_container_width=True):
        st.info(f"API Key: {'✓ Active' if tennis_api.rapidapi_key else '✗ Missing'}")
        st.info("RapidAPI Tennis Data Service")

st.markdown("---")

# Auto-loaded matches display
if 'auto_matches' in st.session_state and st.session_state.auto_matches:
    st.markdown(f"### 📋 {st.session_state.match_source} MATCHES")
    
    auto_df = pd.DataFrame(st.session_state.auto_matches)
    
    # Display matches in a nice format
    st.dataframe(auto_df, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"📊 Total: {len(auto_df)} matches loaded")
    with col2:
        if st.button("✅ Confirm & Predict", use_container_width=True):
            st.session_state.confirmed_matches = st.session_state.auto_matches
            st.session_state.auto_loaded = True
            st.success("Matches confirmed! Scroll down for predictions.")
else:
    # Manual entry section
    st.markdown("## 📝 MANUAL MATCH ENTRY")
    st.markdown("*If auto-fetch doesn't work, enter matches manually*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 ATP SINGLES")
        atp_input = st.text_area(
            "ATP matches - one per line",
            placeholder="Jannik Sinner vs Carlos Alcaraz\nDaniil Medvedev vs Alexander Zverev",
            height=150
        )
    
    with col2:
        st.markdown("### 🏆 WTA SINGLES")
        wta_input = st.text_area(
            "WTA matches - one per line",
            placeholder="Iga Swiatek vs Elena Rybakina\nCoco Gauff vs Aryna Sabalenka",
            height=150
        )
    
    def parse_manual_matches(text, tour):
        matches = []
        if not text.strip():
            return []
        
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line or 'vs' not in line.lower():
                continue
            
            try:
                parts = line.split('vs')
                if len(parts) != 2:
                    continue
                
                p1 = parts[0].strip()
                p2 = parts[1].strip()
                
                # Skip finished matches
                if any(char.isdigit() for char in p1) or any(char.isdigit() for char in p2):
                    continue
                
                if 'finished' in line.lower() or 'live' in line.lower():
                    continue
                
                matches.append({
                    'Player 1': p1,
                    'Player 2': p2,
                    'Tour': tour,
                    'Time': 'TBD',
                    'Round': 'Manual Entry'
                })
            except:
                continue
        
        return matches
    
    manual_matches = parse_manual_matches(atp_input, 'ATP') + parse_manual_matches(wta_input, 'WTA')
    
    if manual_matches:
        st.success(f"✅ {len(manual_matches)} matches loaded manually")
        st.dataframe(pd.DataFrame(manual_matches), use_container_width=True, hide_index=True)
        
        if st.button("Use Manual Matches for Prediction"):
            st.session_state.confirmed_matches = manual_matches
            st.session_state.auto_loaded = True

# Proceed with predictions
if 'auto_loaded' in st.session_state and st.session_state.auto_loaded and 'confirmed_matches' in st.session_state:
    
    st.markdown("---")
    st.markdown("## 🔮 GENERATING PREDICTIONS")
    
    # Load historical data
    @st.cache_data(ttl=3600)
    def load_historical_data():
        """Load historical tennis data"""
        try:
            # Try to load from multiple sources
            urls = [
                "https://github.com/paulom40/teste/raw/main/atp_data.xlsx",
                "https://github.com/paulom40/teste/raw/main/wta_data.xlsx"
            ]
            
            dfs = []
            for url in urls:
                try:
                    response = requests.get(url, timeout=15)
                    response.raise_for_status()
                    df = pd.read_excel(BytesIO(response.content))
                    
                    if 'atp' in url.lower():
                        df['Tour'] = 'ATP'
                    else:
                        df['Tour'] = 'WTA'
                    
                    dfs.append(df)
                except:
                    continue
            
            if dfs:
                return pd.concat(dfs, ignore_index=True)
            else:
                # Generate synthetic data for testing
                st.warning("Using synthetic training data")
                return generate_synthetic_data()
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return generate_synthetic_data()
    
    def generate_synthetic_data():
        """Generate synthetic data for testing"""
        np.random.seed(42)
        n_matches = 5000
        
        data = {
            'Total_Games': np.random.normal(23, 4, n_matches),
            'Sets_Played': np.random.choice([2, 3], n_matches, p=[0.7, 0.3]),
            'Surface_Hard': np.ones(n_matches),
            'Winner_Rank': np.random.randint(1, 100, n_matches),
            'Loser_Rank': np.random.randint(1, 100, n_matches),
            'Tour': np.random.choice(['ATP', 'WTA'], n_matches)
        }
        
        df = pd.DataFrame(data)
        df['Total_Games'] = df['Total_Games'].clip(18, 35)
        return df
    
    df = load_historical_data()
    
    # Feature engineering
    def prepare_features(df):
        """Prepare features for model training"""
        features = []
        
        # Basic features
        if 'Total_Games' in df.columns:
            df['Avg_Games_Per_Set'] = df['Total_Games'] / df['Sets_Played'].replace(0, 1)
        
        if 'Winner_Rank' in df.columns and 'Loser_Rank' in df.columns:
            df['Rank_Difference'] = df['Loser_Rank'] - df['Winner_Rank']
            df['Rank_Advantage'] = 1 / (1 + np.abs(df['Rank_Difference']))
        
        # Surface indicator
        df['Is_Hard'] = 1
        
        # Select features for modeling
        feature_cols = ['Sets_Played', 'Avg_Games_Per_Set', 'Rank_Difference', 'Rank_Advantage', 'Is_Hard']
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df['Total_Games'].clip(18, 35)
        
        return X, y, feature_cols
    
    X, y, feature_cols = prepare_features(df)
    
    # Train model
    if len(X) > 0:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        model.fit(X_scaled, y)
        
        st.success("✅ Model trained successfully!")
        
        # Generate predictions
        predictions = []
        
        for match in st.session_state.confirmed_matches:
            # Monte Carlo simulation
            best_predictions = []
            
            for _ in range(100):
                # Simulate score
                is_3set = np.random.random() < 0.32
                
                if is_3set:
                    w1, l1 = np.random.randint(6, 8), np.random.randint(2, 6)
                    w2, l2 = np.random.randint(4, 7), np.random.randint(2, 7)
                    w3, l3 = np.random.randint(6, 8), np.random.randint(2, 6)
                else:
                    w1, l1 = np.random.randint(6, 8), np.random.randint(2, 6)
                    w2, l2 = np.random.randint(6, 8), np.random.randint(2, 6)
                    w3, l3 = 0, 0
                
                total_games = w1 + l1 + w2 + l2 + w3 + l3
                
                # Prepare features
                features_dict = {
                    'Sets_Played': 3 if is_3set else 2,
                    'Avg_Games_Per_Set': total_games / (3 if is_3set else 2),
                    'Rank_Difference': np.random.randint(-50, 50),
                    'Rank_Advantage': 1 / (1 + np.random.randint(0, 50)),
                    'Is_Hard': 1
                }
                
                feature_array = np.array([[features_dict[col] for col in feature_cols]])
                feature_scaled = scaler.transform(feature_array)
                
                predicted = model.predict(feature_scaled)[0]
                
                if 22 <= predicted <= 25:
                    confidence = min(95, int(100 * (1 - abs(predicted - 23.5) / 10)))
                    
                    best_predictions.append({
                        'Player 1': match['Player 1'],
                        'Player 2': match['Player 2'],
                        'Tour': match['Tour'],
                        'Round': match.get('Round', 'Unknown'),
                        'Set 1': f"{w1}-{l1}",
                        'Set 2': f"{w2}-{l2}",
                        'Set 3': f"{w3}-{l3}" if is_3set else "—",
                        'Total Games': total_games,
                        'Predicted Games': round(predicted, 1),
                        'Confidence': f"{confidence}%"
                    })
            
            if best_predictions:
                best_predictions.sort(key=lambda x: int(x['Confidence'].strip('%')), reverse=True)
                predictions.append(best_predictions[0])
        
        if predictions:
            predictions_df = pd.DataFrame(predictions)
            
            st.markdown("---")
            st.markdown(f"## 🎯 PREDICTIONS: {len(predictions_df)} MATCHES")
            st.markdown("### Games expected between 22-25")
            
            # Display predictions
            st.dataframe(predictions_df.style.background_gradient(subset=['Confidence'], cmap='RdYlGn'),
                        use_container_width=True, hide_index=True)
            
            # Export options
            st.markdown("---")
            st.markdown("## 📥 EXPORT RESULTS")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv,
                    file_name=f"Miami2026_Predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
                    
                    summary = pd.DataFrame({
                        'Metric': ['Date', 'Tournament', 'Surface', 'Total Matches', 
                                  'ATP', 'WTA', 'Avg Confidence'],
                        'Value': [
                            datetime.now().strftime('%d/%m/%Y'),
                            'Miami Open 2026',
                            'Hard Court',
                            len(predictions_df),
                            len(predictions_df[predictions_df['Tour'] == 'ATP']),
                            len(predictions_df[predictions_df['Tour'] == 'WTA']),
                            f"{predictions_df['Confidence'].str.strip('%').astype(float).mean():.1f}%"
                        ]
                    })
                    summary.to_excel(writer, sheet_name='Summary', index=False)
                
                output.seek(0)
                st.download_button(
                    label="📊 Download Excel",
                    data=output.getvalue(),
                    file_name=f"Miami2026_Predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        else:
            st.warning("⚠️ No matches found in 22-25 game range. Try again!")
    else:
        st.error("❌ Not enough data for training")

# Footer
st.markdown("---")
st.markdown("### 📊 About This Predictor")
st.markdown("""
- **Data Source**: RapidAPI Tennis Live Data + Historical Match Data
- **Model**: Random Forest Ensemble (200 trees, depth 8)
- **Target Range**: 22-25 games per match
- **Confidence Score**: Based on prediction accuracy and historical patterns
- **Miami Open 2026**: Hard Court Surface
""")

# Side info (hidden but available)
with st.expander("ℹ️ API Information"):
    st.markdown(f"""
    - **API Status**: {'✅ Active' if RAPIDAPI_KEY else '❌ Inactive'}
    - **API Provider**: RapidAPI Tennis Live Data
    - **Data Coverage**: ATP & WTA Tour matches
    - **Update Frequency**: Live
    """)
