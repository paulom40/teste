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

st.set_page_config(page_title="ATP/WTA Match Predictor", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🎾 ATP/WTA MATCH PREDICTOR")
st.markdown("Enter today/tomorrow matches and predict games (22-25 range)")
st.markdown("---")

today = datetime.now().date()
tomorrow = today + timedelta(days=1)

# SECTION 1: INPUT MATCHES
st.markdown("## 📝 STEP 1: ENTER MATCHES FOR TODAY & TOMORROW")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### 📅 TODAY ({today.strftime('%d/%m/%Y')})")
    st.info("Enter ATP and WTA matches scheduled for today")
    
    today_matches_text = st.text_area(
        "Enter matches (one per line): Player1 vs Player2 - Surface",
        placeholder="Example:\nJannik Sinner vs Carlos Alcaraz - Hard\nIga Swiatek vs Coco Gauff - Clay",
        height=150,
        key="today_matches"
    )

with col2:
    st.markdown(f"### 📅 TOMORROW ({tomorrow.strftime('%d/%m/%Y')})")
    st.info("Enter ATP and WTA matches scheduled for tomorrow")
    
    tomorrow_matches_text = st.text_area(
        "Enter matches (one per line): Player1 vs Player2 - Surface",
        placeholder="Example:\nNovak Djovic vs Daniil Medvedev - Hard\nAryna Sabalenka vs Elena Rybakina - Clay",
        height=150,
        key="tomorrow_matches"
    )

st.markdown("---")

# Parse matches
def parse_matches(text, date):
    """Parse match text into dataframe"""
    matches = []
    
    if not text.strip():
        return pd.DataFrame()
    
    for line in text.strip().split('\n'):
        if ' - ' in line and ' vs ' in line:
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

# Parse input
today_matches = parse_matches(today_matches_text, today)
tomorrow_matches = parse_matches(tomorrow_matches_text, tomorrow)

all_scheduled = pd.concat([today_matches, tomorrow_matches], ignore_index=True)

if len(all_scheduled) > 0:
    st.markdown("## ✅ SCHEDULED MATCHES")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"### 📅 TODAY ({len(today_matches)} matches)")
        if len(today_matches) > 0:
            st.dataframe(today_matches, use_container_width=True, hide_index=True)
        else:
            st.info("No matches entered for today")
    
    with col2:
        st.markdown(f"### 📅 TOMORROW ({len(tomorrow_matches)} matches)")
        if len(tomorrow_matches) > 0:
            st.dataframe(tomorrow_matches, use_container_width=True, hide_index=True)
        else:
            st.info("No matches entered for tomorrow")
    
    st.markdown("---")
    
    # SECTION 2: LOAD HISTORICAL DATA
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
    
    st.info(f"📊 Historical matches: {len(df):,}")
    
    st.markdown("---")
    
    # SECTION 3: ANALYZE BY SURFACE
    st.markdown("## 🏆 STEP 3: ANALYZE & PREDICT")
    
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
    
    available_surfaces = sorted(df_analysis['Surface'].dropna().unique())
    
    # Get surfaces from scheduled matches
    scheduled_surfaces = all_scheduled['Surface'].unique()
    st.info(f"🎾 Surfaces in your schedule: {', '.join(scheduled_surfaces)}")
    st.info(f"📊 Available in historical data: {', '.join(available_surfaces)}")
    
    st.markdown("---")
    
    # Predict for each match
    st.markdown("## 🔮 GAME PREDICTIONS (22-25 GAMES)")
    
    predictions = []
    
    for idx, match in all_scheduled.iterrows():
        surface = match['Surface']
        
        # Get surface data
        surface_data = df_analysis[df_analysis['Surface'] == surface]
        
        if len(surface_data) == 0:
            st.warning(f"⚠️ No historical data for {surface} surface. Skipping {match['Player 1']} vs {match['Player 2']}")
            continue
        
        # Train model for this surface
        @st.cache_resource(hash_funcs={pd.DataFrame: lambda x: None})
        def train_surface_model(surf):
            surf_data = df_analysis[df_analysis['Surface'] == surf].copy()
            surf_data = surf_data.dropna(subset=['Total_Games'])
            
            if len(surf_data) < 30:
                return None, None
            
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
        
        model, scaler = train_surface_model(surface)
        
        if model is None:
            continue
        
        # Generate scenarios for this match
        avg_games = surface_data['Total_Games'].mean()
        
        # Create realistic scenarios
        scenario_games = []
        
        for _ in range(10):
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
                scenario_games.append({
                    'Set1': f"{w1}-{l1}",
                    'Set2': f"{w2}-{l2}",
                    'Set3': f"{w3}-{l3}" if is_3set else "N/A",
                    'Actual': actual,
                    'Predicted': round(predicted, 1)
                })
        
        if scenario_games:
            best_scenario = max(scenario_games, key=lambda x: x['Predicted'])
            
            predictions.append({
                'Date': match['Date'],
                'Player 1': match['Player 1'],
                'Player 2': match['Player 2'],
                'Surface': surface,
                'Set 1': best_scenario['Set1'],
                'Set 2': best_scenario['Set2'],
                'Set 3': best_scenario['Set3'],
                'Predicted Games': best_scenario['Predicted'],
                '✅ Probability': "HIGH" if 22 <= best_scenario['Predicted'] <= 25 else "MEDIUM"
            })
    
    if predictions:
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.sort_values('Predicted Games', ascending=False)
        
        st.markdown(f"### ✅ {len(predictions_df)} MATCHES WITH 22-25 GAMES PREDICTION")
        st.dataframe(predictions_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("## 📥 EXPORT")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="📊 Download as CSV",
                data=csv,
                file_name=f"Predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
                        predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
                        
                        workbook = writer.book
                        worksheet = workbook['Predictions']
                        
                        header_fill = PatternFill(start_color="667eea", end_color="667eea", fill_type="solid")
                        header_font = Font(bold=True, color="FFFFFF")
                        
                        for cell in worksheet[1]:
                            if cell.value:
                                cell.fill = header_fill
                                cell.font = header_font
                    
                    output.seek(0)
                    st.download_button(
                        label="✅ Download Excel",
                        data=output.getvalue(),
                        file_name=f"Predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        st.markdown("### 📊 STATISTICS")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", len(predictions_df))
        col2.metric("Avg Games", f"{predictions_df['Predicted Games'].mean():.1f}")
        col3.metric("Min Games", f"{predictions_df['Predicted Games'].min():.1f}")
        col4.metric("Max Games", f"{predictions_df['Predicted Games'].max():.1f}")
    
    else:
        st.warning("⚠️ No predictions found in 22-25 games range")

else:
    st.info("👆 Enter matches above to get predictions")
