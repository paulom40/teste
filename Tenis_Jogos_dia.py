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

st.set_page_config(page_title="ATP/WTA Smart Predictor", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🎾 ATP/WTA SMART GAMES PREDICTOR")
st.markdown("Predict matches with 22-25 total games based on surface analysis")
st.markdown("---")

# LOAD DATA
st.markdown("## 📥 LOADING MATCH DATA")

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
    except Exception as e:
        st.warning(f"Could not load WTA")
    
    try:
        response = requests.get(atp_url, timeout=15)
        response.raise_for_status()
        atp_df = pd.read_excel(BytesIO(response.content))
        atp_df['Tour'] = 'ATP'
        st.success("✅ ATP data loaded")
        dfs.append(atp_df)
    except Exception as e:
        st.warning(f"Could not load ATP")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

# Load historical data
df = load_github_data()

if df is None:
    st.error("❌ Could not load data")
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
st.markdown("## 🔍 ANALYZING HISTORICAL DATA")

df_analysis = df.copy()
df_analysis['Total_Games'] = df_analysis.apply(calculate_total_games, axis=1)
df_analysis = df_analysis.dropna(subset=['Total_Games'])

# Get available surfaces
available_surfaces = sorted(df_analysis['Surface'].dropna().unique())
st.info(f"📊 Historical matches analyzed: {len(df_analysis):,}")
st.info(f"🏆 Surfaces found: {', '.join(available_surfaces)}")

st.markdown("---")

# STEP 1: Select Surface
st.markdown("## 🏆 STEP 1: SELECT SURFACE")

surface = st.selectbox(
    "Choose surface for analysis",
    available_surfaces,
    key="surface"
)

# Analyze games by surface
surface_matches = df_analysis[df_analysis['Surface'] == surface].copy()

st.info(f"📊 Matches on {surface}: {len(surface_matches):,}")

if len(surface_matches) > 0:
    avg_games = surface_matches['Total_Games'].mean()
    min_games_hist = surface_matches['Total_Games'].min()
    max_games_hist = surface_matches['Total_Games'].max()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Avg Games ({surface})", f"{avg_games:.1f}")
    col2.metric(f"Min Games", int(min_games_hist))
    col3.metric(f"Max Games", int(max_games_hist))
    col4.metric(f"Match Count", len(surface_matches))
    
    st.markdown("---")
    
    # STEP 2: Train prediction model
    st.markdown("## 🧠 STEP 2: TRAINING PREDICTION MODEL")
    
    @st.cache_resource
    def train_model(df_train, surf):
        """Train model on surface-specific data"""
        df_model = df_train[df_train['Surface'] == surf].copy()
        df_model = df_model.dropna(subset=['Total_Games'])
        
        if len(df_model) < 30:
            return None, None, None
        
        # Features
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
    
    with st.spinner("Training model..."):
        model, scaler, model_df = train_model(df_analysis, surface)
    
    if model is None:
        st.error(f"❌ Not enough data for {surface}")
    else:
        st.success(f"✅ Model trained on {len(model_df):,} {surface} matches")
        
        st.markdown("---")
        
        # STEP 3: Select tour and generate scenarios
        st.markdown("## 🎾 STEP 3: GENERATE MATCH SCENARIOS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tour = st.selectbox("Select tour", ['ATP', 'WTA'], key="tour")
        
        with col2:
            num_scenarios = st.slider("Generate N scenarios", 5, 50, 20, key="scenarios")
        
        st.markdown("---")
        
        # STEP 4: Generate scenarios
        st.markdown("## 🔮 STEP 4: PREDICT MATCHES (22-25 GAMES)")
        
        if st.button("🎯 Generate Match Predictions", use_container_width=True, key="generate"):
            with st.spinner("Generating match scenarios..."):
                # Get top players from selected tour
                tour_data = df_analysis[df_analysis['Tour'] == tour].copy()
                players = pd.concat([tour_data['Winner'], tour_data['Loser']]).unique()
                players = [p for p in players if pd.notna(p)]
                
                if len(players) < 2:
                    st.error("Not enough players")
                else:
                    # Generate scenarios
                    scenarios = []
                    np.random.seed(42)
                    
                    for _ in range(num_scenarios):
                        p1, p2 = np.random.choice(players, 2, replace=False)
                        
                        # Simulate realistic scores
                        w1 = np.random.randint(4, 7)
                        l1 = np.random.randint(2, 7)
                        w2 = np.random.randint(4, 7)
                        l2 = np.random.randint(2, 7)
                        
                        # 80% 2-set, 20% 3-set
                        is_3set = np.random.random() < 0.2
                        w3 = np.random.randint(6, 8) if is_3set else 0
                        l3 = np.random.randint(2, 6) if is_3set else 0
                        
                        # Predict games
                        X_scenario = np.array([[
                            w1 + l1,
                            w2 + l2,
                            1.0 if not is_3set else 0.0,
                            1.0 if is_3set else 0.0,
                            1 / (1 + abs(w1 - l1) + abs(w2 - l2))
                        ]])
                        
                        X_scenario_scaled = scaler.transform(X_scenario)
                        predicted_games = model.predict(X_scenario_scaled)[0]
                        
                        # Only include if 22-25 games
                        if 22 <= predicted_games <= 25:
                            scenarios.append({
                                'Tour': tour,
                                'Player 1': p1,
                                'Player 2': p2,
                                'Surface': surface,
                                'Set1': f"{w1}-{l1}",
                                'Set2': f"{w2}-{l2}",
                                'Set3': f"{w3}-{l3}" if is_3set else "N/A",
                                'Actual Games': w1 + l1 + w2 + l2 + (w3 + l3 if is_3set else 0),
                                'Predicted Games': round(predicted_games, 1),
                                'Probability': "✅ HIGH"
                            })
                    
                    if scenarios:
                        scenarios_df = pd.DataFrame(scenarios)
                        scenarios_df = scenarios_df.sort_values('Predicted Games', ascending=False)
                        
                        st.markdown(f"### ✅ Found {len(scenarios_df)} matches with 22-25 games prediction")
                        
                        st.dataframe(scenarios_df, use_container_width=True, hide_index=True)
                        
                        st.markdown("---")
                        st.markdown("## 📥 EXPORT OPTIONS")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("📊 Download as Excel", use_container_width=True, key="export_excel"):
                                try:
                                    import openpyxl
                                    from openpyxl.styles import Font, PatternFill, Alignment
                                    
                                    output = BytesIO()
                                    
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        summary_data = {
                                            'Metric': [
                                                'Generated',
                                                'Tour',
                                                'Surface',
                                                'Avg Games (Historical)',
                                                'Predicted Games Range',
                                                'Match Scenarios Found',
                                                'Historical Data Points'
                                            ],
                                            'Value': [
                                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                tour,
                                                surface,
                                                f"{avg_games:.1f}",
                                                "22-25",
                                                len(scenarios_df),
                                                len(model_df)
                                            ]
                                        }
                                        summary_df = pd.DataFrame(summary_data)
                                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                        
                                        scenarios_df.to_excel(writer, sheet_name='Predictions', index=False)
                                        
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
                                                        max_length = max(max_length, len(str(cell.value)))
                                                    except:
                                                        pass
                                                worksheet.column_dimensions[column_letter].width = min(max_length + 2, 50)
                                    
                                    output.seek(0)
                                    st.download_button(
                                        label="📊 Download Excel Report",
                                        data=output.getvalue(),
                                        file_name=f"ATP_WTA_Predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True
                                    )
                                    st.success("✅ Excel ready!")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                        
                        with col2:
                            csv = scenarios_df.to_csv(index=False)
                            st.download_button(
                                label="📄 Download as CSV",
                                data=csv,
                                file_name=f"ATP_WTA_Predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        st.markdown("---")
                        st.markdown("### 📊 STATISTICS")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Avg Predicted Games", f"{scenarios_df['Predicted Games'].mean():.1f}")
                        col2.metric("Min Predicted", f"{scenarios_df['Predicted Games'].min():.1f}")
                        col3.metric("Max Predicted", f"{scenarios_df['Predicted Games'].max():.1f}")
                        col4.metric("Scenarios Found", len(scenarios_df))
                    
                    else:
                        st.warning(f"⚠️ No matches found with 22-25 games. Try different parameters.")

else:
    st.warning(f"No historical data for {surface}")
