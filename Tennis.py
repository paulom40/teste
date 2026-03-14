import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import requests
from io import BytesIO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="WTA Advanced Predictor", page_icon="🎾", layout="wide")

def fetch_wta_github_data():
    try:
        url = "https://github.com/paulom40/teste/raw/main/wta_data.xlsx"
        response = requests.get(url, timeout=10)
        df = pd.read_excel(BytesIO(response.content))
        return df, "GitHub WTA Database"
    except Exception as e:
        st.warning(f"Could not fetch GitHub data: {str(e)}")
        return None, None

def load_custom_excel(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded: {uploaded_file.name}")
        return df, uploaded_file.name
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None, None

def calculate_total_games(row):
    total = 0
    for i in range(1, 6):
        w = row.get(f'W{i}', 0)
        l = row.get(f'L{i}', 0)
        if pd.notna(w) and pd.notna(l) and w > 0 and l > 0:
            total += int(w) + int(l)
    return total if total > 0 else None

def calculate_mean_stats_from_last_15(df, player_name, surface):
    """Calculate MEAN statistics from last 15 games on surface"""
    last_15 = df[
        ((df['Winner'] == player_name) | (df['Loser'] == player_name)) &
        (df['Surface'] == surface)
    ].tail(15).copy()
    
    if len(last_15) == 0:
        return {
            'winners': 0, 'unforced_errors': 0, 'net_points_won': 0,
            'service_points_won': 0, 'return_points_won': 0, 'total_points_won': 0,
            'break_points_converted': 0, 'first_serve_percentage': 0, 'matches_analyzed': 0
        }
    
    last_15['Total_Games'] = last_15.apply(calculate_total_games, axis=1)
    last_15['W1'] = pd.to_numeric(last_15['W1'], errors='coerce')
    last_15['L1'] = pd.to_numeric(last_15['L1'], errors='coerce')
    last_15['WRank'] = pd.to_numeric(last_15['WRank'], errors='coerce')
    last_15['LRank'] = pd.to_numeric(last_15['LRank'], errors='coerce')
    
    stats = {}
    
    if len(last_15) > 0:
        last_15['is_winner'] = last_15['Winner'] == player_name
        last_15['player_rank'] = last_15.apply(
            lambda row: row['WRank'] if row['is_winner'] else row['LRank'], axis=1
        )
        last_15['opponent_rank'] = last_15.apply(
            lambda row: row['LRank'] if row['is_winner'] else row['WRank'], axis=1
        )
        
        last_15['rank_diff'] = last_15['opponent_rank'] - last_15['player_rank']
        mean_rank_diff = last_15['rank_diff'].mean()
        mean_winners = 10 + (min(mean_rank_diff, 150) / 150) * 25
        stats['winners'] = int(round(mean_winners))
        
        mean_ue = 25 - (mean_winners - 10) * 0.5
        stats['unforced_errors'] = int(round(mean_ue))
        
        mean_total_games = last_15['Total_Games'].mean()
        mean_net_points = 15 + (mean_total_games / 40) * 30
        stats['net_points_won'] = int(round(mean_net_points))
        
        service_points_list = []
        for idx, row in last_15.iterrows():
            if row['is_winner']:
                service_points_list.append(75 if row['Wsets'] == 2 else 60)
            else:
                service_points_list.append(45 if row['Wsets'] == 2 else 55)
        mean_service_points = np.mean(service_points_list) if service_points_list else 55
        stats['service_points_won'] = int(round(mean_service_points))
        
        w1_values = last_15['W1'].dropna()
        l1_values = last_15['L1'].dropna()
        if len(w1_values) > 0 and len(l1_values) > 0:
            set1_win_ratio = w1_values.mean() / (w1_values.mean() + l1_values.mean())
        else:
            set1_win_ratio = 0.6
        mean_return_points = 30 + set1_win_ratio * 35
        stats['return_points_won'] = int(round(mean_return_points))
        
        stats['total_points_won'] = int(round((stats['service_points_won'] + stats['return_points_won']) / 2))
        
        break_points_list = []
        for idx, row in last_15.iterrows():
            if row['is_winner']:
                break_points_list.append(60 if row['Wsets'] == 3 else 30)
            else:
                break_points_list.append(20 if row['Wsets'] == 3 else 40)
        mean_break_points = np.mean(break_points_list) if break_points_list else 40
        stats['break_points_converted'] = int(round(mean_break_points))
        
        mean_rank = last_15['player_rank'].mean()
        mean_first_serve = 45 + (min(mean_rank, 100) / 100) * 30
        stats['first_serve_percentage'] = int(round(mean_first_serve))
        
        stats['matches_analyzed'] = len(last_15)
    
    return stats

def analyze_last_15_surface_games(df, player_name, surface):
    matches = df[
        ((df['Winner'] == player_name) | (df['Loser'] == player_name)) &
        (df['Surface'] == surface)
    ].tail(15).sort_values('Date', ascending=False)
    
    if len(matches) == 0:
        return {'matches': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.5, 'avg_games': 22, 'form': 'No Data'}
    
    matches = matches.copy()
    matches['Total_Games'] = matches.apply(calculate_total_games, axis=1)
    matches = matches.dropna(subset=['Total_Games'])
    
    if len(matches) == 0:
        return {'matches': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.5, 'avg_games': 22, 'form': 'No Data'}
    
    wins = len(matches[matches['Winner'] == player_name])
    losses = len(matches[matches['Loser'] == player_name])
    avg_games = matches['Total_Games'].mean()
    
    if wins >= 11:
        form = "🔥 Excellent"
    elif wins >= 8:
        form = "✓ Good"
    elif wins >= 5:
        form = "⚠️ Mixed"
    else:
        form = "❌ Poor"
    
    return {'matches': len(matches), 'wins': wins, 'losses': losses, 'win_rate': wins / len(matches) if len(matches) > 0 else 0.5, 'avg_games': avg_games, 'form': form}

def calculate_fatigue(df, player_name):
    matches = df[(df['Winner'] == player_name) | (df['Loser'] == player_name)].sort_values('Date', ascending=False)
    if len(matches) == 0:
        return {'days_rest': 0, 'fatigue_level': 'Unknown'}
    try:
        days_rest = (pd.Timestamp.now() - pd.to_datetime(matches.iloc[0]['Date'])).days
    except:
        days_rest = 0
    if days_rest >= 7:
        level = "✓ Fresh"
    elif days_rest >= 4:
        level = "⚔️ Normal"
    elif days_rest >= 2:
        level = "⚠️ Tired"
    else:
        level = "🔴 Exhausted"
    return {'days_rest': days_rest, 'fatigue_level': level}

def analyze_player_skills(df, player_name, surface):
    matches = df[((df['Winner'] == player_name) | (df['Loser'] == player_name)) & (df['Surface'] == surface)].tail(20)
    if len(matches) == 0:
        return {'serve_strength': 0.5, 'consistency': 0.5, 'aggression': 0.5}
    wins = matches[matches['Winner'] == player_name]
    serve_strength = min(0.9, 0.5 + (len(wins) / len(matches) * 0.4)) if len(matches) > 0 else 0.5
    matches = matches.copy()
    matches['Total_Games'] = matches.apply(calculate_total_games, axis=1)
    avg_games = matches['Total_Games'].mean()
    aggression = np.clip((avg_games - 15) / 20, 0.1, 0.9)
    consistency = min(0.9, 0.5 + (serve_strength * 0.4))
    return {'serve_strength': serve_strength, 'consistency': consistency, 'aggression': aggression}

@st.cache_resource
def build_model(df):
    df_train = df.copy()
    df_train['Total_Games'] = df_train.apply(calculate_total_games, axis=1)
    df_train = df_train.dropna(subset=['Total_Games'])
    df_train = df_train[(df_train['Total_Games'] > 0) & (df_train['Total_Games'] < 50)]
    if len(df_train) < 100:
        st.error("Not enough training data")
        return None
    features = []
    w1 = pd.to_numeric(df_train['W1'], errors='coerce').fillna(0).values
    l1 = pd.to_numeric(df_train['L1'], errors='coerce').fillna(0).values
    w2 = pd.to_numeric(df_train['W2'], errors='coerce').fillna(0).values
    l2 = pd.to_numeric(df_train['L2'], errors='coerce').fillna(0).values
    w3 = pd.to_numeric(df_train['W3'], errors='coerce').fillna(0).values
    l3 = pd.to_numeric(df_train['L3'], errors='coerce').fillna(0).values
    features.append(w1 + l1)
    features.append(w2 + l2)
    features.append(np.where(w3 + l3 > 0, w3 + l3, 0))
    features.append((df_train['Wsets'] == 2).astype(float).values)
    features.append((df_train['Wsets'] == 3).astype(float).values)
    rank_diff = df_train['LRank'] - df_train['WRank']
    features.append(rank_diff.fillna(0).values)
    competitiveness = 1 / (1 + np.abs(w1 - l1) + np.abs(w2 - l2))
    features.append(competitiveness)
    if 'Surface' in df_train.columns:
        for surface in df_train['Surface'].dropna().unique():
            is_surface = (df_train['Surface'] == surface).astype(int).values
            features.append(is_surface)
    X = np.column_stack(features)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = df_train['Total_Games'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.02, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return {'model': model, 'scaler': scaler, 'r2': r2, 'mae': mae, 'df': df_train}

def main():
    st.sidebar.title("🎾 WTA Advanced Predictor")
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 DATA SOURCE")
    
    # PROMINENT FILE UPLOAD
    st.sidebar.markdown("### 📥 UPLOAD YOUR EXCEL FILE")
    st.sidebar.markdown("Select an Excel file (.xlsx or .xls)")
    uploaded_file = st.sidebar.file_uploader(
        "Choose file",
        type=['xlsx', 'xls'],
        label_visibility="collapsed",
        help="Columns needed: Winner, Loser, W1-W5, L1-L5, WRank, LRank, Surface, Date, Wsets"
    )
    
    df = None
    source_name = ""
    
    if uploaded_file is not None:
        df, source_name = load_custom_excel(uploaded_file)
    else:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🌐 OR USE GITHUB DATA")
        if st.sidebar.button("📥 Load Default WTA Database", use_container_width=True, key="github_btn"):
            with st.spinner("📥 Loading GitHub data..."):
                df, source_name = fetch_wta_github_data()
            if df is None:
                st.error("❌ Could not load GitHub data")
                return
        else:
            st.sidebar.info("📤 Upload an Excel file above\nOR\n📥 Click button to load GitHub data")
            return
    
    if df is None:
        return
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**✅ Data Loaded:**\n{source_name}")
    st.sidebar.metric("📊 Matches", len(df))
    
    with st.spinner("🔧 Training ML model..."):
        build_model.clear()
        model_data = build_model(df)
    
    if model_data is None:
        return
    
    st.sidebar.success("✅ Model Ready!")
    st.sidebar.metric("R² Score", f"{model_data['r2']:.3f}")
    st.sidebar.metric("Accuracy", f"±{model_data['mae']:.2f} games")
    
    st.header("🎾 WTA Advanced Match Predictor")
    st.markdown(f"**Source:** {source_name} • **Analysis:** Last 15 Games")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        all_players = sorted(list(set(df['Winner'].unique()) | set(df['Loser'].unique())))
        player_a = st.selectbox("🎾 Player 1", all_players, key="p1")
    with col2:
        player_b = st.selectbox("🎾 Player 2", all_players, index=1 if len(all_players) > 1 else 0, key="p2")
    with col3:
        surfaces = sorted(df['Surface'].dropna().unique())
        surface = st.selectbox("🏆 Surface", surfaces, key="surf")
    
    st.markdown("---")
    
    if st.button("🔮 PREDICT MATCH", use_container_width=True, key="predict_btn"):
        with st.spinner("Analyzing players..."):
            analysis_a = {
                'last15': analyze_last_15_surface_games(df, player_a, surface),
                'fatigue': calculate_fatigue(df, player_a),
                'skills': analyze_player_skills(df, player_a, surface)
            }
            analysis_b = {
                'last15': analyze_last_15_surface_games(df, player_b, surface),
                'fatigue': calculate_fatigue(df, player_b),
                'skills': analyze_player_skills(df, player_b, surface)
            }
            with st.spinner("Calculating MEAN statistics..."):
                stats_a = calculate_mean_stats_from_last_15(df, player_a, surface)
                stats_b = calculate_mean_stats_from_last_15(df, player_b, surface)
        
        st.markdown("---")
        st.subheader("📊 COMPLETE MATCH ANALYSIS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"🎾 {player_a}")
            st.write(f"**Last 15 Games:** {analysis_a['last15']['wins']}-{analysis_a['last15']['losses']} {analysis_a['last15']['form']}")
            st.write(f"**Avg Games/Match:** {analysis_a['last15']['avg_games']:.1f}")
            st.write(f"**Fatigue:** {analysis_a['fatigue']['fatigue_level']} ({analysis_a['fatigue']['days_rest']} days rest)")
            
            st.write("\n**📊 MEAN Statistics (Last 15 Games):**")
            st.write(f"• Winners: **{stats_a['winners']}**")
            st.write(f"• Unforced Errors: **{stats_a['unforced_errors']}**")
            st.write(f"• Net Points Won: **{stats_a['net_points_won']}**")
            st.write(f"• Service Points Won: **{stats_a['service_points_won']}%**")
            st.write(f"• Return Points Won: **{stats_a['return_points_won']}%**")
            st.write(f"• Total Points Won: **{stats_a['total_points_won']}%**")
            st.write(f"• Break Points Converted: **{stats_a['break_points_converted']}%**")
            st.write(f"• First Serve %: **{stats_a['first_serve_percentage']}%**")
            
            st.write("\n**⚡ Skills:**")
            st.write(f"• Serve: {int(analysis_a['skills']['serve_strength']*100)}% | Consistency: {int(analysis_a['skills']['consistency']*100)}% | Aggression: {int(analysis_a['skills']['aggression']*100)}%")
        
        with col2:
            st.subheader(f"🎾 {player_b}")
            st.write(f"**Last 15 Games:** {analysis_b['last15']['wins']}-{analysis_b['last15']['losses']} {analysis_b['last15']['form']}")
            st.write(f"**Avg Games/Match:** {analysis_b['last15']['avg_games']:.1f}")
            st.write(f"**Fatigue:** {analysis_b['fatigue']['fatigue_level']} ({analysis_b['fatigue']['days_rest']} days rest)")
            
            st.write("\n**📊 MEAN Statistics (Last 15 Games):**")
            st.write(f"• Winners: **{stats_b['winners']}**")
            st.write(f"• Unforced Errors: **{stats_b['unforced_errors']}**")
            st.write(f"• Net Points Won: **{stats_b['net_points_won']}**")
            st.write(f"• Service Points Won: **{stats_b['service_points_won']}%**")
            st.write(f"• Return Points Won: **{stats_b['return_points_won']}%**")
            st.write(f"• Total Points Won: **{stats_b['total_points_won']}%**")
            st.write(f"• Break Points Converted: **{stats_b['break_points_converted']}%**")
            st.write(f"• First Serve %: **{stats_b['first_serve_percentage']}%**")
            
            st.write("\n**⚡ Skills:**")
            st.write(f"• Serve: {int(analysis_b['skills']['serve_strength']*100)}% | Consistency: {int(analysis_b['skills']['consistency']*100)}% | Aggression: {int(analysis_b['skills']['aggression']*100)}%")

if __name__ == "__main__":
    main()
