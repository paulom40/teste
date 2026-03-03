import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="WTA Predictor", page_icon="🎾", layout="wide")

@st.cache_resource
def load_and_train_model(csv_file):
    df = pd.read_csv(csv_file)
    
    for col in ['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['Player_1', 'Player_2', 'Winner', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2'])
    df['Player_1_Won'] = (df['Winner'] == df['Player_1']).astype(int)
    
    features = []
    feature_names = []
    
    features.append((df['Rank_2'] - df['Rank_1']).values)
    feature_names.append('Ranking_Differential')
    
    features.append((df['Pts_1'] - df['Pts_2']).values)
    feature_names.append('Points_Differential')
    
    features.append(df['Rank_1'].values)
    feature_names.append('Player_1_Rank')
    
    if 'Surface' in df.columns:
        surfaces = pd.get_dummies(df['Surface'], prefix='Surface')
        for col in surfaces.columns:
            features.append(surfaces[col].values)
            feature_names.append(col)
    
    if 'Round' in df.columns:
        rounds = pd.get_dummies(df['Round'], prefix='Round')
        for col in rounds.columns:
            features.append(rounds[col].values)
            feature_names.append(col)
    
    if 'Court' in df.columns:
        courts = pd.get_dummies(df['Court'], prefix='Court')
        for col in courts.columns:
            features.append(courts[col].values)
            feature_names.append(col)
    
    if 'Odd_1' in df.columns and 'Odd_2' in df.columns:
        features.append((df['Odd_1'] - df['Odd_2']).values)
        feature_names.append('Odds_Differential')
    
    X = np.column_stack(features)
    y = df['Player_1_Won'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    y_test_pred = model.predict(X_test_scaled)
    
    return {
        'model': model,
        'scaler': scaler,
        'df': df,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'feature_names': feature_names,
        'importance_df': pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred)
    }

def get_last_30_matches(df, player_name):
    """Get last 30 matches for a player"""
    p1_matches = df[df['Player_1'] == player_name].copy()
    p2_matches = df[df['Player_2'] == player_name].copy()
    
    # Combine and sort by index (assuming chronological order)
    all_matches = pd.concat([p1_matches, p2_matches], ignore_index=False)
    all_matches = all_matches.sort_index()
    
    # Get last 30
    last_30 = all_matches.tail(30)
    
    return last_30

def calculate_player_stats_last_30(df, player_name):
    """Calculate stats from last 30 matches"""
    last_30 = get_last_30_matches(df, player_name)
    
    if len(last_30) == 0:
        return None
    
    # Calculate wins
    wins = len(last_30[last_30['Winner'] == player_name])
    losses = len(last_30) - wins
    win_rate = wins / len(last_30) if len(last_30) > 0 else 0
    
    # Latest stats
    latest = last_30.iloc[-1]
    if player_name == latest.get('Player_1'):
        rank = latest['Rank_1']
        points = latest['Pts_1']
        odds = latest.get('Odd_1', 1.5)
    else:
        rank = latest['Rank_2']
        points = latest['Pts_2']
        odds = latest.get('Odd_2', 2.5)
    
    # Surface stats
    surface_stats = {}
    if 'Surface' in last_30.columns:
        for surface in last_30['Surface'].unique():
            surface_matches = last_30[last_30['Surface'] == surface]
            surface_wins = len(surface_matches[surface_matches['Winner'] == player_name])
            surface_rate = surface_wins / len(surface_matches) if len(surface_matches) > 0 else 0
            surface_stats[surface] = {
                'matches': len(surface_matches),
                'wins': surface_wins,
                'rate': surface_rate
            }
    
    return {
        'last_30': last_30,
        'rank': rank,
        'points': points,
        'odds': odds,
        'total_matches': len(last_30),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'surface_stats': surface_stats
    }

def calculate_game_lines(p_a_prob, player_a_name, player_b_name):
    """Calculate betting lines based on probability"""
    
    if p_a_prob >= 0.5:
        american_odds_fav = int(-100 / (1/p_a_prob - 1)) if p_a_prob < 1 else -9999
        american_odds_under = int(100 * (1/((1-p_a_prob)) - 1)) if p_a_prob > 0 else 9999
        favorite = player_a_name
        underdog = player_b_name
    else:
        american_odds_fav = int(-100 / (1/(1-p_a_prob) - 1)) if (1-p_a_prob) < 1 else -9999
        american_odds_under = int(100 * (1/(p_a_prob) - 1)) if p_a_prob > 0 else 9999
        favorite = player_b_name
        underdog = player_a_name
    
    spread = abs(p_a_prob - 0.5) * 20
    over_under = 2.5 + (abs(p_a_prob - 0.5) * 2)
    
    return {
        'favorite': favorite,
        'underdog': underdog,
        'spread': spread,
        'american_odds_fav': american_odds_fav,
        'american_odds_under': american_odds_under,
        'over_under': over_under
    }

def show_home(model_data):
    st.header("WTA Match Predictor & Game Lines")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", len(model_data['df']))
    with col2:
        st.metric("Model Accuracy", f"{model_data['test_accuracy']:.1%}")
    with col3:
        st.metric("Features", len(model_data['feature_names']))
    with col4:
        st.metric("Status", "✓ Ready")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 The 6-Factor Framework")
        st.markdown("""
        1. **Ranking Differential**
        2. **Points Differential**  
        3. **Surface Performance**
        4. **Tournament Context**
        5. **Physical Load**
        6. **Momentum**
        """)
    
    with col2:
        st.subheader("🎯 Key Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score': [model_data['test_accuracy'], model_data['precision'], model_data['recall'], model_data['f1']]
        })
        st.dataframe(metrics_df.style.format({'Score': '{:.1%}'}), use_container_width=True, hide_index=True)

def show_predictions(model_data):
    st.header("🔮 Predict & Game Lines (Last 30 Matches Analysis)")
    st.markdown("Select players to get predictions based on their last 30 matches")
    
    st.markdown("---")
    
    df = model_data['df']
    all_players = sorted(list(set(df['Player_1'].unique()) | set(df['Player_2'].unique())))
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("👤 Player A")
        player_a_name = st.selectbox("Select Player A", all_players)
        
        if player_a_name:
            stats_a = calculate_player_stats_last_30(df, player_a_name)
            if stats_a:
                col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                with col_metrics1:
                    st.metric("Last 30 Matches", stats_a['total_matches'])
                with col_metrics2:
                    st.metric("Wins", stats_a['wins'])
                with col_metrics3:
                    st.metric("Win Rate", f"{stats_a['win_rate']:.1%}")
                
                rank_1 = st.number_input("Rank", value=int(stats_a['rank']), key="rank_a")
                pts_1 = st.number_input("Points", value=int(stats_a['points']), key="pts_a")
                odds_1 = st.number_input("Odds", value=float(stats_a['odds']), step=0.1, key="odds_a")
    
    with col_b:
        st.subheader("👤 Player B")
        player_b_name = st.selectbox("Select Player B", all_players, index=1 if len(all_players) > 1 else 0)
        
        if player_b_name:
            stats_b = calculate_player_stats_last_30(df, player_b_name)
            if stats_b:
                col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                with col_metrics1:
                    st.metric("Last 30 Matches ", stats_b['total_matches'])
                with col_metrics2:
                    st.metric("Wins ", stats_b['wins'])
                with col_metrics3:
                    st.metric("Win Rate ", f"{stats_b['win_rate']:.1%}")
                
                rank_2 = st.number_input("Rank ", value=int(stats_b['rank']), key="rank_b")
                pts_2 = st.number_input("Points ", value=int(stats_b['points']), key="pts_b")
                odds_2 = st.number_input("Odds ", value=float(stats_b['odds']), step=0.1, key="odds_b")
    
    st.markdown("---")
    
    st.subheader("🏟️ Match Conditions")
    col1, col2, col3 = st.columns(3)
    with col1:
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
    with col2:
        court = st.selectbox("Court", ["Indoor", "Outdoor"])
    with col3:
        round_type = st.selectbox("Round", ["1st Round", "2nd Round", "Quarterfinal", "Semifinal", "Final"])
    
    st.markdown("---")
    
    if st.button("⚡ Predict Winner & Game Lines", use_container_width=True):
        
        features = [rank_2 - rank_1, pts_1 - pts_2, rank_1]
        
        for s in ["Hard", "Clay", "Grass"]:
            features.append(1.0 if surface == s else 0.0)
        
        for r in ["1st Round", "2nd Round", "Quarterfinal", "Semifinal", "Final"]:
            features.append(1.0 if round_type == r else 0.0)
        
        for c in ["Indoor", "Outdoor"]:
            features.append(1.0 if court == c else 0.0)
        
        features.append(odds_1 - odds_2)
        
        while len(features) < len(model_data['feature_names']):
            features.append(0.0)
        
        X_new = np.array(features[:len(model_data['feature_names'])]).reshape(1, -1)
        X_new_scaled = model_data['scaler'].transform(X_new)
        
        prob = model_data['model'].predict_proba(X_new_scaled)[0]
        p_a = prob[1]
        p_b = prob[0]
        
        lines = calculate_game_lines(p_a, player_a_name, player_b_name)
        
        st.markdown("---")
        st.subheader("📊 PREDICTION RESULTS")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if p_a > p_b:
                st.success(f"🏆 {player_a_name}")
                st.metric("Win Probability", f"{p_a:.1%}")
            else:
                st.success(f"🏆 {player_b_name}")
                st.metric("Win Probability", f"{p_b:.1%}")
        
        with col2:
            st.metric("Confidence Level", f"{abs(p_a - 0.5):.1%}")
            if abs(p_a - 0.5) < 0.05:
                st.warning("Toss-up")
            elif abs(p_a - 0.5) < 0.15:
                st.info("Moderate")
            elif abs(p_a - 0.5) < 0.30:
                st.success("High")
            else:
                st.success("Very High")
        
        with col3:
            st.metric("Model Accuracy", f"{model_data['test_accuracy']:.1%}")
        
        st.markdown("---")
        st.subheader("📈 GAME LINES")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("SPREAD", f"{lines['favorite']} -{lines['spread']:.1f}")
            st.caption(f"{lines['favorite']} favored by {lines['spread']:.1f} games")
        
        with col2:
            st.metric("OVER/UNDER", f"{lines['over_under']:.1f}")
            st.caption("Total games in match")
        
        with col3:
            st.metric("MONEYLINE", f"{lines['american_odds_fav']}")
            st.caption(f"{lines['underdog']}: +{lines['american_odds_under']}")
        
        st.markdown("---")
        st.subheader("📊 LAST 30 MATCHES ANALYSIS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{player_a_name} - Last 30 Matches")
            if stats_a:
                st.write(f"**Total Matches:** {stats_a['total_matches']}")
                st.write(f"**Wins:** {stats_a['wins']}")
                st.write(f"**Losses:** {stats_a['losses']}")
                st.write(f"**Win Rate:** {stats_a['win_rate']:.1%}")
                
                if stats_a['surface_stats']:
                    st.write("**Surface Performance:**")
                    for surface, s_stats in stats_a['surface_stats'].items():
                        st.write(f"- {surface}: {s_stats['wins']}/{s_stats['matches']} ({s_stats['rate']:.1%})")
                
                # Display last 10 matches
                last_10 = stats_a['last_30'].tail(10)
                st.write("**Last 10 Matches:**")
                for idx, (_, match) in enumerate(last_10.iterrows(), 1):
                    opponent = match['Player_2'] if match['Player_1'] == player_a_name else match['Player_1']
                    result = "✓ Win" if match['Winner'] == player_a_name else "✗ Loss"
                    st.write(f"{idx}. vs {opponent} - {result} ({match['Surface']})")
        
        with col2:
            st.subheader(f"{player_b_name} - Last 30 Matches")
            if stats_b:
                st.write(f"**Total Matches:** {stats_b['total_matches']}")
                st.write(f"**Wins:** {stats_b['wins']}")
                st.write(f"**Losses:** {stats_b['losses']}")
                st.write(f"**Win Rate:** {stats_b['win_rate']:.1%}")
                
                if stats_b['surface_stats']:
                    st.write("**Surface Performance:**")
                    for surface, s_stats in stats_b['surface_stats'].items():
                        st.write(f"- {surface}: {s_stats['wins']}/{s_stats['matches']} ({s_stats['rate']:.1%})")
                
                # Display last 10 matches
                last_10 = stats_b['last_30'].tail(10)
                st.write("**Last 10 Matches:**")
                for idx, (_, match) in enumerate(last_10.iterrows(), 1):
                    opponent = match['Player_2'] if match['Player_1'] == player_b_name else match['Player_1']
                    result = "✓ Win" if match['Winner'] == player_b_name else "✗ Loss"
                    st.write(f"{idx}. vs {opponent} - {result} ({match['Surface']})")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Win Probability")
            fig = go.Figure([go.Bar(x=[player_a_name, player_b_name], y=[p_a, p_b], marker_color=['#667eea', '#764ba2'])])
            fig.update_layout(title="Probability Comparison", yaxis=dict(range=[0, 1]), showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Comparison Table")
            comp = pd.DataFrame({
                'Metric': ['Rank', 'Points', 'Last 30 W/L', 'Win Rate', 'P(Win)'],
                player_a_name: [f"#{rank_1}", pts_1, f"{stats_a['wins']}/{stats_a['total_matches']}", f"{stats_a['win_rate']:.1%}", f"{p_a:.1%}"],
                player_b_name: [f"#{rank_2}", pts_2, f"{stats_b['wins']}/{stats_b['total_matches']}", f"{stats_b['win_rate']:.1%}", f"{p_b:.1%}"]
            })
            st.dataframe(comp, use_container_width=True, hide_index=True)

def main():
    st.sidebar.title("🎾 WTA Predictor")
    page = st.sidebar.radio("Page", ["🏠 Home", "🔮 Predict & Lines"])
    
    st.sidebar.title("📁 Upload")
    uploaded_file = st.sidebar.file_uploader("CSV", type=['csv'])
    
    if uploaded_file:
        model_data = load_and_train_model(uploaded_file)
        st.sidebar.success("✓ Ready!")
        
        if page == "🏠 Home":
            show_home(model_data)
        else:
            show_predictions(model_data)
    else:
        st.title("🎾 WTA Predictor")
        st.info("👈 Upload CSV to start!")

if __name__ == "__main__":
    main()
