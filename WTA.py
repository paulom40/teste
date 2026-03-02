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
    """Load CSV and train Random Forest model"""
    
    df = pd.read_csv(csv_file)
    
    # Convert numeric
    for col in ['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean
    df = df.dropna(subset=['Player_1', 'Player_2', 'Winner', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2'])
    df['Player_1_Won'] = (df['Winner'] == df['Player_1']).astype(int)
    
    # Features
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
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_test_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    # Importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return {
        'model': model,
        'scaler': scaler,
        'df': df,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'feature_names': feature_names,
        'importance_df': importance_df,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def get_player_stats(df, player_name):
    """Get player stats from dataframe"""
    p1_matches = df[df['Player_1'] == player_name]
    p2_matches = df[df['Player_2'] == player_name]
    
    all_matches = pd.concat([p1_matches, p2_matches], ignore_index=True)
    
    if len(all_matches) == 0:
        return None
    
    latest_match = all_matches.iloc[-1]
    
    if player_name == latest_match.get('Player_1'):
        return {
            'rank': latest_match['Rank_1'],
            'points': latest_match['Pts_1'],
            'odds': latest_match.get('Odd_1', 1.5),
            'matches': len(all_matches),
            'wins': len(all_matches[all_matches['Winner'] == player_name])
        }
    else:
        return {
            'rank': latest_match['Rank_2'],
            'points': latest_match['Pts_2'],
            'odds': latest_match.get('Odd_2', 2.5),
            'matches': len(all_matches),
            'wins': len(all_matches[all_matches['Winner'] == player_name])
        }

def show_home(model_data):
    st.header("Welcome to WTA Match Predictor")
    st.markdown("*Powered by Modern Machine Learning*")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Matches", len(model_data['df']))
    with col2:
        st.metric("Accuracy", f"{model_data['test_accuracy']:.1%}")
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
    
    st.markdown("---")
    st.subheader("📊 Data Distribution")
    
    col1, col2, col3 = st.columns(3)
    df = model_data['df']
    
    with col1:
        if 'Surface' in df.columns:
            surface_counts = df['Surface'].value_counts()
            fig = px.pie(values=surface_counts.values, names=surface_counts.index, title="By Surface")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Round' in df.columns:
            round_counts = df['Round'].value_counts()
            fig = px.bar(x=round_counts.index, y=round_counts.values, title="By Round")
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        p1_wins = model_data['y'].sum()
        p2_wins = len(model_data['y']) - p1_wins
        fig = px.pie(values=[p1_wins, p2_wins], names=['P1 Wins', 'P2 Wins'], title="Win Distribution")
        st.plotly_chart(fig, use_container_width=True)

def show_training(model_data):
    st.header("📈 Model Training & Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training", len(model_data['X_train']))
    with col2:
        st.metric("Testing", len(model_data['X_test']))
    with col3:
        st.metric("Accuracy", f"{model_data['test_accuracy']:.1%}")
    with col4:
        st.metric("Improvement", f"+{(model_data['test_accuracy'] - 0.5)*100:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        metrics = {
            'Accuracy': model_data['test_accuracy'],
            'Precision': model_data['precision'],
            'Recall': model_data['recall'],
            'F1-Score': model_data['f1']
        }
        fig = go.Figure(data=[go.Bar(x=list(metrics.keys()), y=list(metrics.values()), marker_color='#667eea')])
        fig.update_layout(title="Model Performance", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 15 Features")
        top_features = model_data['importance_df'].head(15)
        fig = go.Figure(data=[go.Bar(y=top_features['Feature'], x=top_features['Importance'], orientation='h', marker_color='#764ba2')])
        fig.update_layout(title="Feature Importance", xaxis_title="Importance", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("All Features")
    st.dataframe(model_data['importance_df'].style.format({'Importance': '{:.4f}'}), use_container_width=True, hide_index=True)

def show_predictions(model_data):
    st.header("🔮 Match Prediction")
    st.markdown("Select players from dataset and match conditions")
    
    st.markdown("---")
    
    # Get unique players from dataset
    df = model_data['df']
    all_players = sorted(list(set(df['Player_1'].unique()) | set(df['Player_2'].unique())))
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("👤 Player A")
        player_a_name = st.selectbox("Select Player A", all_players)
        
        # Get Player A stats from dataset
        if player_a_name:
            stats_a = get_player_stats(df, player_a_name)
            if stats_a:
                st.metric("Matches", stats_a['matches'])
                st.metric("Wins", stats_a['wins'])
                
                rank_1 = st.number_input("Rank", value=int(stats_a['rank']), min_value=1, max_value=1000, key="rank_a")
                pts_1 = st.number_input("Points", value=int(stats_a['points']), min_value=0, max_value=10000, key="pts_a")
                odds_1 = st.number_input("Odds", value=float(stats_a['odds']), min_value=1.0, max_value=100.0, step=0.1, key="odds_a")
    
    with col_b:
        st.subheader("👤 Player B")
        player_b_name = st.selectbox("Select Player B", all_players, index=1 if len(all_players) > 1 else 0)
        
        # Get Player B stats from dataset
        if player_b_name:
            stats_b = get_player_stats(df, player_b_name)
            if stats_b:
                st.metric("Matches ", stats_b['matches'])
                st.metric("Wins ", stats_b['wins'])
                
                rank_2 = st.number_input("Rank ", value=int(stats_b['rank']), min_value=1, max_value=1000, key="rank_b")
                pts_2 = st.number_input("Points ", value=int(stats_b['points']), min_value=0, max_value=10000, key="pts_b")
                odds_2 = st.number_input("Odds ", value=float(stats_b['odds']), min_value=1.0, max_value=100.0, step=0.1, key="odds_b")
    
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
    
    if st.button("⚡ Predict Winner", use_container_width=True):
        
        # Build features
        features = []
        features.append(rank_2 - rank_1)
        features.append(pts_1 - pts_2)
        features.append(rank_1)
        
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
        conf = abs(p_a - 0.5)
        
        st.markdown("---")
        
        st.subheader("📊 Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if p_a > p_b:
                st.success(f"🏆 {player_a_name}")
                st.metric("Probability", f"{p_a:.1%}")
            else:
                st.success(f"🏆 {player_b_name}")
                st.metric("Probability", f"{p_b:.1%}")
        
        with col2:
            st.metric("Confidence", f"{conf:.1%}")
            if conf < 0.05:
                st.warning("Toss-up")
            elif conf < 0.15:
                st.info("Moderate")
            elif conf < 0.30:
                st.success("High")
            else:
                st.success("Very High")
        
        with col3:
            st.metric("Model Accuracy", f"{model_data['test_accuracy']:.1%}")
        
        st.markdown("---")
        
        st.subheader("⚖️ Match Info")
        st.info(f"**{player_a_name} vs {player_b_name}** | {surface} Court | {round_type} | {court}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure([go.Bar(x=[player_a_name, player_b_name], y=[p_a, p_b], marker_color=['#667eea', '#764ba2'], text=[f'{p_a:.1%}', f'{p_b:.1%}'], textposition='auto')])
            fig.update_layout(title="Win Probability", yaxis=dict(range=[0, 1]), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Comparison")
            comp = pd.DataFrame({
                'Metric': ['Rank', 'Points', 'Odds', 'P(Win)'],
                player_a_name: [f"#{rank_1}", pts_1, f"{odds_1:.2f}", f"{p_a:.1%}"],
                player_b_name: [f"#{rank_2}", pts_2, f"{odds_2:.2f}", f"{p_b:.1%}"]
            })
            st.dataframe(comp, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.subheader("📈 Analysis")
        
        winner = player_a_name if p_a > 0.5 else player_b_name
        win_prob = p_a if p_a > 0.5 else p_b
        loser = player_b_name if p_a > 0.5 else player_a_name
        
        st.success(f"""
        **Prediction:** {winner} favored at {win_prob:.1%} over {loser}
        
        **Factors:**
        1. Ranking: {abs(rank_2 - rank_1)} position difference
        2. Form: {player_a_name} {pts_1} pts vs {player_b_name} {pts_2} pts
        3. Surface: {surface}
        4. Round: {round_type}
        5. Court: {court}
        6. Market: {player_a_name if odds_1 < odds_2 else player_b_name} favored
        """)
        
        st.markdown("---")
        
        st.subheader("🎯 Interpretation")
        
        if conf < 0.05:
            st.warning("**50-55%**: Nearly even - unpredictable match")
        elif conf < 0.15:
            st.info("**55-65%**: Slight favorite - upset possible")
        elif conf < 0.30:
            st.success("**65-80%**: Clear favorite - strong advantage")
        else:
            st.success("**80%+**: Dominant - overwhelming advantage")

def show_analytics(model_data):
    st.header("📊 Analytics")
    
    df = model_data['df']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ranking Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['Rank_1'], name='P1', nbinsx=30, marker_color='#667eea'))
        fig.add_trace(go.Histogram(x=df['Rank_2'], name='P2', nbinsx=30, marker_color='#764ba2'))
        fig.update_layout(barmode='overlap')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Points Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['Pts_1'], name='P1', nbinsx=30, marker_color='#667eea'))
        fig.add_trace(go.Histogram(x=df['Pts_2'], name='P2', nbinsx=30, marker_color='#764ba2'))
        fig.update_layout(barmode='overlap')
        st.plotly_chart(fig, use_container_width=True)
    
    if 'Surface' in df.columns:
        st.subheader("Surface Analysis")
        surface_stats = df.groupby('Surface').agg({'Player_1_Won': ['sum', 'count']})
        surface_stats.columns = ['P1 Wins', 'Total']
        surface_stats['Win %'] = (surface_stats['P1 Wins'] / surface_stats['Total'] * 100).round(1)
        st.dataframe(surface_stats, use_container_width=True)
    
    if 'Round' in df.columns:
        st.subheader("Round Analysis")
        round_stats = df.groupby('Round').agg({'Player_1_Won': ['sum', 'count']})
        round_stats.columns = ['P1 Wins', 'Total']
        round_stats['Win %'] = (round_stats['P1 Wins'] / round_stats['Total'] * 100).round(1)
        st.dataframe(round_stats, use_container_width=True)

def main():
    st.sidebar.title("🎾 WTA Predictor")
    page = st.sidebar.radio("Page", ["🏠 Home", "📈 Training", "🔮 Predict", "📊 Analytics"])
    
    st.sidebar.title("📁 Upload")
    uploaded_file = st.sidebar.file_uploader("CSV", type=['csv'])
    
    if uploaded_file:
        model_data = load_and_train_model(uploaded_file)
        st.sidebar.success("✓ Model Ready!")
        
        if page == "🏠 Home":
            show_home(model_data)
        elif page == "📈 Training":
            show_training(model_data)
        elif page == "🔮 Predict":
            show_predictions(model_data)
        elif page == "📊 Analytics":
            show_analytics(model_data)
    else:
        st.title("🎾 WTA Predictor")
        st.info("👈 Upload CSV to start!")
        st.markdown("""
        ### Features
        ✓ Player Selection from Data • ✓ Surface Selection • ✓ Real-time Predictions • ✓ Analytics
        
        ### CSV Columns Required
        Tournament, Date, Surface, Court, Round, Player_1, Player_2, Winner, Rank_1, Rank_2, Pts_1, Pts_2, Odd_1, Odd_2, Score
        """)

if __name__ == "__main__":
    main()
