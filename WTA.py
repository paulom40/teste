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
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return {
        'model': model,
        'scaler': scaler,
        'df': df,
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_test_scaled': X_test_scaled,
        'feature_names': feature_names,
        'importance_df': importance_df,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def show_home(model_data):
    st.header("Welcome to WTA Match Predictor")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", len(model_data['df']))
    with col2:
        st.metric("Test Accuracy", f"{model_data['test_accuracy']:.1%}")
    with col3:
        st.metric("Features", len(model_data['feature_names']))
    with col4:
        st.metric("Improvement", f"+{(model_data['test_accuracy'] - 0.5)*100:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 The 6-Factor Framework")
        st.markdown("""
        1. **Ranking Differential** - Head-to-head ranking
        2. **Points Differential** - Recent performance
        3. **Surface Performance** - Court-specific strengths
        4. **Tournament Context** - Round and match load
        5. **Physical Load** - Court type and fatigue
        6. **Momentum** - Betting odds as sentiment
        """)
    
    with col2:
        st.subheader("🎯 Key Metrics")
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score': [
                model_data['test_accuracy'],
                model_data['precision'],
                model_data['recall'],
                model_data['f1']
            ]
        })
        st.dataframe(metrics_df.style.format({'Score': '{:.1%}'}), use_container_width=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        df = model_data['df']
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
        fig = px.pie(values=[p1_wins, p2_wins], names=['Player 1 Wins', 'Player 2 Wins'], title="Win Distribution")
        st.plotly_chart(fig, use_container_width=True)

def show_training(model_data):
    st.header("📈 Model Training & Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Matches", len(model_data['X_train']))
    with col2:
        st.metric("Testing Matches", len(model_data['X_test']))
    with col3:
        st.metric("Test Accuracy", f"{model_data['test_accuracy']:.1%}")
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
        
        fig = go.Figure(data=[
            go.Bar(y=top_features['Feature'], x=top_features['Importance'], orientation='h', marker_color='#764ba2')
        ])
        fig.update_layout(title="Feature Importance", xaxis_title="Importance", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("All Features")
    st.dataframe(model_data['importance_df'].style.format({'Importance': '{:.4f}'}), use_container_width=True)

def show_predictions(model_data):
    st.header("🔮 Make Predictions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rank_1 = st.number_input("Player 1 Rank", min_value=1, max_value=1000, value=50)
    with col2:
        rank_2 = st.number_input("Player 2 Rank", min_value=1, max_value=1000, value=100)
    with col3:
        pts_1 = st.number_input("Player 1 Points", min_value=0, max_value=10000, value=1000)
    with col4:
        pts_2 = st.number_input("Player 2 Points", min_value=0, max_value=10000, value=800)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
    with col2:
        court = st.selectbox("Court", ["Indoor", "Outdoor"])
    with col3:
        round_type = st.selectbox("Round", ["1st Round", "2nd Round", "Quarterfinal", "Semifinal", "Final"])
    with col4:
        odds_1 = st.number_input("Player 1 Odds", min_value=1.0, max_value=100.0, value=1.8)
    
    odds_2 = st.number_input("Player 2 Odds", min_value=1.0, max_value=100.0, value=2.0)
    
    if st.button("🎯 Predict Match"):
        features = []
        
        features.append(rank_2 - rank_1)
        features.append(pts_1 - pts_2)
        features.append(rank_1)
        
        for surf in ["Hard", "Clay", "Grass"]:
            features.append(1.0 if surface == surf else 0.0)
        
        rounds = ["1st Round", "2nd Round", "Quarterfinal", "Semifinal", "Final"]
        for rnd in rounds:
            features.append(1.0 if round_type == rnd else 0.0)
        
        for crt in ["Indoor", "Outdoor"]:
            features.append(1.0 if court == crt else 0.0)
        
        features.append(odds_1 - odds_2)
        
        while len(features) < len(model_data['feature_names']):
            features.append(0.0)
        
        X_new = np.array(features[:len(model_data['feature_names'])]).reshape(1, -1)
        X_new_scaled = model_data['scaler'].transform(X_new)
        
        probability = model_data['model'].predict_proba(X_new_scaled)[0]
        
        p1_prob = probability[1]
        p2_prob = probability[0]
        confidence = abs(p1_prob - 0.5)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Prediction Result")
            
            if p1_prob > 0.5:
                st.success(f"🏆 Player 1 Favored")
                st.metric("Winning Probability", f"{p1_prob:.1%}")
            else:
                st.success(f"🏆 Player 2 Favored")
                st.metric("Winning Probability", f"{p2_prob:.1%}")
            
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col2:
            st.subheader("⚖️ Probability Breakdown")
            
            fig = go.Figure(data=[
                go.Bar(x=['Player 1', 'Player 2'], y=[p1_prob, p2_prob], marker_color=['#667eea', '#764ba2'])
            ])
            fig.update_layout(title="Win Probability", yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("📈 Confidence Interpretation")
        
        if confidence < 0.05:
            st.info("**50-55%**: Near toss-up, unpredictable match")
        elif confidence < 0.15:
            st.info("**55-65%**: Slight favorite, moderate confidence")
        elif confidence < 0.30:
            st.info("**65-80%**: Clear favorite, high confidence")
        else:
            st.info("**80%+**: Dominant expected outcome")

def show_analytics(model_data):
    st.header("📊 Data Analytics")
    
    df = model_data['df']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Ranking Distribution")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['Rank_1'], name='Player 1', nbinsx=30, marker_color='#667eea'))
        fig.add_trace(go.Histogram(x=df['Rank_2'], name='Player 2', nbinsx=30, marker_color='#764ba2'))
        fig.update_layout(title="Ranking Distribution", barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Points Distribution")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['Pts_1'], name='Player 1', nbinsx=30, marker_color='#667eea'))
        fig.add_trace(go.Histogram(x=df['Pts_2'], name='Player 2', nbinsx=30, marker_color='#764ba2'))
        fig.update_layout(title="Points Distribution", barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)
    
    if 'Surface' in df.columns:
        st.subheader("🏟️ Surface Analysis")
        
        surface_stats = df.groupby('Surface').agg({'Player_1_Won': ['sum', 'count']}).round(2)
        surface_stats.columns = ['Player 1 Wins', 'Total Matches']
        surface_stats['Win Rate %'] = (surface_stats['Player 1 Wins'] / surface_stats['Total Matches'] * 100).round(1)
        
        st.dataframe(surface_stats, use_container_width=True)
    
    if 'Round' in df.columns:
        st.subheader("🎯 Round Analysis")
        
        round_stats = df.groupby('Round').agg({'Player_1_Won': ['sum', 'count']}).round(2)
        round_stats.columns = ['Player 1 Wins', 'Total Matches']
        round_stats['Win Rate %'] = (round_stats['Player 1 Wins'] / round_stats['Total Matches'] * 100).round(1)
        
        st.dataframe(round_stats, use_container_width=True)

def main():
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio("Select Page", ["Home", "Model Training", "Predictions", "Analytics"])
    
    st.sidebar.title("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload WTA CSV", type=['csv'])
    
    if uploaded_file is not None:
        with st.spinner("Loading and training model..."):
            model_data = load_and_train_model(uploaded_file)
        
        if page == "Home":
            show_home(model_data)
        elif page == "Model Training":
            show_training(model_data)
        elif page == "Predictions":
            show_predictions(model_data)
        elif page == "Analytics":
            show_analytics(model_data)
    else:
        st.title("🎾 WTA Match Predictor")
        st.markdown("### Interactive Dashboard for Tennis Match Forecasting")
        st.info("👈 Please upload a CSV file in the sidebar to get started")

if __name__ == "__main__
