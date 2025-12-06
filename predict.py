import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Football Betting Model", layout="wide", page_icon="⚽")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">⚽ Professional Football Betting Model</h1>', unsafe_allow_html=True)

# Load and process data
@st.cache_data
def load_data():
    url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    return df

def engineer_features(df):
    """Create advanced features for the model"""
    features_df = df.copy()
    
    # Basic match stats
    features_df['TotalGoals'] = features_df['FTHG'] + features_df['FTAG']
    features_df['GoalDiff'] = features_df['FTHG'] - features_df['FTAG']
    features_df['HTGoalDiff'] = features_df['HTHG'] - features_df['HTAG']
    
    # Shot efficiency
    features_df['HomeShotAccuracy'] = features_df['HST'] / (features_df['HS'] + 0.1)
    features_df['AwayShotAccuracy'] = features_df['AST'] / (features_df['AS'] + 0.1)
    
    # Odds analysis
    features_df['HomeOddsImplied'] = 1 / features_df['B365H']
    features_df['DrawOddsImplied'] = 1 / features_df['B365D']
    features_df['AwayOddsImplied'] = 1 / features_df['B365A']
    features_df['BookmakerMargin'] = (features_df['HomeOddsImplied'] + 
                                       features_df['DrawOddsImplied'] + 
                                       features_df['AwayOddsImplied'])
    
    # Over/Under metrics
    features_df['Over2.5Implied'] = 1 / features_df['B365>2.5']
    
    # Form tracking (rolling averages)
    for team in features_df['HomeTeam'].unique():
        home_mask = features_df['HomeTeam'] == team
        away_mask = features_df['AwayTeam'] == team
        
        # Home form
        features_df.loc[home_mask, 'HomeForm_Goals'] = features_df.loc[home_mask, 'FTHG'].rolling(3, min_periods=1).mean()
        features_df.loc[home_mask, 'HomeForm_Conceded'] = features_df.loc[home_mask, 'FTAG'].rolling(3, min_periods=1).mean()
        
        # Away form
        features_df.loc[away_mask, 'AwayForm_Goals'] = features_df.loc[away_mask, 'FTAG'].rolling(3, min_periods=1).mean()
        features_df.loc[away_mask, 'AwayForm_Conceded'] = features_df.loc[away_mask, 'FTHG'].rolling(3, min_periods=1).mean()
    
    # Fill NaN values
    features_df = features_df.fillna(method='bfill').fillna(0)
    
    return features_df

def create_model_data(df):
    """Prepare data for machine learning"""
    feature_cols = [
        'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY',
        'HomeShotAccuracy', 'AwayShotAccuracy',
        'HomeOddsImplied', 'DrawOddsImplied', 'AwayOddsImplied',
        'BookmakerMargin', 'Over2.5Implied',
        'HomeForm_Goals', 'HomeForm_Conceded',
        'AwayForm_Goals', 'AwayForm_Conceded'
    ]
    
    X = df[feature_cols].fillna(0)
    y = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    
    return X, y, feature_cols

def calculate_value_bets(predictions, odds, threshold=0.05):
    """Identify value betting opportunities"""
    value_bets = []
    outcomes = ['Home Win', 'Draw', 'Away Win']
    
    for i, (pred_prob, odd) in enumerate(zip(predictions, odds)):
        implied_prob = 1 / odd
        if pred_prob > implied_prob + threshold:
            edge = ((pred_prob * odd) - 1) * 100
            value_bets.append({
                'outcome': outcomes[i],
                'model_prob': pred_prob,
                'implied_prob': implied_prob,
                'odds': odd,
                'edge': edge
            })
    
    return value_bets

# Load data
with st.spinner('Loading data...'):
    df = load_data()
    df = engineer_features(df)

# Sidebar
st.sidebar.title("🎯 Model Settings")
model_type = st.sidebar.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "Ensemble"])
value_threshold = st.sidebar.slider("Value Bet Threshold (%)", 1, 20, 5) / 100

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🤖 Predictions", "💰 Value Finder", "📈 Analysis"])

with tab1:
    st.header("Season Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Matches", len(df))
    with col2:
        home_wins = (df['FTR'] == 'H').sum()
        st.metric("Home Wins", f"{home_wins} ({home_wins/len(df)*100:.1f}%)")
    with col3:
        draws = (df['FTR'] == 'D').sum()
        st.metric("Draws", f"{draws} ({draws/len(df)*100:.1f}%)")
    with col4:
        away_wins = (df['FTR'] == 'A').sum()
        st.metric("Away Wins", f"{away_wins} ({away_wins/len(df)*100:.1f}%)")
    
    # Results distribution
    col1, col2 = st.columns(2)
    
    with col1:
        result_counts = df['FTR'].value_counts()
        fig = px.pie(values=result_counts.values, names=['Home', 'Draw', 'Away'],
                     title="Match Results Distribution", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        avg_goals = df.groupby('HomeTeam').agg({
            'FTHG': 'mean',
            'FTAG': 'mean'
        }).reset_index()
        avg_goals['Total'] = avg_goals['FTHG'] + avg_goals['FTAG']
        fig = px.bar(avg_goals.nlargest(10, 'Total'), x='HomeTeam', y='Total',
                     title="Top 10 Teams by Average Goals", color='Total')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Match Predictions")
    
    # Train model
    X, y, feature_cols = create_model_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    else:
        model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
    
    model.fit(X_train_scaled, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    
    st.success(f"Model Accuracy: {accuracy*100:.2f}%")
    
    # Make predictions for recent matches
    recent_matches = df.tail(10).copy()
    X_recent = recent_matches[feature_cols].fillna(0)
    X_recent_scaled = scaler.transform(X_recent)
    predictions = model.predict_proba(X_recent_scaled)
    
    st.subheader("Recent Matches Analysis")
    
    for idx, (i, match) in enumerate(recent_matches.iterrows()):
        with st.expander(f"{match['HomeTeam']} vs {match['AwayTeam']} - {match['Date'].strftime('%d/%m/%Y')}"):
            col1, col2, col3 = st.columns(3)
            
            probs = predictions[idx]
            result_map = {0: 'H', 1: 'D', 2: 'A'}
            predicted = result_map[np.argmax(probs)]
            actual = match['FTR']
            
            with col1:
                st.write("**Actual Result**")
                st.write(f"Score: {match['FTHG']} - {match['FTAG']}")
                st.write(f"Result: {actual}")
            
            with col2:
                st.write("**Model Prediction**")
                st.write(f"Home: {probs[0]*100:.1f}%")
                st.write(f"Draw: {probs[1]*100:.1f}%")
                st.write(f"Away: {probs[2]*100:.1f}%")
                st.write(f"Predicted: {predicted}")
            
            with col3:
                correct = "✅" if predicted == actual else "❌"
                st.write("**Match Stats**")
                st.write(f"Shots: {match['HS']} - {match['AS']}")
                st.write(f"On Target: {match['HST']} - {match['AST']}")
                st.write(f"Prediction: {correct}")

with tab3:
    st.header("Value Betting Finder")
    
    st.info("Value bets are identified when the model's probability exceeds the bookmaker's implied probability by the threshold percentage.")
    
    # Analyze all matches for value
    X_all = df[feature_cols].fillna(0)
    X_all_scaled = scaler.transform(X_all)
    all_predictions = model.predict_proba(X_all_scaled)
    
    value_opportunities = []
    
    for idx, (i, match) in enumerate(df.iterrows()):
        probs = all_predictions[idx]
        odds = [match['B365H'], match['B365D'], match['B365A']]
        
        value_bets = calculate_value_bets(probs, odds, value_threshold)
        
        if value_bets:
            for vb in value_bets:
                value_opportunities.append({
                    'Date': match['Date'],
                    'Match': f"{match['HomeTeam']} vs {match['AwayTeam']}",
                    'Outcome': vb['outcome'],
                    'Model Prob': f"{vb['model_prob']*100:.1f}%",
                    'Implied Prob': f"{vb['implied_prob']*100:.1f}%",
                    'Odds': f"{vb['odds']:.2f}",
                    'Edge': f"{vb['edge']:.2f}%",
                    'Result': match['FTR']
                })
    
    if value_opportunities:
        value_df = pd.DataFrame(value_opportunities)
        st.write(f"**Found {len(value_df)} value betting opportunities**")
        st.dataframe(value_df, use_container_width=True)
        
        # Calculate ROI
        total_bets = len(value_df)
        # Simulate betting results
        st.subheader("Simulated Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Value Bets", total_bets)
        with col2:
            st.metric("Avg Edge", f"{value_df['Edge'].str.rstrip('%').astype(float).mean():.2f}%")
        with col3:
            avg_odds = value_df['Odds'].str.extract(r'(\d+\.\d+)')[0].astype(float).mean()
            st.metric("Avg Odds", f"{avg_odds:.2f}")
    else:
        st.warning("No value opportunities found with current threshold.")

with tab4:
    st.header("Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df.head(10), x='Importance', y='Feature',
                        title="Top 10 Most Important Features", orientation='h')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Goals distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['FTHG'], name='Home Goals', opacity=0.7))
        fig.add_trace(go.Histogram(x=df['FTAG'], name='Away Goals', opacity=0.7))
        fig.update_layout(title="Goals Distribution", barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)
    
    # Team performance matrix
    st.subheader("Team Performance Matrix")
    team_stats = df.groupby('HomeTeam').agg({
        'FTHG': 'mean',
        'FTAG': 'mean',
        'HS': 'mean',
        'HST': 'mean'
    }).round(2)
    team_stats.columns = ['Avg Goals Scored', 'Avg Goals Conceded', 'Avg Shots', 'Avg Shots on Target']
    st.dataframe(team_stats, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>⚽ Professional Football Betting Model | Data from football-data.co.uk</p>
    <p>⚠️ For educational purposes only. Always gamble responsibly.</p>
</div>
""", unsafe_allow_html=True)
