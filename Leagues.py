"""
Standalone Streamlit Betting Model Dashboard
All-in-one file (no external dependencies needed)
Works with Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ==================== MODEL CLASS ====================
class PremierLeagueBettingModel:
    """Betting model with all functionality inline"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.coefficients = {}
        
    def calculate_attacking_pressure(self, shots, shots_on_target, corners, fouls):
        shot_quality = shots_on_target / max(shots, 1)
        attacking_pressure = (
            (shots * 0.35) +
            (shots_on_target * 0.40) +
            (corners * 0.15) +
            (fouls * 0.10)
        )
        return attacking_pressure
    
    def calculate_defensive_pressure(self, shots_against, shots_on_target_against, 
                                    corners_against, yellow_cards):
        opponent_threat = (shots_against * 0.35) + (shots_on_target_against * 0.40)
        defensive_actions = (corners_against * 0.15) + (yellow_cards * 0.10)
        defensive_pressure = defensive_actions - (opponent_threat * 0.5)
        return defensive_pressure
    
    def create_match_features_from_row(self, row):
        home_attack = self.calculate_attacking_pressure(
            row['HS'], row['HST'], row['HC'], row['AF']
        )
        home_defense = self.calculate_defensive_pressure(
            row['AS'], row['AST'], row['AC'], row['HY']
        )
        away_attack = self.calculate_attacking_pressure(
            row['AS'], row['AST'], row['AC'], row['HF']
        )
        away_defense = self.calculate_defensive_pressure(
            row['HS'], row['HST'], row['HC'], row['AY']
        )
        
        return {
            'home_attack': home_attack,
            'home_defense': home_defense,
            'away_attack': away_attack,
            'away_defense': away_defense,
            'total_attack': home_attack + away_attack,
            'total_defense': home_defense + away_defense,
            'total_goals': row['FTHG'] + row['FTAG']
        }
    
    def prepare_training_data(self, csv_data):
        df = csv_data.copy()
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
        
        training_data = []
        for idx, row in df.iterrows():
            try:
                features = self.create_match_features_from_row(row)
                training_data.append(features)
            except:
                continue
        
        return pd.DataFrame(training_data), df
    
    def train_model(self, training_df):
        X = training_df[['home_attack', 'home_defense', 'away_attack', 'away_defense', 
                        'total_attack', 'total_defense']]
        y = training_df['total_goals']
        
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        X_scaled = self.scaler.fit_transform(X)
        self.model = LinearRegression()
        self.model.fit(X_scaled, y)
        
        self.coefficients = {
            'home_attack': self.model.coef_[0],
            'home_defense': self.model.coef_[1],
            'away_attack': self.model.coef_[2],
            'away_defense': self.model.coef_[3],
            'total_attack': self.model.coef_[4],
            'total_defense': self.model.coef_[5],
            'intercept': self.model.intercept_
        }
        
        return self.coefficients
    
    def predict_total_goals(self, match_features):
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        X = np.array([[
            match_features['home_attack'],
            match_features['home_defense'],
            match_features['away_attack'],
            match_features['away_defense'],
            match_features['total_attack'],
            match_features['total_defense']
        ]])
        
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        return max(prediction, 0)
    
    def calculate_implied_probability(self, odds):
        return 1 / odds
    
    def generate_betting_signal(self, predicted_goals, over_odds, under_odds, 
                              threshold=0.55, min_edge=0.02):
        goal_threshold = 2.5
        std_dev = 1.0
        z_score = (goal_threshold - predicted_goals) / std_dev
        prob_over = 1 - norm.cdf(z_score)
        
        market_prob_over = self.calculate_implied_probability(over_odds)
        market_prob_under = self.calculate_implied_probability(under_odds)
        
        edge_over = prob_over - market_prob_over
        edge_under = (1 - prob_over) - market_prob_under
        
        signal = 'PASS'
        edge = 0
        confidence = 0
        
        if edge_over > min_edge and prob_over > threshold:
            signal = 'OVER'
            edge = edge_over
            confidence = prob_over
        elif edge_under > min_edge and (1 - prob_over) > threshold:
            signal = 'UNDER'
            edge = edge_under
            confidence = 1 - prob_over
        
        return {
            'signal': signal,
            'edge': edge * 100,
            'confidence': confidence * 100,
            'predicted_goals': predicted_goals,
            'prob_over': prob_over * 100,
            'prob_under': (1 - prob_over) * 100
        }
    
    def backtest_csv(self, csv_data):
        training_data, df = self.prepare_training_data(csv_data)
        
        split_idx = int(len(training_data) * 0.7)
        train_data = training_data.iloc[:split_idx].copy()
        test_data = training_data.iloc[split_idx:].copy()
        test_matches = df.iloc[split_idx:].reset_index(drop=True)
        
        self.train_model(train_data)
        
        results = []
        for idx, row in test_matches.iterrows():
            actual_goals = row['FTHG'] + row['FTAG']
            
            try:
                features = self.create_match_features_from_row(row)
                over_odds = row['B365>2.5'] if not pd.isna(row['B365>2.5']) else 1.90
                under_odds = row['B365<2.5'] if not pd.isna(row['B365<2.5']) else 1.95
                
                prediction = self.predict_total_goals(features)
                signal = self.generate_betting_signal(prediction, over_odds, under_odds)
                
                predicted_over = prediction > 2.5
                actual_over = actual_goals > 2.5
                correct = predicted_over == actual_over
                
                results.append({
                    'match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                    'date': row['Date'],
                    'predicted_goals': prediction,
                    'actual_goals': actual_goals,
                    'signal': signal['signal'],
                    'edge': signal['edge'],
                    'correct': correct,
                    'over_odds': over_odds,
                    'under_odds': under_odds
                })
            except:
                continue
        
        results_df = pd.DataFrame(results)
        bettable = results_df[results_df['signal'] != 'PASS']
        
        stats = {
            'total_matches': len(results_df),
            'bettable_matches': len(bettable),
            'bet_rate': len(bettable) / len(results_df) * 100 if len(results_df) > 0 else 0,
            'avg_prediction': results_df['predicted_goals'].mean(),
            'avg_actual': results_df['actual_goals'].mean(),
            'avg_edge': bettable['edge'].mean() if len(bettable) > 0 else 0,
            'accuracy': (results_df['correct'].sum() / len(results_df) * 100) if len(results_df) > 0 else 0,
            'bet_accuracy': (bettable['correct'].sum() / len(bettable) * 100) if len(bettable) > 0 else 0,
            'over_signals': len(bettable[bettable['signal'] == 'OVER']),
            'under_signals': len(bettable[bettable['signal'] == 'UNDER']),
            'results_df': results_df,
            'bettable_df': bettable
        }
        
        return stats


# ==================== STREAMLIT APP ====================

st.set_page_config(
    page_title="Betting Model Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload E0.csv", type=['csv'])

if uploaded_file is not None:
    csv_data = pd.read_csv(uploaded_file)
    
    with st.spinner("Loading and training model..."):
        model = PremierLeagueBettingModel()
        training_data, df = model.prepare_training_data(csv_data)
        stats = model.backtest_csv(csv_data)
        
        st.session_state.model = model
        st.session_state.stats = stats
        st.session_state.df = df
        
        st.sidebar.success("✅ Model trained successfully!")
else:
    st.sidebar.warning("⚠️ Upload a CSV file to start")
    st.title("⚽ Premier League Betting Model Dashboard")
    st.markdown("*Upload your E0.csv in the sidebar to begin*")
    st.stop()

# Main title
st.title("⚽ Premier League Betting Model Dashboard")
st.markdown("*Pressure-based predictions for over/under 2.5 goals*")
st.markdown("---")

# Get data from session
model = st.session_state.model
stats = st.session_state.stats
df = st.session_state.df
results = stats['results_df']
bettable = stats['bettable_df']

# Tab navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🎯 Predictions",
    "📈 Analysis",
    "🏆 Teams",
    "💰 Profit",
    "⚙️ Details"
])

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    st.header("Model Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Accuracy",
            f"{stats['accuracy']:.1f}%",
            delta="vs 50% (random)" if stats['accuracy'] > 50 else None
        )
    
    with col2:
        st.metric(
            "Bettable Matches",
            f"{stats['bettable_matches']}/{stats['total_matches']}",
            delta=f"{stats['bet_rate']:.1f}%"
        )
    
    with col3:
        st.metric(
            "Bet Accuracy",
            f"{stats['bet_accuracy']:.1f}%",
            delta=f"+{stats['bet_accuracy'] - 50:.1f}% vs random"
        )
    
    with col4:
        st.metric(
            "Avg Edge",
            f"{stats['avg_edge']:.2f}%",
            delta="When betting"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=stats['accuracy'],
            title={'text': "Accuracy"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#09ab3b" if stats['accuracy'] > 55 else "#d62728"},
                'steps': [
                    {'range': [0, 50], 'color': "#f0f0f0"},
                    {'range': [50, 100], 'color': "#e8f5e9"}
                ]
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        signal_counts = results['signal'].value_counts()
        fig = go.Figure(data=[
            go.Bar(
                x=['OVER', 'UNDER', 'PASS'],
                y=[
                    signal_counts.get('OVER', 0),
                    signal_counts.get('UNDER', 0),
                    signal_counts.get('PASS', 0)
                ],
                marker_color=['#09ab3b', '#d62728', '#ff7f0e']
            )
        ])
        fig.update_layout(
            title="Signal Distribution",
            xaxis_title="Signal Type",
            yaxis_title="Count",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 2: PREDICTIONS ====================
with tab2:
    st.header("🎯 Match Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_all = st.checkbox("Show all matches", value=False)
    
    with col2:
        signal_filter = st.selectbox("Filter by signal", 
            ["All", "OVER", "UNDER", "PASS"])
    
    with col3:
        sort_by = st.selectbox("Sort by", 
            ["Date", "Edge", "Confidence", "Prediction"])
    
    display_results = results.copy()
    
    if not show_all:
        display_results = display_results[display_results['signal'] != 'PASS']
    
    if signal_filter != "All":
        display_results = display_results[display_results['signal'] == signal_filter]
    
    if sort_by == "Date":
        display_results = display_results.sort_values('date', ascending=False)
    elif sort_by == "Edge":
        display_results = display_results.sort_values('edge', ascending=False)
    elif sort_by == "Prediction":
        display_results = display_results.sort_values('predicted_goals', ascending=False)
    
    st.markdown("---")
    
    for idx, row in display_results.head(10).iterrows():
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            st.write(f"**{row['match']}**")
            st.write(f"*{row['date'].strftime('%Y-%m-%d')}*")
        
        with col2:
            st.write(f"**Pred:** {row['predicted_goals']:.2f}")
            st.write(f"**Actual:** {row['actual_goals']:.0f}")
        
        with col3:
            signal_color = "🟢" if row['signal'] == 'OVER' else "🔴" if row['signal'] == 'UNDER' else "⚪"
            st.write(f"{signal_color} **{row['signal']}**")
            st.write(f"Edge: {row['edge']:.2f}%")
        
        with col4:
            if row['correct']:
                st.write("✅ **CORRECT**")
            else:
                st.write("❌ **WRONG**")
            st.write(f"Odds: {row['over_odds'] if row['signal'] == 'OVER' else row['under_odds']:.2f}")
        
        st.markdown("---")

# ==================== TAB 3: ANALYSIS ====================
with tab3:
    st.header("📈 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction vs Actual")
        
        fig = px.scatter(
            results,
            x='predicted_goals',
            y='actual_goals',
            color='correct',
            hover_data=['match', 'signal', 'edge'],
            color_discrete_map={True: '#09ab3b', False: '#d62728'},
            labels={'correct': 'Correct'}
        )
        
        fig.add_trace(go.Scatter(
            x=[0, max(results['predicted_goals'].max(), results['actual_goals'].max())],
            y=[0, max(results['predicted_goals'].max(), results['actual_goals'].max())],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Accuracy by Prediction Range")
        
        ranges = [
            (0, 1.5, "Low (0-1.5)"),
            (1.5, 2.5, "Medium (1.5-2.5)"),
            (2.5, 3.5, "High (2.5-3.5)"),
            (3.5, 10, "Very High (3.5+)")
        ]
        
        range_data = []
        for min_pred, max_pred, label in ranges:
            in_range = results[(results['predicted_goals'] >= min_pred) & 
                              (results['predicted_goals'] < max_pred)]
            
            if len(in_range) > 0:
                correct = in_range['correct'].sum()
                accuracy = (correct / len(in_range)) * 100
                range_data.append({
                    'Range': label,
                    'Accuracy': accuracy,
                    'Count': len(in_range)
                })
        
        range_df = pd.DataFrame(range_data)
        
        fig = go.Figure(data=[
            go.Bar(
                x=range_df['Range'],
                y=range_df['Accuracy'],
                text=range_df['Accuracy'].round(1),
                textposition='auto',
                marker_color=['#d62728' if x < 55 else '#09ab3b' for x in range_df['Accuracy']],
                hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.1f}%<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Accuracy by Prediction Range",
            xaxis_title="Prediction Range",
            yaxis_title="Accuracy (%)",
            height=400,
            showlegend=False,
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 4: TEAMS ====================
with tab4:
    st.header("🏆 Team Performance")
    
    teams = {}
    
    for idx, row in results.iterrows():
        home, away = row['match'].split(' vs ')
        
        if home not in teams:
            teams[home] = {'correct': 0, 'total': 0}
        if away not in teams:
            teams[away] = {'correct': 0, 'total': 0}
        
        teams[home]['total'] += 1
        teams[away]['total'] += 1
        
        if row['correct']:
            teams[home]['correct'] += 1
            teams[away]['correct'] += 1
    
    team_list = []
    for team, data in teams.items():
        if data['total'] > 0:
            team_list.append({
                'Team': team,
                'Accuracy': (data['correct'] / data['total']) * 100,
                'Matches': data['total'],
                'Correct': data['correct']
            })
    
    team_df = pd.DataFrame(team_list)
    team_df = team_df.sort_values('Accuracy', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Team Accuracy")
        
        fig = px.bar(
            team_df.head(10),
            x='Team',
            y='Accuracy',
            color='Accuracy',
            color_continuous_scale=['#d62728', '#ff7f0e', '#09ab3b'],
            labels={'Accuracy': 'Accuracy (%)'},
            range_color=[0, 100]
        )
        
        fig.update_layout(height=400, xaxis_tickangle=-45, yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Match Count by Team")
        
        fig = px.bar(
            team_df.sort_values('Matches', ascending=False).head(10),
            x='Team',
            y='Matches',
            color='Matches',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Team Statistics")
    st.dataframe(team_df.sort_values('Accuracy', ascending=False), use_container_width=True, hide_index=True)

# ==================== TAB 5: PROFIT ====================
with tab5:
    st.header("💰 Profit Analysis")
    
    over_bets = bettable[bettable['signal'] == 'OVER']
    under_bets = bettable[bettable['signal'] == 'UNDER']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if len(over_bets) > 0:
            over_wins = over_bets[over_bets['correct']]['over_odds'].sum() - len(over_bets)
            over_losses = -over_bets[~over_bets['correct']].shape[0]
            over_profit = over_wins + over_losses
            over_accuracy = (over_bets['correct'].sum() / len(over_bets)) * 100
            
            st.metric("OVER Bets P&L", f"{over_profit:+.2f} units", delta=f"{over_accuracy:.1f}% accuracy")
    
    with col2:
        if len(under_bets) > 0:
            under_wins = under_bets[under_bets['correct']]['under_odds'].sum() - len(under_bets)
            under_losses = -under_bets[~under_bets['correct']].shape[0]
            under_profit = under_wins + under_losses
            under_accuracy = (under_bets['correct'].sum() / len(under_bets)) * 100
            
            st.metric("UNDER Bets P&L", f"{under_profit:+.2f} units", delta=f"{under_accuracy:.1f}% accuracy")
    
    with col3:
        total_profit = over_profit + under_profit if len(over_bets) > 0 and len(under_bets) > 0 else \
                       over_profit if len(over_bets) > 0 else under_profit
        roi = (total_profit / len(bettable)) * 100 if len(bettable) > 0 else 0
        
        st.metric("Total P&L", f"{total_profit:+.2f} units", delta=f"{roi:+.2f}% ROI")
    
    st.markdown("---")
    st.subheader("Profit Growth Simulation")
    
    if len(bettable) > 0:
        bettable_sorted = bettable.sort_values('date').reset_index(drop=True)
        bettable_sorted['profit'] = 0.0
        
        for idx, row in bettable_sorted.iterrows():
            if row['signal'] == 'OVER':
                profit = (row['over_odds'] - 1) if row['correct'] else -1
            else:
                profit = (row['under_odds'] - 1) if row['correct'] else -1
            
            bettable_sorted.at[idx, 'profit'] = profit
        
        bettable_sorted['cumulative_profit'] = bettable_sorted['profit'].cumsum()
        bettable_sorted['cumulative_stake'] = range(1, len(bettable_sorted) + 1)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=bettable_sorted['cumulative_stake'],
            y=bettable_sorted['cumulative_profit'],
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color='#09ab3b', width=2),
            fill='tozeroy',
            fillcolor='rgba(9, 171, 59, 0.1)'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Breakeven")
        
        fig.update_layout(
            title="Cumulative Profit Growth",
            xaxis_title="Bets Placed",
            yaxis_title="Cumulative Profit (units)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write(f"**Profit Summary (1 unit per bet):**")
        st.write(f"- Starting bankroll: 100 units")
        st.write(f"- Total bets: {len(bettable_sorted)}")
        st.write(f"- Total profit: {bettable_sorted['profit'].sum():+.2f} units")
        st.write(f"- Final bankroll: {100 + bettable_sorted['profit'].sum():.2f} units")
        st.write(f"- ROI: {(bettable_sorted['profit'].sum() / len(bettable_sorted)) * 100:+.2f}%")

# ==================== TAB 6: DETAILS ====================
with tab6:
    st.header("⚙️ Model Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Coefficients")
        
        coefficients = model.coefficients
        coef_df = pd.DataFrame([
            {'Feature': k, 'Coefficient': v}
            for k, v in coefficients.items()
            if k != 'intercept'
        ]).sort_values('Coefficient', ascending=False)
        
        fig = px.bar(
            coef_df,
            x='Feature',
            y='Coefficient',
            color='Coefficient',
            color_continuous_scale=['#d62728', '#ffffff', '#09ab3b'],
            color_continuous_midpoint=0
        )
        
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Information")
        
        st.write("**Training Data:**")
        st.write(f"- Total matches: {len(df)}")
        st.write(f"- Training set: {int(len(df) * 0.7)} matches")
        st.write(f"- Test set: {int(len(df) * 0.3)} matches")
        st.write(f"- Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        st.write("**Model Configuration:**")
        st.write("- Model type: Linear Regression")
        st.write("- Features: 6 pressure metrics")
        st.write("- Target: Total goals")
        st.write("- Threshold: 2.5 goals (over/under)")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    ⚽ Premier League Betting Model Dashboard | Powered by Streamlit
    </div>
""", unsafe_allow_html=True)
