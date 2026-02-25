"""
Streamlit app for the Pressure-Based Betting Model
Interactive visualization and prediction tool for Premier League matches
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from betting_model_csv_adapted import PremierLeagueBettingModel
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Betting Model Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        color: #09ab3b;
        font-weight: bold;
    }
    .negative {
        color: #d62728;
        font-weight: bold;
    }
    .neutral {
        color: #ff7f0e;
        font-weight: bold;
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

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload E0.csv", type=['csv'])

if uploaded_file is not None:
    csv_path = uploaded_file
    
    # Load data
    with st.spinner("Loading and training model..."):
        model = PremierLeagueBettingModel()
        training_data, df = model.prepare_training_data(csv_path)
        model.train_model(training_data)
        stats = model.backtest_csv(csv_path)
        
        st.session_state.model = model
        st.session_state.stats = stats
        st.session_state.df = df
        
        st.sidebar.success("✅ Model trained successfully!")
else:
    st.sidebar.warning("⚠️ Upload a CSV file to start")
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
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Accuracy",
            f"{stats['accuracy']:.1f}%",
            delta="vs 50% (random)" if stats['accuracy'] > 50 else None,
            delta_color="off"
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
            delta="When betting" if stats['avg_edge'] > 0 else None
        )
    
    st.markdown("---")
    
    # Accuracy distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy gauge
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
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 55
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Signal distribution
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
    
    # Model statistics
    st.subheader("📋 Model Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Prediction Accuracy:**")
        st.write(f"- Total matches: {stats['total_matches']}")
        st.write(f"- Correct predictions: {results['correct'].sum()}")
        st.write(f"- Incorrect predictions: {(~results['correct']).sum()}")
        st.write(f"- Accuracy: {stats['accuracy']:.1f}%")
    
    with col2:
        st.write("**Signal Distribution:**")
        st.write(f"- OVER signals: {len(bettable[bettable['signal'] == 'OVER'])}")
        st.write(f"- UNDER signals: {len(bettable[bettable['signal'] == 'UNDER'])}")
        st.write(f"- PASS signals: {len(results[results['signal'] == 'PASS'])}")
        st.write(f"- Average edge: {stats['avg_edge']:.2f}%")

# ==================== TAB 2: PREDICTIONS ====================
with tab2:
    st.header("🎯 Match Predictions")
    
    # Selector
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_all = st.checkbox("Show all matches", value=False)
    
    with col2:
        signal_filter = st.selectbox("Filter by signal", 
            ["All", "OVER", "UNDER", "PASS"])
    
    with col3:
        sort_by = st.selectbox("Sort by", 
            ["Date", "Edge", "Confidence", "Prediction"])
    
    # Filter results
    display_results = results.copy()
    
    if not show_all:
        display_results = display_results[display_results['signal'] != 'PASS']
    
    if signal_filter != "All":
        display_results = display_results[display_results['signal'] == signal_filter]
    
    # Sort
    if sort_by == "Date":
        display_results = display_results.sort_values('date', ascending=False)
    elif sort_by == "Edge":
        display_results = display_results.sort_values('edge', ascending=False)
    elif sort_by == "Confidence":
        display_results = display_results.sort_values('edge', ascending=False)
    elif sort_by == "Prediction":
        display_results = display_results.sort_values('predicted_goals', ascending=False)
    
    st.markdown("---")
    
    # Display predictions
    for idx, row in display_results.head(10).iterrows():
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            st.write(f"**{row['match']}**")
            st.write(f"*{row['date'].strftime('%Y-%m-%d')}*")
        
        with col2:
            st.write(f"**Prediction:** {row['predicted_goals']:.2f}")
            st.write(f"**Actual:** {row['actual_goals']:.0f}")
        
        with col3:
            signal_color = "🟢" if row['signal'] == 'OVER' else "🔴" if row['signal'] == 'UNDER' else "⚪"
            st.write(f"{signal_color} **{row['signal']}**")
            st.write(f"Edge: {row['edge']:.2f}%")
        
        with col4:
            if row['correct']:
                st.write("✅ **CORRECT**")
            else:
                st.write("❌ **INCORRECT**")
            st.write(f"Odds: {row['over_odds'] if row['signal'] == 'OVER' else row['under_odds']:.2f}")
        
        st.markdown("---")

# ==================== TAB 3: ANALYSIS ====================
with tab3:
    st.header("📈 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    # Prediction vs Actual scatter
    with col1:
        st.subheader("Prediction vs Actual")
        
        fig = px.scatter(
            results,
            x='predicted_goals',
            y='actual_goals',
            color='correct',
            hover_data=['match', 'signal', 'edge'],
            color_discrete_map={True: '#09ab3b', False: '#d62728'},
            labels={'correct': 'Correct', 'predicted_goals': 'Predicted Goals', 'actual_goals': 'Actual Goals'}
        )
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, max(results['predicted_goals'].max(), results['actual_goals'].max())],
            y=[0, max(results['predicted_goals'].max(), results['actual_goals'].max())],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Accuracy by prediction range
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
    
    # Goal distribution
    st.subheader("Goal Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[
            go.Histogram(
                x=results['predicted_goals'],
                name='Predicted',
                opacity=0.7,
                nbinsx=15
            ),
            go.Histogram(
                x=results['actual_goals'],
                name='Actual',
                opacity=0.7,
                nbinsx=15
            )
        ])
        
        fig.update_layout(
            title="Distribution of Goals",
            xaxis_title="Goals",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Residuals
        results['residual'] = results['predicted_goals'] - results['actual_goals']
        
        fig = go.Figure(data=[
            go.Histogram(
                x=results['residual'],
                nbinsx=15,
                marker_color='#ff7f0e'
            )
        ])
        
        fig.update_layout(
            title="Prediction Residuals",
            xaxis_title="Predicted - Actual",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 4: TEAMS ====================
with tab4:
    st.header("🏆 Team Performance")
    
    # Calculate team statistics
    teams = {}
    
    for idx, row in results.iterrows():
        home, away = row['match'].split(' vs ')
        
        if home not in teams:
            teams[home] = {'correct': 0, 'total': 0, 'avg_prediction': [], 'avg_actual': []}
        if away not in teams:
            teams[away] = {'correct': 0, 'total': 0, 'avg_prediction': [], 'avg_actual': []}
        
        teams[home]['total'] += 1
        teams[away]['total'] += 1
        
        if row['correct']:
            teams[home]['correct'] += 1
            teams[away]['correct'] += 1
        
        teams[home]['avg_prediction'].append(row['predicted_goals'])
        teams[away]['avg_prediction'].append(row['predicted_goals'])
        teams[home]['avg_actual'].append(row['actual_goals'])
        teams[away]['avg_actual'].append(row['actual_goals'])
    
    # Calculate averages
    team_list = []
    for team, data in teams.items():
        if data['total'] > 0:
            team_list.append({
                'Team': team,
                'Accuracy': (data['correct'] / data['total']) * 100,
                'Matches': data['total'],
                'Correct': data['correct'],
                'Avg Prediction': np.mean(data['avg_prediction']),
                'Avg Actual': np.mean(data['avg_actual'])
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
        
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45,
            yaxis=dict(range=[0, 100])
        )
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
        
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed team table
    st.subheader("Detailed Team Statistics")
    st.dataframe(
        team_df.sort_values('Accuracy', ascending=False),
        use_container_width=True,
        hide_index=True
    )

# ==================== TAB 5: PROFIT ====================
with tab5:
    st.header("💰 Profit Analysis")
    
    # Calculate profit
    over_bets = bettable[bettable['signal'] == 'OVER']
    under_bets = bettable[bettable['signal'] == 'UNDER']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if len(over_bets) > 0:
            over_wins = over_bets[over_bets['correct']]['over_odds'].sum() - len(over_bets)
            over_losses = -over_bets[~over_bets['correct']].shape[0]
            over_profit = over_wins + over_losses
            over_accuracy = (over_bets['correct'].sum() / len(over_bets)) * 100
            
            st.metric(
                "OVER Bets P&L",
                f"{over_profit:+.2f} units",
                delta=f"{over_accuracy:.1f}% accuracy"
            )
    
    with col2:
        if len(under_bets) > 0:
            under_wins = under_bets[under_bets['correct']]['under_odds'].sum() - len(under_bets)
            under_losses = -under_bets[~under_bets['correct']].shape[0]
            under_profit = under_wins + under_losses
            under_accuracy = (under_bets['correct'].sum() / len(under_bets)) * 100
            
            st.metric(
                "UNDER Bets P&L",
                f"{under_profit:+.2f} units",
                delta=f"{under_accuracy:.1f}% accuracy"
            )
    
    with col3:
        total_profit = over_profit + under_profit if len(over_bets) > 0 and len(under_bets) > 0 else \
                       over_profit if len(over_bets) > 0 else under_profit
        roi = (total_profit / len(bettable)) * 100 if len(bettable) > 0 else 0
        
        profit_color = "positive" if total_profit > 0 else "negative"
        st.metric(
            "Total P&L",
            f"{total_profit:+.2f} units",
            delta=f"{roi:+.2f}% ROI"
        )
    
    st.markdown("---")
    
    # Profit simulation
    st.subheader("Profit Growth Simulation")
    
    # Calculate cumulative profit
    if len(bettable) > 0:
        bettable_sorted = bettable.sort_values('date').reset_index(drop=True)
        bettable_sorted['profit'] = 0.0
        
        for idx, row in bettable_sorted.iterrows():
            if row['signal'] == 'OVER':
                if row['correct']:
                    profit = row['over_odds'] - 1
                else:
                    profit = -1
            else:  # UNDER
                if row['correct']:
                    profit = row['under_odds'] - 1
                else:
                    profit = -1
            
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
        
        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Breakeven"
        )
        
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
        
        coefficients = st.session_state.model.coefficients
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
        
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Coefficient Interpretation:**")
        st.write("- Positive = increases goal predictions")
        st.write("- Negative = decreases goal predictions")
        st.write("- Larger magnitude = stronger effect")
    
    with col2:
        st.subheader("Feature Importance")
        
        coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=True)
        
        fig = px.barh(
            coef_df,
            x='Abs_Coefficient',
            y='Feature',
            color='Abs_Coefficient',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Absolute Coefficient"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model info
    st.subheader("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Data:**")
        st.write(f"- Total matches: {len(st.session_state.df)}")
        st.write(f"- Training set: {int(len(st.session_state.df) * 0.7)} matches")
        st.write(f"- Test set: {int(len(st.session_state.df) * 0.3)} matches")
        st.write(f"- Date range: {st.session_state.df['Date'].min().date()} to {st.session_state.df['Date'].max().date()}")
    
    with col2:
        st.write("**Model Configuration:**")
        st.write("- Model type: Linear Regression")
        st.write("- Features: 6 pressure metrics")
        st.write("- Target: Total goals")
        st.write("- Threshold: 2.5 goals (over/under)")
    
    # Feature descriptions
    st.subheader("Feature Descriptions")
    
    features_info = {
        'home_attack': 'Attacking pressure of home team (shots, accuracy, corners, fouls)',
        'home_defense': 'Defensive strength of home team (tackles, blocks, passes prevented)',
        'away_attack': 'Attacking pressure of away team',
        'away_defense': 'Defensive strength of away team',
        'total_attack': 'Combined attacking pressure (home + away)',
        'total_defense': 'Combined defensive strength (home + away)'
    }
    
    for feature, description in features_info.items():
        st.write(f"**{feature}:** {description}")
    
    # Data table
    st.subheader("Recent Matches Data")
    
    display_cols = ['match', 'date', 'predicted_goals', 'actual_goals', 'signal', 'edge', 'correct']
    st.dataframe(
        results[display_cols].sort_values('date', ascending=False).head(20),
        use_container_width=True,
        hide_index=True
    )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    ⚽ Premier League Betting Model Dashboard | Powered by Streamlit | Data from E0.csv
    </div>
""", unsafe_allow_html=True)
