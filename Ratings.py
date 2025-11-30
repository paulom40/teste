import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(page_title="Football Ratings System", page_icon="⚽", layout="wide")

# Title and description
st.title("⚽ Football Match Rating & Prediction System")
st.markdown("""
This app implements the **Goal Superiority Rating System** described in the Football-Data article.
It calculates match ratings based on recent form and predicts match outcomes with fair odds.
""")

# Sidebar for input
st.sidebar.header("Match Configuration")

# Team input
st.sidebar.subheader("Teams")
home_team = st.sidebar.text_input("Home Team", "Tottenham")
away_team = st.sidebar.text_input("Away Team", "Leeds")

# Recent form input
st.sidebar.subheader("Recent Form (Last 6 Matches)")

col1, col2 = st.sidebar.columns(2)
with col1:
    st.markdown("**Home Team**")
    home_scored = st.number_input(f"{home_team} Goals Scored", min_value=0, max_value=50, value=6)
    home_conceded = st.number_input(f"{home_team} Goals Conceded", min_value=0, max_value=50, value=9)
    
with col2:
    st.markdown("**Away Team**")
    away_scored = st.number_input(f"{away_team} Goals Scored", min_value=0, max_value=50, value=8)
    away_conceded = st.number_input(f"{away_team} Goals Conceded", min_value=0, max_value=50, value=11)

# Bookmaker odds input (optional)
st.sidebar.subheader("Bookmaker Odds (Optional)")
st.sidebar.markdown("*Enter to identify value bets*")
bm_home_odds = st.sidebar.number_input("Home Win Odds", min_value=1.01, max_value=100.0, value=2.20, step=0.01)
bm_draw_odds = st.sidebar.number_input("Draw Odds", min_value=1.01, max_value=100.0, value=3.40, step=0.01)
bm_away_odds = st.sidebar.number_input("Away Win Odds", min_value=1.01, max_value=100.0, value=4.50, step=0.01)

# Calculate ratings
home_rating = home_scored - home_conceded
away_rating = away_scored - away_conceded
match_rating = home_rating - away_rating

# Best-fit equations from the article
def calc_home_win_prob(rating):
    """y = 1.56x + 46.47"""
    return 1.56 * rating + 46.47

def calc_draw_prob(rating):
    """y = -0.03x² - 0.29x + 29.48"""
    return -0.03 * (rating ** 2) - 0.29 * rating + 29.48

def calc_away_win_prob(rating):
    """y = 0.03x² - 1.27x + 23.65"""
    return 0.03 * (rating ** 2) - 1.27 * rating + 23.65

# Calculate probabilities
home_prob = calc_home_win_prob(match_rating)
draw_prob = calc_draw_prob(match_rating)
away_prob = calc_away_win_prob(match_rating)

# Normalize probabilities to sum to 100%
total_prob = home_prob + draw_prob + away_prob
home_prob_norm = (home_prob / total_prob) * 100
draw_prob_norm = (draw_prob / total_prob) * 100
away_prob_norm = (away_prob / total_prob) * 100

# Calculate fair odds
fair_home_odds = 100 / home_prob_norm
fair_draw_odds = 100 / draw_prob_norm
fair_away_odds = 100 / away_prob_norm

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Match Prediction", "📈 Rating Analysis", "💰 Value Bets", "📚 About"])

with tab1:
    st.header("Match Prediction")
    
    # Display team ratings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label=f"{home_team} Rating",
            value=f"{home_rating:+d}",
            delta=f"{home_scored} scored, {home_conceded} conceded"
        )
    
    with col2:
        st.metric(
            label="Match Rating",
            value=f"{match_rating:+d}",
            delta="Home - Away"
        )
    
    with col3:
        st.metric(
            label=f"{away_team} Rating",
            value=f"{away_rating:+d}",
            delta=f"{away_scored} scored, {away_conceded} conceded"
        )
    
    st.divider()
    
    # Probabilities and fair odds
    st.subheader("Result Probabilities & Fair Odds")
    
    results_df = pd.DataFrame({
        'Result': [f'{home_team} Win', 'Draw', f'{away_team} Win'],
        'Probability (%)': [home_prob_norm, draw_prob_norm, away_prob_norm],
        'Fair Odds': [fair_home_odds, fair_draw_odds, fair_away_odds]
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Probability bar chart
        fig = px.bar(
            results_df,
            x='Result',
            y='Probability (%)',
            color='Result',
            text='Probability (%)',
            color_discrete_map={
                f'{home_team} Win': '#2E86AB',
                'Draw': '#A23B72',
                f'{away_team} Win': '#F18F01'
            }
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(
            results_df.style.format({
                'Probability (%)': '{:.2f}%',
                'Fair Odds': '{:.2f}'
            }),
            hide_index=True,
            use_container_width=True
        )
        
        # Most likely result
        most_likely = results_df.loc[results_df['Probability (%)'].idxmax(), 'Result']
        most_likely_prob = results_df['Probability (%)'].max()
        st.success(f"**Most Likely:** {most_likely} ({most_likely_prob:.1f}%)")

with tab2:
    st.header("Rating Analysis")
    
    # Generate rating distribution data
    ratings = list(range(-26, 28))
    home_probs = [calc_home_win_prob(r) for r in ratings]
    draw_probs = [calc_draw_prob(r) for r in ratings]
    away_probs = [calc_away_win_prob(r) for r in ratings]
    
    # Create probability curves
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ratings, y=home_probs,
        mode='lines',
        name='Home Win',
        line=dict(color='#2E86AB', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=ratings, y=draw_probs,
        mode='lines',
        name='Draw',
        line=dict(color='#A23B72', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=ratings, y=away_probs,
        mode='lines',
        name='Away Win',
        line=dict(color='#F18F01', width=3)
    ))
    
    # Add vertical line for current match rating
    fig.add_vline(
        x=match_rating,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Current Match: {match_rating}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Result Probabilities by Match Rating",
        xaxis_title="Match Rating",
        yaxis_title="Probability (%)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Interpretation:**
    - **Positive ratings**: Home team is stronger (more likely to win)
    - **Negative ratings**: Away team is stronger
    - **Rating near 0**: Teams are evenly matched
    - The curves show how probabilities change with team strength differences
    """)
    
    # Rating reliability indicator
    st.subheader("Prediction Confidence")
    
    if -5 <= match_rating <= 5:
        confidence = "High"
        color = "green"
        explanation = "This rating falls in the most reliable range (-5 to +5) with ample historical data."
    elif -10 <= match_rating <= 10:
        confidence = "Medium"
        color = "orange"
        explanation = "This rating has moderate reliability. Predictions are still useful but with increased uncertainty."
    else:
        confidence = "Low"
        color = "red"
        explanation = "Extreme ratings have limited historical data. Use predictions with caution."
    
    st.markdown(f"**Confidence Level:** :{color}[{confidence}]")
    st.caption(explanation)

with tab3:
    st.header("Value Bet Analysis")
    
    st.markdown("""
    A **value bet** occurs when the bookmaker's odds are higher than the calculated fair odds,
    meaning the bookmaker has underestimated the probability of that outcome.
    """)
    
    # Calculate value
    value_df = pd.DataFrame({
        'Result': [f'{home_team} Win', 'Draw', f'{away_team} Win'],
        'Fair Odds': [fair_home_odds, fair_draw_odds, fair_away_odds],
        'Bookmaker Odds': [bm_home_odds, bm_draw_odds, bm_away_odds],
        'Value (%)': [
            ((bm_home_odds / fair_home_odds) - 1) * 100,
            ((bm_draw_odds / fair_draw_odds) - 1) * 100,
            ((bm_away_odds / fair_away_odds) - 1) * 100
        ]
    })
    
    value_df['Is Value Bet?'] = value_df['Value (%)'] > 0
    
    # Display table
    st.dataframe(
        value_df.style.format({
            'Fair Odds': '{:.2f}',
            'Bookmaker Odds': '{:.2f}',
            'Value (%)': '{:+.2f}%'
        }).apply(lambda x: ['background-color: #90EE90' if v else '' 
                           for v in (x.name == 'Value (%)' and value_df['Is Value Bet?'])], axis=0),
        hide_index=True,
        use_container_width=True
    )
    
    # Highlight value bets
    value_bets = value_df[value_df['Is Value Bet?']]
    
    if len(value_bets) > 0:
        st.success(f"✅ **{len(value_bets)} Value Bet(s) Identified!**")
        for _, row in value_bets.iterrows():
            st.write(f"""
            - **{row['Result']}**: Bookmaker odds {row['Bookmaker Odds']:.2f} vs Fair odds {row['Fair Odds']:.2f} 
              ({row['Value (%)']:+.1f}% value)
            """)
    else:
        st.warning("⚠️ No value bets found with current bookmaker odds.")
    
    st.divider()
    
    st.subheader("Expected Value Calculator")
    st.markdown("Calculate potential profit from value betting over multiple matches.")
    
    col1, col2 = st.columns(2)
    with col1:
        stake = st.number_input("Stake per bet (£)", min_value=1, max_value=1000, value=10)
        num_bets = st.number_input("Number of bets", min_value=1, max_value=1000, value=100)
    
    with col2:
        if len(value_bets) > 0:
            selected_bet = st.selectbox("Select value bet", value_bets['Result'].tolist())
            bet_data = value_bets[value_bets['Result'] == selected_bet].iloc[0]
            
            probability = results_df[results_df['Result'] == selected_bet]['Probability (%)'].values[0] / 100
            odds = bet_data['Bookmaker Odds']
            
            expected_value = (probability * odds * stake) - stake
            total_ev = expected_value * num_bets
            roi = (expected_value / stake) * 100
            
            st.metric("Expected Value per bet", f"£{expected_value:.2f}")
            st.metric("Total EV over all bets", f"£{total_ev:.2f}")
            st.metric("ROI per bet", f"{roi:.2f}%")
        else:
            st.info("No value bets available for calculation")

with tab4:
    st.header("About This System")
    
    st.markdown("""
    ### Goal Superiority Rating System
    
    This application implements the rating system described in the Football-Data article 
    *"Rating Systems for Fixed Odds Football Match Prediction"* by Joe Buchdahl.
    
    #### How It Works
    
    1. **Calculate Team Ratings**: Each team's rating = goals scored - goals conceded (last 6 matches)
    
    2. **Match Rating**: Home team rating - Away team rating
    
    3. **Probability Calculation**: Using best-fit equations derived from 14,002 English league matches (1993-2001):
       - Home Win: `y = 1.56x + 46.47` (R² = 0.86)
       - Draw: `y = -0.03x² - 0.29x + 29.48` (R² = 0.39)
       - Away Win: `y = 0.03x² - 1.27x + 23.65` (R² = 0.75)
    
    4. **Fair Odds**: Calculated as `100 / probability`
    
    5. **Value Betting**: Compare fair odds to bookmaker odds to identify profitable opportunities
    
    #### Key Findings
    
    The original research showed:
    - **+2.1% yield** betting on all value home wins (2001/02 season)
    - **+10.1% yield** for match ratings between -2 and +2 (most reliable range)
    - Predictions most reliable for ratings close to 0 (evenly matched teams)
    
    #### Limitations
    
    - Does not account for **quality of opposition** in the simple form
    - Based on **historical data** (1993-2001 English leagues)
    - **Draw predictions** have lower reliability (R² = 0.39)
    - Requires at least 6 matches of form data
    
    #### References
    
    Source: Football-Data.co.uk - "Rating Systems for Fixed Odds Football Match Prediction"
    
    Material adapted from *Fixed Odds Sports Betting: The Essential Guide* by Joe Buchdahl
    """)

# Footer
st.divider()
st.caption("⚽ Football Ratings System | Based on Goal Superiority Rating methodology | Data-driven match predictions")
