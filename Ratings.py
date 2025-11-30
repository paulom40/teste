import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson

# Page config
st.set_page_config(page_title="Football Ratings System", page_icon="⚽", layout="wide")

# Title and description
st.title("⚽ Football Match Rating & Prediction System")
st.markdown("""
This app implements the **Goal Superiority Rating System** described in the Football-Data article.
It calculates match ratings based on recent form and predicts match outcomes with fair odds.
""")

# Instructions
with st.expander("📖 How to Use This App", expanded=True):
    st.markdown("""
    ### Quick Start Guide
    
    1. **Enter the team names** in the sidebar (default: Tottenham vs Leeds)
    
    2. **Input goals scored/conceded** for the last 6 matches for each team
       - Home team recent form
       - Away team recent form
    
    3. **(Optional) Enter bookmaker odds** to identify value bets
       - Home win odds
       - Draw odds
       - Away win odds
    
    4. **Explore the tabs** to see:
       - 📊 Match predictions and probabilities
       - 📈 Rating analysis and confidence levels
       - 💰 Value betting opportunities
       - 📚 Methodology and research background
    
    ---
    
    **About the System:** This app uses exact equations from the original research with R² values of 
    **0.86** for home wins, **0.75** for away wins, and **0.39** for draws, based on 14,002 English 
    league matches from 1993-2001. The system achieved a **+10.1% yield** for matches with ratings 
    between -2 and +2 in the 2001/02 season.
    
    The **Wisdom of the Crowd** method removes bookmaker margins to find true probabilities, achieving 
    **+3.4% yield** across 22,318 European matches (2012-2015).
    
    The **Monte Carlo simulation** uses Poisson distribution to model goal scoring, providing exact score 
    probabilities and alternative betting markets.
    
    The **xG Calculator** helps estimate goal-scoring probability based on shot characteristics like distance, 
    angle, body part, and defensive pressure.
    """)

st.divider()

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Match Prediction", "📈 Rating Analysis", "💰 Value Bets", "🧠 Wisdom of Crowd", "🎯 xG Calculator", "🎲 Monte Carlo", "📚 About"])

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
    
    # Display table with conditional formatting
    def highlight_value(row):
        if row['Is Value Bet?']:
            return ['background-color: #90EE90'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        value_df.style.format({
            'Fair Odds': '{:.2f}',
            'Bookmaker Odds': '{:.2f}',
            'Value (%)': '{:+.2f}%'
        }).apply(highlight_value, axis=1),
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
    st.header("🧠 Wisdom of the Crowd")
    
    st.markdown("""
    This method uses market odds (particularly from sharp bookmakers like Pinnacle) to estimate 
    "true" probabilities by removing the bookmaker's margin. The betting market collectively 
    contains the wisdom of thousands of bettors.
    """)
    
    st.divider()
    
    st.subheader("Fair Odds Calculator")
    st.markdown("Remove bookmaker margins to find the 'true' probabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Input Bookmaker Odds")
        wotc_home_odds = st.number_input("Home Win Odds", min_value=1.01, max_value=100.0, value=2.50, step=0.01, key="wotc_home")
        wotc_draw_odds = st.number_input("Draw Odds", min_value=1.01, max_value=100.0, value=3.40, step=0.01, key="wotc_draw")
        wotc_away_odds = st.number_input("Away Win Odds", min_value=1.01, max_value=100.0, value=2.90, step=0.01, key="wotc_away")
        
        # Calculate margin
        margin = (1/wotc_home_odds + 1/wotc_draw_odds + 1/wotc_away_odds) - 1
        overround = (margin + 1) * 100
        
        st.metric("Book Margin", f"{margin*100:.2f}%")
        st.metric("Overround", f"{overround:.2f}%")
    
    with col2:
        st.markdown("#### Fair Odds (Margin Removed)")
        
        # Calculate fair odds using differential margin weighting
        # Fair odds = (3 × Published odds) / (3 - Margin × Published odds)
        fair_home = (3 * wotc_home_odds) / (3 - margin * wotc_home_odds)
        fair_draw = (3 * wotc_draw_odds) / (3 - margin * wotc_draw_odds)
        fair_away = (3 * wotc_away_odds) / (3 - margin * wotc_away_odds)
        
        # Calculate implied probabilities
        fair_home_prob = (1 / fair_home) * 100
        fair_draw_prob = (1 / fair_draw) * 100
        fair_away_prob = (1 / fair_away) * 100
        
        st.metric("Fair Home Odds", f"{fair_home:.2f}", delta=f"{fair_home_prob:.1f}% probability")
        st.metric("Fair Draw Odds", f"{fair_draw:.2f}", delta=f"{fair_draw_prob:.1f}% probability")
        st.metric("Fair Away Odds", f"{fair_away:.2f}", delta=f"{fair_away_prob:.1f}% probability")
        
        # Verify probabilities sum to 100%
        total_fair_prob = fair_home_prob + fair_draw_prob + fair_away_prob
        st.caption(f"Total probability: {total_fair_prob:.2f}% (should be ~100%)")
    
    st.divider()
    
    st.subheader("Compare Against Other Bookmakers")
    st.markdown("Find value bets by comparing other bookmaker odds to the fair odds")
    
    # Create input for multiple bookmakers
    num_bookies = st.number_input("Number of bookmakers to compare", min_value=1, max_value=10, value=3)
    
    comparison_data = []
    
    for i in range(num_bookies):
        with st.expander(f"Bookmaker {i+1}", expanded=(i==0)):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                bookie_name = st.text_input("Name", f"Bookie {i+1}", key=f"name_{i}")
            with col2:
                bookie_home = st.number_input("Home", 1.01, 100.0, 2.60, 0.01, key=f"home_{i}")
            with col3:
                bookie_draw = st.number_input("Draw", 1.01, 100.0, 3.50, 0.01, key=f"draw_{i}")
            with col4:
                bookie_away = st.number_input("Away", 1.01, 100.0, 3.00, 0.01, key=f"away_{i}")
            
            # Calculate value
            home_value = ((bookie_home / fair_home) - 1) * 100
            draw_value = ((bookie_draw / fair_draw) - 1) * 100
            away_value = ((bookie_away / fair_away) - 1) * 100
            
            comparison_data.append({
                'Bookmaker': bookie_name,
                'Home Odds': bookie_home,
                'Home Value %': home_value,
                'Draw Odds': bookie_draw,
                'Draw Value %': draw_value,
                'Away Odds': bookie_away,
                'Away Value %': away_value
            })
    
    # Display comparison table
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        
        # Highlight positive values
        def highlight_positive(val):
            if isinstance(val, (int, float)) and val > 0:
                return 'background-color: #90EE90'
            return ''
        
        st.dataframe(
            comp_df.style.format({
                'Home Odds': '{:.2f}',
                'Home Value %': '{:+.2f}',
                'Draw Odds': '{:.2f}',
                'Draw Value %': '{:+.2f}',
                'Away Odds': '{:.2f}',
                'Away Value %': '{:+.2f}'
            }).applymap(highlight_positive),
            hide_index=True,
            use_container_width=True
        )
        
        # Find best values
        st.subheader("Best Value Opportunities")
        
        all_values = []
        for row in comparison_data:
            if row['Home Value %'] > 0:
                all_values.append((row['Bookmaker'], 'Home Win', row['Home Odds'], row['Home Value %']))
            if row['Draw Value %'] > 0:
                all_values.append((row['Bookmaker'], 'Draw', row['Draw Odds'], row['Draw Value %']))
            if row['Away Value %'] > 0:
                all_values.append((row['Bookmaker'], 'Away Win', row['Away Odds'], row['Away Value %']))
        
        if all_values:
            # Sort by value
            all_values.sort(key=lambda x: x[3], reverse=True)
            
            for bookie, outcome, odds, value in all_values[:5]:  # Show top 5
                st.success(f"**{bookie}** - {outcome}: {odds:.2f} odds ({value:+.2f}% value)")
        else:
            st.info("No value bets found. Try adjusting the odds.")
    
    st.divider()
    
    st.markdown("""
    ### How It Works
    
    1. **The Wisdom of the Crowd**: Betting markets aggregate the opinions of thousands of bettors, 
       creating remarkably accurate probability estimates
    
    2. **Margin Removal**: Bookmakers add a margin (overround) to make profit. By removing this using 
       differential weighting, we can estimate the "true" odds
    
    3. **Differential Weighting**: The formula accounts for the favourite-longshot bias where bookmakers 
       shorten longer odds more than shorter ones
    
    4. **Finding Value**: Compare other bookmaker odds to these fair odds to identify where the market 
       has made mistakes
    
    **Formula Used:**
    ```
    Fair Odds = (3 × Published Odds) / (3 - Margin × Published Odds)
    ```
    
    **Research Results** (2012/13 to 2014/15, 22,318 matches):
    - Betting all value opportunities (where odds > fair odds): **+3.4% yield**
    - Betting only when advantage > 3%: **+8.8% yield**
    - The fair odds broke even (0.08% yield), confirming market accuracy
    """)

with tab5:
    st.header("🎯 Expected Goals (xG) Calculator")
    
    st.markdown("""
    Calculate Expected Goals for individual shots or aggregate team xG based on multiple factors.
    This tool helps estimate the quality of goal-scoring opportunities.
    """)
    
    # Mode selection
    calc_mode = st.radio(
        "Calculation Mode",
        ["Single Shot xG", "Team Match xG Builder"],
        horizontal=True
    )
    
    st.divider()
    
    if calc_mode == "Single Shot xG":
        st.subheader("Calculate xG for a Single Shot")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Shot Location")
            
            shot_distance = st.slider("Distance from Goal (meters)", 1, 35, 12, 1)
            shot_angle = st.slider("Angle from Center (degrees)", 0, 90, 15, 5,
                                  help="0° = straight on, 90° = from sideline")
            
            # Visual representation
            st.caption(f"**Shot Position**: {shot_distance}m at {shot_angle}° angle")
            
            # Calculate goal mouth width visible
            import math
            if shot_angle == 0:
                goal_width_visible = 7.32  # Full goal width
            else:
                # Approximate visible goal width based on angle
                goal_width_visible = 7.32 * math.cos(math.radians(shot_angle))
            
            st.caption(f"Visible goal width: {goal_width_visible:.2f}m")
        
        with col2:
            st.markdown("#### Shot Characteristics")
            
            body_part = st.selectbox(
                "Body Part",
                ["Foot (Right)", "Foot (Left)", "Header", "Other"]
            )
            
            shot_type = st.selectbox(
                "Shot Type",
                ["Regular Shot", "Volley", "Half-Volley", "Diving Header"]
            )
            
            assist_type = st.selectbox(
                "Assist Type",
                ["None (Dribble)", "Through Ball", "Cross", "Cut Back", "Set Piece", "Corner"],
                help="How did the ball arrive to the shooter?"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### Game Context")
            
            defenders_nearby = st.slider("Defenders Within 2m", 0, 5, 1)
            
            big_chance = st.checkbox("Big Chance", 
                                    help="Clear goal-scoring opportunity (1-on-1, open goal, etc.)")
            
            counter_attack = st.checkbox("Counter Attack")
        
        with col4:
            st.markdown("#### Goalkeeper")
            
            gk_position = st.selectbox(
                "Goalkeeper Position",
                ["Normal Position", "Out of Position", "Off Line"]
            )
            
            one_on_one = st.checkbox("1-on-1 with Keeper")
        
        # Calculate xG using a simplified logistic regression model
        # Base coefficients (simplified version of real models)
        import math
        
        def calculate_xg(distance, angle, body_part, shot_type, assist_type, 
                        defenders, big_chance, counter, gk_pos, one_v_one):
            
            # Start with base intercept
            logit = -1.5
            
            # Distance factor (negative - further = worse)
            logit += -0.09 * distance
            
            # Angle factor (negative - wider angle = worse)
            logit += -0.02 * angle
            
            # Body part adjustments
            if "Header" in body_part:
                logit += -0.4
            elif "Foot" in body_part:
                logit += 0.1
            
            # Shot type adjustments
            if shot_type == "Volley":
                logit += -0.2
            elif shot_type == "Half-Volley":
                logit += -0.1
            elif shot_type == "Diving Header":
                logit += -0.5
            
            # Assist type adjustments
            if assist_type == "Through Ball":
                logit += 0.4
            elif assist_type == "Cross":
                logit += -0.2
            elif assist_type == "Cut Back":
                logit += 0.5
            elif assist_type == "Set Piece":
                logit += 0.2
            elif assist_type == "Corner":
                logit += -0.1
            
            # Defensive pressure (negative)
            logit += -0.3 * defenders
            
            # Big chance bonus
            if big_chance:
                logit += 0.8
            
            # Counter attack bonus
            if counter:
                logit += 0.3
            
            # Goalkeeper position
            if gk_pos == "Out of Position":
                logit += 0.6
            elif gk_pos == "Off Line":
                logit += 0.4
            
            # 1-on-1 bonus
            if one_v_one:
                logit += 0.7
            
            # Convert logit to probability using sigmoid function
            xg = 1 / (1 + math.exp(-logit))
            
            return xg
        
        calculated_xg = calculate_xg(
            shot_distance, shot_angle, body_part, shot_type, assist_type,
            defenders_nearby, big_chance, counter_attack, gk_position, one_on_one
        )
        
        st.divider()
        
        # Display result
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Expected Goals (xG)", f"{calculated_xg:.3f}")
        
        with col2:
            st.metric("Conversion Rate", f"{calculated_xg*100:.1f}%")
        
        with col3:
            shots_to_score = 1 / calculated_xg if calculated_xg > 0 else 999
            st.metric("Expected Shots to Score", f"{shots_to_score:.1f}")
        
        # Quality rating
        if calculated_xg >= 0.5:
            quality = "🔥 Excellent Chance"
            color = "green"
        elif calculated_xg >= 0.3:
            quality = "✅ Good Chance"
            color = "blue"
        elif calculated_xg >= 0.15:
            quality = "⚠️ Half Chance"
            color = "orange"
        elif calculated_xg >= 0.05:
            quality = "📉 Poor Chance"
            color = "orange"
        else:
            quality = "❌ Very Difficult"
            color = "red"
        
        st.markdown(f"**Quality Rating:** :{color}[{quality}]")
        
        # Comparison examples
        st.divider()
        st.subheader("Reference Examples")
        
        ref_col1, ref_col2, ref_col3 = st.columns(3)
        
        with ref_col1:
            st.markdown("**Penalty Kick**")
            st.metric("xG", "0.76 - 0.79")
            st.caption("Standard conversion rate for penalties")
        
        with ref_col2:
            st.markdown("**6-Yard Box Tap-in**")
            st.metric("xG", "0.80 - 0.95")
            st.caption("Close range, open goal")
        
        with ref_col3:
            st.markdown("**Long Range Strike**")
            st.metric("xG", "0.01 - 0.05")
            st.caption("25+ meters, low success rate")
    
    else:  # Team Match xG Builder
        st.subheader("Build Team xG from Multiple Shots")
        
        st.markdown("""
        Add multiple shots to calculate the total Expected Goals for a team in a match.
        This helps evaluate overall attacking performance.
        """)
        
        # Initialize session state for shots
        if 'shots_list' not in st.session_state:
            st.session_state.shots_list = []
        
        with st.expander("➕ Add New Shot", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                add_distance = st.number_input("Distance (m)", 1, 35, 12, key="add_dist")
                add_angle = st.number_input("Angle (°)", 0, 90, 15, key="add_angle")
            
            with col2:
                add_body = st.selectbox("Body Part", ["Foot", "Header"], key="add_body")
                add_assist = st.selectbox("Assist", ["Open Play", "Through Ball", "Cross", "Set Piece"], key="add_assist")
            
            with col3:
                add_defenders = st.number_input("Defenders", 0, 5, 1, key="add_def")
                add_big = st.checkbox("Big Chance", key="add_big")
            
            with col4:
                add_result = st.selectbox("Result", ["Goal ⚽", "Saved", "Missed", "Blocked"], key="add_result")
                
                if st.button("➕ Add Shot", type="primary", use_container_width=True):
                    # Calculate xG for this shot
                    logit = -1.5 - 0.09 * add_distance - 0.02 * add_angle
                    if add_body == "Header":
                        logit -= 0.4
                    if add_assist == "Through Ball":
                        logit += 0.4
                    elif add_assist == "Cross":
                        logit -= 0.2
                    elif add_assist == "Set Piece":
                        logit += 0.2
                    logit -= 0.3 * add_defenders
                    if add_big:
                        logit += 0.8
                    
                    shot_xg = 1 / (1 + math.exp(-logit))
                    
                    st.session_state.shots_list.append({
                        'Distance': add_distance,
                        'Angle': add_angle,
                        'Body': add_body,
                        'Assist': add_assist,
                        'Defenders': add_defenders,
                        'Big Chance': add_big,
                        'Result': add_result,
                        'xG': shot_xg
                    })
                    st.success(f"Shot added! xG: {shot_xg:.3f}")
                    st.rerun()
        
        # Display shots table
        if st.session_state.shots_list:
            st.divider()
            st.subheader(f"Shots Summary ({len(st.session_state.shots_list)} shots)")
            
            shots_df = pd.DataFrame(st.session_state.shots_list)
            
            # Format the dataframe
            display_df = shots_df.copy()
            display_df['xG'] = display_df['xG'].apply(lambda x: f"{x:.3f}")
            display_df['Big Chance'] = display_df['Big Chance'].apply(lambda x: '✓' if x else '')
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Calculate totals
            total_xg = shots_df['xG'].sum()
            goals_scored = len(shots_df[shots_df['Result'] == 'Goal ⚽'])
            shots_on_target = len(shots_df[shots_df['Result'].isin(['Goal ⚽', 'Saved'])])
            big_chances_count = shots_df['Big Chance'].sum()
            
            # Performance metrics
            st.divider()
            st.subheader("Match Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total xG", f"{total_xg:.2f}")
            
            with col2:
                st.metric("Goals Scored", goals_scored, delta=f"{goals_scored - total_xg:+.2f} vs xG")
            
            with col3:
                st.metric("Shots on Target", f"{shots_on_target}/{len(shots_df)}")
            
            with col4:
                st.metric("Big Chances", big_chances_count)
            
            # Performance analysis
            st.divider()
            st.subheader("Performance Analysis")
            
            xg_diff = goals_scored - total_xg
            
            if xg_diff > 1:
                st.success(f"""
                **🔥 Outstanding Finishing!** The team scored {goals_scored} goal(s) from {total_xg:.2f} xG, 
                outperforming by {xg_diff:+.2f}. This suggests exceptional finishing or good fortune.
                """)
            elif xg_diff > 0:
                st.info(f"""
                **✅ Above Average Finishing.** The team scored {goals_scored} goal(s) from {total_xg:.2f} xG, 
                slightly outperforming by {xg_diff:+.2f}.
                """)
            elif xg_diff > -1:
                st.warning(f"""
                **⚠️ Below Average Finishing.** The team scored {goals_scored} goal(s) from {total_xg:.2f} xG, 
                underperforming by {xg_diff:.2f}. More clinical finishing needed.
                """)
            else:
                st.error(f"""
                **❌ Poor Finishing.** The team scored {goals_scored} goal(s) from {total_xg:.2f} xG, 
                significantly underperforming by {xg_diff:.2f}. Major finishing issues.
                """)
            
            # Shot map visualization
            st.divider()
            st.subheader("Shot Map")
            
            # Create a simple shot map
            fig = go.Figure()
            
            # Add pitch outline (simplified)
            fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, 
                         line=dict(color="green", width=2))
            
            # Add goal
            fig.add_shape(type="rect", x0=45, y0=0, x1=55, y1=5,
                         line=dict(color="white", width=3), fillcolor="lightgray")
            
            # Plot shots
            for idx, shot in shots_df.iterrows():
                # Convert distance and angle to x,y coordinates
                x = 50 + shot['Angle'] * 0.5 * (1 if idx % 2 == 0 else -1)
                y = shot['Distance'] * 2
                
                # Color based on result
                if shot['Result'] == 'Goal ⚽':
                    color = 'green'
                    symbol = 'circle'
                elif shot['Result'] == 'Saved':
                    color = 'orange'
                    symbol = 'circle'
                elif shot['Result'] == 'Blocked':
                    color = 'red'
                    symbol = 'x'
                else:
                    color = 'gray'
                    symbol = 'x'
                
                # Size based on xG
                size = max(10, shot['xG'] * 50)
                
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers',
                    marker=dict(size=size, color=color, symbol=symbol, 
                              line=dict(width=2, color='white')),
                    name=f"{shot['Result']} ({shot['xG']:.2f})",
                    hovertemplate=f"<b>{shot['Result']}</b><br>" +
                                 f"xG: {shot['xG']:.3f}<br>" +
                                 f"{shot['Distance']}m, {shot['Angle']}°<br>" +
                                 f"{shot['Body']}, {shot['Assist']}<extra></extra>"
                ))
            
            fig.update_layout(
                title="Shot Locations (size = xG value)",
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                showlegend=False,
                height=500,
                plot_bgcolor='rgba(0,100,0,0.1)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Clear button
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🗑️ Clear All Shots", type="secondary"):
                    st.session_state.shots_list = []
                    st.rerun()
        
        else:
            st.info("👆 Add shots using the form above to build your team's xG")
    
    st.divider()
    
    st.markdown("""
    ### Understanding xG Values
    
    | xG Range | Quality | Example |
    |----------|---------|---------|
    | 0.76-0.79 | Penalty | Standard penalty kick |
    | 0.50-1.00 | Big Chance | 1-on-1, tap-in, open goal |
    | 0.30-0.49 | Good Chance | Inside box, decent position |
    | 0.15-0.29 | Half Chance | Edge of box, some pressure |
    | 0.05-0.14 | Poor Chance | Outside box, difficult angle |
    | 0.00-0.04 | Very Low | Long range, tight angle |
    
    ### Key Factors in xG Calculation
    
    1. **Shot Location** (Most Important)
       - Distance to goal (closer = higher xG)
       - Angle to goal (center = higher xG)
       - Inside vs outside penalty box
    
    2. **Shot Type**
       - Foot shots generally better than headers
       - Volleys are more difficult
       - Diving headers very difficult
    
    3. **Assist Type**
       - Through balls create better chances
       - Cut backs are very dangerous
       - Crosses typically lower xG (headers)
    
    4. **Defensive Context**
       - Number of defenders nearby
       - Goalkeeper position
       - Counter-attack situations
    
    5. **Big Chances**
       - Clear goal-scoring opportunities
       - 1-on-1 with goalkeeper
       - Open goal situations
    
    ### Using xG for Analysis
    
    - **Team Performance**: Total xG shows quality of chances created
    - **Finishing Quality**: Compare goals scored to xG
    - **Underperformance**: xG > goals suggests poor finishing or bad luck
    - **Overperformance**: Goals > xG suggests great finishing or good luck
    - **Expected**: Over time, goals should roughly match xG
    
    ### Limitations
    
    - Doesn't account for individual player skill
    - Historical averages may not reflect specific situations
    - Can't predict individual outcomes, only probabilities
    - Different models use different variables
    """)

with tab7:
    st.header("🎲 Monte Carlo Simulation")
    
    st.markdown("""
    This model simulates match outcomes thousands of times using **Poisson distribution** for goal scoring.
    It provides detailed probability distributions for exact scores and alternative betting markets.
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Expected Goals (xG)")
        st.markdown("Enter the expected goals for each team based on their attacking/defensive strength")
        
        home_xg = st.number_input("Home Team Expected Goals", min_value=0.1, max_value=5.0, value=1.5, step=0.1, key="mc_home_xg")
        away_xg = st.number_input("Away Team Expected Goals", min_value=0.1, max_value=5.0, value=1.2, step=0.1, key="mc_away_xg")
        
        st.info("""
        **Tip**: Calculate xG from:
        - Recent goals scored/conceded averages
        - League average adjustments
        - Home advantage factor (~1.3x multiplier)
        """)
    
    with col2:
        st.subheader("Simulation Settings")
        num_simulations = st.select_slider(
            "Number of Simulations",
            options=[1000, 5000, 10000, 50000, 100000],
            value=10000
        )
        
        max_goals = st.slider("Maximum Goals per Team", 5, 10, 8)
        
        st.metric("Simulations", f"{num_simulations:,}")
        st.caption(f"Calculating {num_simulations:,} random matches...")
    
    # Run Monte Carlo simulation
    if st.button("🎲 Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Running Monte Carlo simulation..."):
            # Simulate using Poisson distribution
            np.random.seed(42)
            home_goals_sim = np.random.poisson(home_xg, num_simulations)
            away_goals_sim = np.random.poisson(away_xg, num_simulations)
            
            # Calculate results
            home_wins = np.sum(home_goals_sim > away_goals_sim)
            draws = np.sum(home_goals_sim == away_goals_sim)
            away_wins = np.sum(home_goals_sim < away_goals_sim)
            
            # Calculate probabilities
            home_win_prob = (home_wins / num_simulations) * 100
            draw_prob = (draws / num_simulations) * 100
            away_win_prob = (away_wins / num_simulations) * 100
            
            # Calculate fair odds
            mc_fair_home_odds = 100 / home_win_prob
            mc_fair_draw_odds = 100 / draw_prob
            mc_fair_away_odds = 100 / away_win_prob
            
            st.success("✅ Simulation Complete!")
            
            st.divider()
            
            # Display main results
            st.subheader("Match Outcome Probabilities")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    f"{home_team} Win",
                    f"{home_win_prob:.2f}%",
                    delta=f"Fair odds: {mc_fair_home_odds:.2f}"
                )
            
            with col2:
                st.metric(
                    "Draw",
                    f"{draw_prob:.2f}%",
                    delta=f"Fair odds: {mc_fair_draw_odds:.2f}"
                )
            
            with col3:
                st.metric(
                    f"{away_team} Win",
                    f"{away_win_prob:.2f}%",
                    delta=f"Fair odds: {mc_fair_away_odds:.2f}"
                )
            
            # Probability distribution chart
            results_df_mc = pd.DataFrame({
                'Result': [f'{home_team} Win', 'Draw', f'{away_team} Win'],
                'Probability (%)': [home_win_prob, draw_prob, away_win_prob],
                'Simulations': [home_wins, draws, away_wins]
            })
            
            fig = px.bar(
                results_df_mc,
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
            
            st.divider()
            
            # Exact score probabilities
            st.subheader("Most Likely Exact Scores")
            
            # Calculate exact score probabilities
            from scipy.stats import poisson
            
            score_probs = []
            for h in range(max_goals + 1):
                for a in range(max_goals + 1):
                    prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg) * 100
                    score_probs.append({
                        'Score': f"{h}-{a}",
                        'Home': h,
                        'Away': a,
                        'Probability (%)': prob,
                        'Fair Odds': 100 / prob if prob > 0 else 999
                    })
            
            # Sort by probability
            score_probs_df = pd.DataFrame(score_probs).sort_values('Probability (%)', ascending=False)
            
            # Display top 10 scores
            top_scores = score_probs_df.head(10)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_scores = px.bar(
                    top_scores,
                    x='Score',
                    y='Probability (%)',
                    text='Probability (%)',
                    title='Top 10 Most Likely Scores'
                )
                fig_scores.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig_scores.update_layout(height=400)
                st.plotly_chart(fig_scores, use_container_width=True)
            
            with col2:
                st.dataframe(
                    top_scores[['Score', 'Probability (%)', 'Fair Odds']].style.format({
                        'Probability (%)': '{:.2f}%',
                        'Fair Odds': '{:.2f}'
                    }),
                    hide_index=True,
                    use_container_width=True,
                    height=400
                )
            
            st.divider()
            
            # Alternative markets
            st.subheader("Alternative Betting Markets")
            
            col1, col2, col3 = st.columns(3)
            
            # Over/Under goals
            with col1:
                st.markdown("**Over/Under 2.5 Goals**")
                
                over_25_count = np.sum((home_goals_sim + away_goals_sim) > 2.5)
                under_25_count = num_simulations - over_25_count
                
                over_25_prob = (over_25_count / num_simulations) * 100
                under_25_prob = (under_25_count / num_simulations) * 100
                
                st.metric("Over 2.5", f"{over_25_prob:.1f}%", delta=f"Odds: {100/over_25_prob:.2f}")
                st.metric("Under 2.5", f"{under_25_prob:.1f}%", delta=f"Odds: {100/under_25_prob:.2f}")
            
            # Both teams to score
            with col2:
                st.markdown("**Both Teams to Score**")
                
                btts_yes = np.sum((home_goals_sim > 0) & (away_goals_sim > 0))
                btts_no = num_simulations - btts_yes
                
                btts_yes_prob = (btts_yes / num_simulations) * 100
                btts_no_prob = (btts_no / num_simulations) * 100
                
                st.metric("Yes", f"{btts_yes_prob:.1f}%", delta=f"Odds: {100/btts_yes_prob:.2f}")
                st.metric("No", f"{btts_no_prob:.1f}%", delta=f"Odds: {100/btts_no_prob:.2f}")
            
            # Total goals exact
            with col3:
                st.markdown("**Most Likely Total Goals**")
                
                total_goals = home_goals_sim + away_goals_sim
                unique_totals, counts = np.unique(total_goals, return_counts=True)
                
                for i in range(min(3, len(unique_totals))):
                    idx = np.argsort(counts)[-i-1]
                    total = unique_totals[idx]
                    prob = (counts[idx] / num_simulations) * 100
                    st.metric(f"{total} goals", f"{prob:.1f}%", delta=f"Odds: {100/prob:.2f}")
            
            st.divider()
            
            # Goal distribution heatmap
            st.subheader("Score Probability Heatmap")
            
            # Create matrix for heatmap
            score_matrix = np.zeros((max_goals + 1, max_goals + 1))
            for h in range(max_goals + 1):
                for a in range(max_goals + 1):
                    score_matrix[a, h] = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg) * 100
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=score_matrix,
                x=[str(i) for i in range(max_goals + 1)],
                y=[str(i) for i in range(max_goals + 1)],
                colorscale='Viridis',
                text=score_matrix,
                texttemplate='%{text:.1f}%',
                textfont={"size": 10},
                colorbar=dict(title="Probability (%)")
            ))
            
            fig_heatmap.update_layout(
                title="Score Probability Matrix",
                xaxis_title=f"{home_team} Goals",
                yaxis_title=f"{away_team} Goals",
                height=500
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.divider()
            
            # Expected statistics
            st.subheader("Expected Match Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_home_goals = np.mean(home_goals_sim)
                st.metric("Avg Home Goals", f"{avg_home_goals:.2f}")
            
            with col2:
                avg_away_goals = np.mean(away_goals_sim)
                st.metric("Avg Away Goals", f"{avg_away_goals:.2f}")
            
            with col3:
                avg_total_goals = np.mean(home_goals_sim + away_goals_sim)
                st.metric("Avg Total Goals", f"{avg_total_goals:.2f}")
            
            with col4:
                goal_diff = np.mean(home_goals_sim - away_goals_sim)
                st.metric("Avg Goal Difference", f"{goal_diff:+.2f}")
    
    else:
        st.info("👆 Click 'Run Simulation' to generate predictions")
    
    st.divider()
    
    st.markdown("""
    ### How Monte Carlo Simulation Works
    
    1. **Poisson Distribution**: Goals in football follow a Poisson distribution, where the probability 
       of scoring X goals depends on the expected goals (λ parameter)
    
    2. **Random Sampling**: The simulation generates thousands of random match outcomes based on each 
       team's expected goals
    
    3. **Aggregation**: By counting how many simulations result in each outcome, we estimate the true 
       probability distribution
    
    4. **Law of Large Numbers**: With enough simulations (10,000+), the results converge to the theoretical 
       probabilities
    
    ### Advantages of Monte Carlo
    
    - ✅ Provides **exact score probabilities**
    - ✅ Can calculate **any betting market** (over/under, BTTS, handicaps, etc.)
    - ✅ **Accounts for variance** in goal scoring
    - ✅ Based on **statistical reality** (Poisson distribution fits football well)
    - ✅ **Flexible**: Easy to add factors like red cards, weather, etc.
    
    ### Calculating Expected Goals (xG)
    
    ```
    Home xG = (Home Attack Strength × Away Defense Weakness × League Avg Goals) × Home Advantage
    Away xG = (Away Attack Strength × Home Defense Weakness × League Avg Goals)
    ```
    
    Typical home advantage multiplier: **1.3x**
    """)

with tab6:
    st.header("About This System")
    
    st.markdown("""
    ### Four Powerful Rating Systems
    
    This app combines four complementary approaches to football match prediction:
    
    ---
    
    ### 1. Goal Superiority Rating System
    
    Implemented in tabs 1-3, based on Football-Data's article by Joe Buchdahl.
    
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
    
    - **+2.1% yield** betting on all value home wins (2001/02 season)
    - **+10.1% yield** for match ratings between -2 and +2 (most reliable range)
    - Predictions most reliable for ratings close to 0 (evenly matched teams)
    
    ---
    
    ### 2. Wisdom of the Crowd System
    
    Implemented in tab 4, based on the principle that betting markets aggregate collective intelligence.
    
    #### How It Works
    
    1. **Market Wisdom**: Sharp bookmakers like Pinnacle reflect the collective knowledge of thousands of bettors
    
    2. **Margin Removal**: Remove the bookmaker's profit margin using differential weighting:
       ```
       Fair Odds = (3 × Published Odds) / (3 - Margin × Published Odds)
       ```
    
    3. **Favourite-Longshot Bias**: The formula accounts for bookmakers shortening longer odds more than shorter ones
    
    4. **Value Identification**: Compare other bookmakers' odds to the fair odds to find mistakes
    
    #### Key Findings (2012/13 to 2014/15, 22,318 matches)
    
    - **+3.4% yield** betting all value opportunities
    - **+8.8% yield** when value advantage > 3%
    - Fair odds broke even (0.08% yield), confirming market accuracy
    - Works with 67% of opportunities even in overround books
    
    ---
    
    ### 3. Monte Carlo Simulation
    
    Implemented in tab 5, uses statistical modeling of goal-scoring patterns.
    
    #### How It Works
    
    1. **Poisson Distribution**: Models goal scoring based on expected goals (xG) for each team
    
    2. **Random Sampling**: Simulates thousands of matches by randomly generating goals according to Poisson distribution
    
    3. **Probability Calculation**: Aggregates results to determine outcome probabilities
    
    4. **Exact Scores**: Can predict probability of any specific scoreline (e.g., 2-1, 0-0, etc.)
    
    5. **Alternative Markets**: Calculates probabilities for over/under, BTTS, and other betting markets
    
    #### Key Advantages
    
    - **Exact score probabilities** not available from other methods
    - **Any betting market** can be calculated (corners, cards, etc. with proper data)
    - **Statistically grounded** in Poisson distribution (proven to fit football)
    - **Flexible modeling** - can incorporate additional factors
    
    #### Calculating Expected Goals
    
    ```
    Home xG = (Home Attack × Away Defense × League Avg) × Home Advantage (1.3x)
    Away xG = (Away Attack × Home Defense × League Avg)
    ```
    
    ---
    
    ### 4. Expected Goals (xG) Calculator
    
    Implemented in tab 5, calculates individual shot quality and team attacking performance.
    
    #### How It Works
    
    1. **Shot Characteristics**: Considers distance, angle, body part, shot type
    
    2. **Context Factors**: Assist type, defensive pressure, goalkeeper position
    
    3. **Logistic Regression**: Uses mathematical model to calculate probability (0 to 1)
    
    4. **Quality Assessment**: Rates chances from "Very Difficult" to "Excellent"
    
    5. **Team Builder**: Aggregate multiple shots to calculate total match xG
    
    #### Key Applications
    
    - **Individual Shot Analysis**: Evaluate quality of specific chances
    - **Team Performance**: Measure attacking effectiveness
    - **Finishing Quality**: Compare actual goals to expected goals
    - **Player Evaluation**: Assess strikers' conversion rates vs xG
    - **Match Analysis**: See who created better quality chances
    
    #### The Formula
    
    ```
    xG = 1 / (1 + e^(-logit))
    
    where logit includes:
    - Base intercept
    - Distance penalty
    - Angle penalty  
    - Body part adjustments
    - Assist type bonuses
    - Defensive pressure penalties
    - Context bonuses (big chance, 1v1, etc.)
    ```
    
    ---
    
    ### Which System to Use?
    
    - **Goal Superiority**: Best for early predictions before markets are fully formed, or when you have detailed form data
    - **Wisdom of Crowd**: Best when sharp bookmaker odds are available; relies on market efficiency
    - **xG Calculator**: Best for evaluating individual chances and team attacking quality
    - **Monte Carlo**: Best for exact scores and alternative markets; requires expected goals estimation
    - **Combined**: Use all four! Agreement across systems = highest confidence
    
    ---
    
    ### Limitations
    
    **Goal Superiority:**
    - Does not account for **quality of opposition** in simple form
    - Based on **historical data** (1993-2001 English leagues)
    - **Draw predictions** have lower reliability (R² = 0.39)
    - Requires at least 6 matches of form data
    
    **Wisdom of Crowd:**
    - Requires access to sharp bookmaker odds (e.g., Pinnacle)
    - Assumes market efficiency and independence of opinions
    - Bookmakers may limit accounts of consistent winners
    - Margin removal model is simplified
    
    **Monte Carlo:**
    - Accuracy depends on **quality of xG estimates**
    - Assumes **Poisson distribution** (good for football, not perfect)
    - Does not account for **dynamic game situations** (red cards, weather, etc.)
    - Requires **more input data** to calculate xG properly
    
    **xG Calculator:**
    - Simplified model vs professional xG (StatsBomb, Opta use thousands of variables)
    - Doesn't account for **individual player skill**
    - Based on **historical averages**, not specific situations
    - Can't predict **individual outcomes**, only probabilities
    - Different providers use different models, values will vary
    
    ---
    
    ### References
    
    1. **Goal Superiority System**: Football-Data.co.uk - "Rating Systems for Fixed Odds Football Match Prediction" 
       - Material adapted from *Fixed Odds Sports Betting: The Essential Guide* by Joe Buchdahl
    
    2. **Wisdom of Crowd System**: Football-Data.co.uk - "The Wisdom of the Crowd"
       - Based on research by Francis Galton (1906) and Vernon Lomax Smith
    
    3. **Monte Carlo Method**: Based on Poisson distribution modeling of football scores
       - Widely used in sports analytics and betting
    
    4. **xG Calculator**: Based on logistic regression models used by analytics providers
       - Methodology adapted from StatsBomb, Opta, and academic research
    """)

# Footer
st.divider()
st.caption("⚽ Football Ratings System | Four proven methodologies: Goal Superiority + Wisdom of the Crowd + xG Calculator + Monte Carlo | Data-driven match predictions")
