import streamlit as st
import pandas as pd
import numpy as np

# Streamlit app title
st.title("🏆 Daily Soccer Acca Bets - Professional Tipster Edition (European Leagues Only)")
st.markdown("---")

# Introduction
st.header("Today's Best 5-Fold Accumulator (November 8, 2025)")
st.write("Focusing exclusively on European leagues: A balanced acca with solid picks from Bundesliga, Ligue 1, La Liga, Serie A, and Primeira Liga. Combined decimal odds: **22.10**. Stake £10 for potential £221.00 return!")
st.markdown("*Odds sourced from major bookmakers like Bet365. Always shop around and bet responsibly.*")

# Data for the bets (European leagues only)
bets_data = {
    'Match': [
        'Union Berlin vs Bayern Munich (Bundesliga, 14:30 GMT)',
        'Auxerre vs Lille (Ligue 1, 16:00 GMT)',
        'Real Sociedad vs Barcelona (La Liga, 17:30 GMT)',
        'AC Milan vs Juventus (Serie A, 19:45 GMT)',
        'Santa Clara vs Sporting Lisbon (Primeira Liga, 20:00 GMT)'
    ],
    'Selection': [
        'Bayern Munich to Win',
        'Lille to Win & Under 2.5 Goals',
        'Barcelona to Win',
        'AC Milan to Win',
        'Sporting Lisbon to Win & Over 1.5 Goals'
    ],
    'Decimal Odds': [1.45, 2.80, 1.85, 2.10, 1.50],
    'Reasoning': [
        "Bayern unbeaten in 18 Bundesliga games; Union Berlin struggle at home vs top sides.",
        "Lille solid away (W4/5); Auxerre's games low-scoring (Under 2.5 in 6/6 homes).",
        "Barca flying post-injuries; Sociedad lost 3 straight home league games.",
        "Milan on 5-game win streak; Juventus winless in 4 aways, leaky defense.",
        "Sporting top of league (W6/7); Santa Clara just 1W all season, expect goals."
    ]
}

# Create DataFrame
df = pd.DataFrame(bets_data)

# Display table
st.subheader("Acca Legs")
st.dataframe(df, use_container_width=True)

# Calculate combined odds
combined_odds = np.prod(df['Decimal Odds'])
st.metric("Combined Odds", f"{combined_odds:.2f}")

# Potential returns chart
st.subheader("Potential Returns")
stake = st.slider("Enter Stake (£)", min_value=1.0, max_value=100.0, value=10.0, step=0.5)
potential_return = stake * combined_odds
st.metric("Potential Return (incl. stake)", f"£{potential_return:.2f}")

# Simple bar chart for individual leg contributions (odds)
st.subheader("Leg Odds Breakdown")
chart_data = pd.DataFrame({
    'Leg': range(1, 6),
    'Odds': df['Decimal Odds']
})
st.bar_chart(chart_data.set_index('Leg'))

# Footer
st.markdown("---")
st.write("Update daily for fresh picks! Follow for more tips. ⚽ *Gamble responsibly. 18+.*")
