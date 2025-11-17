import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Configure the page
st.set_page_config(
    page_title="Sports Betting Odds Tracker",
    page_icon="⚽",
    layout="wide"
)

# API configuration
API_KEY = "2fc8ca1227c5f69b90c485199c8eabee"  # Replace with your actual API key
BASE_URL = "https://api.the-odds-api.com/v4/sports"

# League mappings
LEAGUES = {
    'Portugal': {
        'Primeira Liga': 'soccer_portugal_primeira_liga',
        'Segunda Liga': 'soccer_portugal_segunda_liga'
    },
    'Spain': {
        'La Liga': 'soccer_spain_la_liga',
        'Segunda Division': 'soccer_spain_segunda_division'
    },
    'Italy': {
        'Serie A': 'soccer_italy_serie_a',
        'Serie B': 'soccer_italy_serie_b'
    },
    'England': {
        'Premier League': 'soccer_epl',
        'Championship': 'soccer_england_championship'
    },
    'Germany': {
        'Bundesliga': 'soccer_germany_bundesliga',
        'Bundesliga 2': 'soccer_germany_bundesliga2'
    }
}

BOOKMAKERS = ['pinnacle', 'bet365']

def get_odds(sport_key, regions='eu', markets='h2h'):
    """Fetch odds from the API"""
    url = f"{BASE_URL}/{sport_key}/odds"
    params = {
        'api_key': API_KEY,
        'regions': regions,
        'markets': markets,
        'bookmakers': ','.join(BOOKMAKERS)
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_value_bets(home_odds, away_odds, draw_odds=None):
    """Calculate implied probabilities and value indicators"""
    if draw_odds:
        # For sports with draw possibility
        home_implied = 1 / home_odds if home_odds else 0
        away_implied = 1 / away_odds if away_odds else 0
        draw_implied = 1 / draw_odds if draw_odds else 0
        total_implied = home_implied + away_implied + draw_implied
        home_value = (1/home_odds - home_implied/total_implied) * 100 if home_odds else 0
        away_value = (1/away_odds - away_implied/total_implied) * 100 if away_odds else 0
        draw_value = (1/draw_odds - draw_implied/total_implied) * 100 if draw_odds else 0
        return home_value, away_value, draw_value
    else:
        # For sports without draw
        home_implied = 1 / home_odds if home_odds else 0
        away_implied = 1 / away_odds if away_odds else 0
        total_implied = home_implied + away_implied
        home_value = (1/home_odds - home_implied/total_implied) * 100 if home_odds else 0
        away_value = (1/away_odds - away_implied/total_implied) * 100 if away_odds else 0
        return home_value, away_value

def create_odds_dataframe(odds_data):
    """Create a formatted DataFrame from odds data"""
    matches = []
    
    for match in odds_data:
        home_team = match['home_team']
        away_team = match['away_team']
        commence_time = match['commence_time']
        
        pinnacle_odds = None
        bet365_odds = None
        
        # Extract odds from different bookmakers
        for bookmaker in match['bookmakers']:
            if bookmaker['key'] == 'pinnacle':
                pinnacle_odds = bookmaker['markets'][0]['outcomes']
            elif bookmaker['key'] == 'bet365':
                bet365_odds = bookmaker['markets'][0]['outcomes']
        
        # Create match entry
        match_data = {
            'Match': f"{home_team} vs {away_team}",
            'Date': pd.to_datetime(commence_time).strftime('%Y-%m-%d %H:%M'),
            'Home Team': home_team,
            'Away Team': away_team
        }
        
        # Add Pinnacle odds
        if pinnacle_odds:
            for outcome in pinnacle_odds:
                if outcome['name'] == home_team:
                    match_data['Pinnacle Home'] = outcome['price']
                elif outcome['name'] == away_team:
                    match_data['Pinnacle Away'] = outcome['price']
                else:
                    match_data['Pinnacle Draw'] = outcome['price']
        
        # Add Bet365 odds
        if bet365_odds:
            for outcome in bet365_odds:
                if outcome['name'] == home_team:
                    match_data['Bet365 Home'] = outcome['price']
                elif outcome['name'] == away_team:
                    match_data['Bet365 Away'] = outcome['price']
                else:
                    match_data['Bet365 Draw'] = outcome['price']
        
        # Calculate value bets if we have both bookmakers
        if pinnacle_odds and bet365_odds:
            home_pinnacle = match_data.get('Pinnacle Home')
            away_pinnacle = match_data.get('Pinnacle Away')
            draw_pinnacle = match_data.get('Pinnacle Draw')
            
            home_bet365 = match_data.get('Bet365 Home')
            away_bet365 = match_data.get('Bet365 Away')
            draw_bet365 = match_data.get('Bet365 Draw')
            
            if draw_pinnacle:
                pinnacle_values = calculate_value_bets(home_pinnacle, away_pinnacle, draw_pinnacle)
                bet365_values = calculate_value_bets(home_bet365, away_bet365, draw_bet365)
                
                match_data['Pinnacle Home Value'] = f"{pinnacle_values[0]:.1f}%"
                match_data['Pinnacle Away Value'] = f"{pinnacle_values[1]:.1f}%"
                match_data['Pinnacle Draw Value'] = f"{pinnacle_values[2]:.1f}%"
                
                match_data['Bet365 Home Value'] = f"{bet365_values[0]:.1f}%"
                match_data['Bet365 Away Value'] = f"{bet365_values[1]:.1f}%"
                match_data['Bet365 Draw Value'] = f"{bet365_values[2]:.1f}%"
            else:
                pinnacle_values = calculate_value_bets(home_pinnacle, away_pinnacle)
                bet365_values = calculate_value_bets(home_bet365, away_bet365)
                
                match_data['Pinnacle Home Value'] = f"{pinnacle_values[0]:.1f}%"
                match_data['Pinnacle Away Value'] = f"{pinnacle_values[1]:.1f}%"
                
                match_data['Bet365 Home Value'] = f"{bet365_values[0]:.1f}%"
                match_data['Bet365 Away Value'] = f"{bet365_values[1]:.1f}%"
        
        matches.append(match_data)
    
    return pd.DataFrame(matches)

def create_pnl_tracker():
    """Create a P&L tracker for the season"""
    st.subheader("💰 Season P&L Tracker")
    
    # Initialize session state for P&L data
    if 'pnl_data' not in st.session_state:
        st.session_state.pnl_data = pd.DataFrame({
            'Date': [],
            'League': [],
            'Match': [],
            'Bet Type': [],
            'Stake': [],
            'Odds': [],
            'Result': [],  # Win/Loss
            'P/L': []
        })
    
    # Add new bet form
    with st.expander("Add New Bet"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bet_date = st.date_input("Bet Date")
            league = st.selectbox("League", list(LEAGUES.keys()))
            match = st.text_input("Match")
        
        with col2:
            bet_type = st.selectbox("Bet Type", ["Home Win", "Away Win", "Draw", "Over", "Under"])
            stake = st.number_input("Stake (€)", min_value=1.0, value=10.0)
            odds = st.number_input("Odds", min_value=1.01, value=2.0, step=0.01)
        
        with col3:
            result = st.selectbox("Result", ["Pending", "Win", "Loss"])
            if st.button("Add Bet"):
                if match:
                    # Calculate P/L
                    if result == "Win":
                        pl = (stake * odds) - stake
                    elif result == "Loss":
                        pl = -stake
                    else:
                        pl = 0
                    
                    # Add to dataframe
                    new_bet = pd.DataFrame({
                        'Date': [bet_date.strftime('%Y-%m-%d')],
                        'League': [league],
                        'Match': [match],
                        'Bet Type': [bet_type],
                        'Stake': [stake],
                        'Odds': [odds],
                        'Result': [result],
                        'P/L': [pl]
                    })
                    
                    st.session_state.pnl_data = pd.concat([st.session_state.pnl_data, new_bet], ignore_index=True)
                    st.success("Bet added successfully!")
    
    # Display P&L data
    if not st.session_state.pnl_data.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        total_pl = st.session_state.pnl_data['P/L'].sum()
        total_stake = st.session_state.pnl_data['Stake'].sum()
        roi = (total_pl / total_stake * 100) if total_stake > 0 else 0
        
        with col1:
            st.metric("Total P/L", f"€{total_pl:.2f}")
        with col2:
            st.metric("Total Stake", f"€{total_stake:.2f}")
        with col3:
            st.metric("ROI", f"{roi:.1f}%")
        with col4:
            wins = len(st.session_state.pnl_data[st.session_state.pnl_data['Result'] == 'Win'])
            total_bets = len(st.session_state.pnl_data[st.session_state.pnl_data['Result'] != 'Pending'])
            win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Display detailed table
        st.dataframe(st.session_state.pnl_data, use_container_width=True)
        
        # Download button
        csv = st.session_state.pnl_data.to_csv(index=False)
        st.download_button(
            label="Download P&L Data",
            data=csv,
            file_name=f"betting_pnl_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No bets recorded yet. Add your first bet above.")

# Main app
def main():
    st.title("⚽ European Football Odds Tracker")
    st.markdown("Track odds from Pinnacle and Bet365 across top European leagues")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    selected_country = st.sidebar.selectbox("Select Country", list(LEAGUES.keys()))
    selected_league = st.sidebar.selectbox("Select League", list(LEAGUES[selected_country].keys()))
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Current Odds", "💰 P&L Tracker", "ℹ️ About"])
    
    with tab1:
        st.header(f"{selected_country} - {selected_league}")
        
        if st.button("Fetch Latest Odds"):
            with st.spinner("Fetching odds..."):
                sport_key = LEAGUES[selected_country][selected_league]
                odds_data = get_odds(sport_key)
                
                if odds_data:
                    df = create_odds_dataframe(odds_data)
                    
                    if not df.empty:
                        # Display summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Matches", len(df))
                        with col2:
                            pinnacle_matches = len(df[df['Pinnacle Home'].notna()])
                            st.metric("Pinnacle Markets", pinnacle_matches)
                        with col3:
                            bet365_matches = len(df[df['Bet365 Home'].notna()])
                            st.metric("Bet365 Markets", bet365_matches)
                        
                        # Display the odds table
                        st.dataframe(df, use_container_width=True)
                        
                        # Value bets highlight
                        st.subheader("🎯 Value Bet Opportunities")
                        value_columns = [col for col in df.columns if 'Value' in col]
                        if value_columns:
                            for col in value_columns:
                                try:
                                    df[col + '_num'] = df[col].str.rstrip('%').astype(float)
                                except:
                                    continue
                            
                            # Find best value bets
                            value_threshold = 5.0  # 5% value threshold
                            high_value_bets = []
                            
                            for idx, row in df.iterrows():
                                for col in value_columns:
                                    if '_num' in col:
                                        try:
                                            value = row[col]
                                            if value > value_threshold:
                                                bookmaker = 'Pinnacle' if 'Pinnacle' in col else 'Bet365'
                                                bet_type = 'Home' if 'Home' in col else 'Away' if 'Away' in col else 'Draw'
                                                high_value_bets.append({
                                                    'Match': row['Match'],
                                                    'Bookmaker': bookmaker,
                                                    'Bet Type': bet_type,
                                                    'Value %': f"{value:.1f}%"
                                                })
                                        except:
                                            continue
                            
                            if high_value_bets:
                                value_df = pd.DataFrame(high_value_bets)
                                st.dataframe(value_df, use_container_width=True)
                            else:
                                st.info("No high-value bets found above 5% threshold")
                    else:
                        st.warning("No odds data available for the selected league")
                else:
                    st.error("Failed to fetch odds data. Please check your API key and connection.")
    
    with tab2:
        create_pnl_tracker()
    
    with tab3:
        st.header("About This App")
        st.markdown("""
        This app provides:
        
        - **Real-time odds** from Pinnacle and Bet365
        - **Coverage of top European leagues**:
          - Portugal (Primeira & Segunda Liga)
          - Spain (La Liga & Segunda Division)
          - Italy (Serie A & Serie B)
          - England (Premier League & Championship)
          - Germany (Bundesliga & Bundesliga 2)
        
        - **Value bet calculations** based on implied probabilities
        - **Season-long P&L tracking** for your bets
        
        ### How to Use:
        1. Select your country and league in the sidebar
        2. Click "Fetch Latest Odds" to get current markets
        3. Use the P&L tracker to monitor your betting performance
        4. Look for value bets where the implied probability suggests positive expected value
        
        **Note:** Replace `YOUR_API_KEY_HERE` with your actual API key from The Odds API.
        """)

if __name__ == "__main__":
    main()
