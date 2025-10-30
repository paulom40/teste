import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Soccer Odds & Matches",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .match-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .odds-card {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
        text-align: center;
    }
    .team-badge {
        width: 30px;
        height: 30px;
        margin-right: 10px;
    }
    .stat-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

class FootballDataAPI:
    def __init__(self):
        self.api_key = st.secrets.get("FOOTBALL_DATA_API_KEY", "your-free-api-key-here")
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {'X-Auth-Token': self.api_key}
    
    def get_competitions(self):
        """Get available competitions"""
        url = f"{self.base_url}/competitions"
        params = {
            'plan': 'TIER_ONE'  # Free tier only shows tier one competitions
        }
        try:
            response = requests.get(url, params=params, headers=self.headers)
            if response.status_code == 200:
                return response.json()['competitions']
            else:
                st.error(f"API Error: {response.status_code} - {response.json().get('message', 'Unknown error')}")
                return []
        except Exception as e:
            st.error(f"Error fetching competitions: {str(e)}")
            return []
    
    def get_matches(self, competition_code, date_from=None, date_to=None):
        """Get matches for a specific competition"""
        url = f"{self.base_url}/competitions/{competition_code}/matches"
        params = {}
        
        if date_from:
            params['dateFrom'] = date_from
        if date_to:
            params['dateTo'] = date_to
            
        try:
            response = requests.get(url, params=params, headers=self.headers)
            if response.status_code == 200:
                return response.json()['matches']
            else:
                st.error(f"API Error: {response.status_code}")
                return []
        except Exception as e:
            st.error(f"Error fetching matches: {str(e)}")
            return []
    
    def get_standings(self, competition_code):
        """Get standings for a competition"""
        url = f"{self.base_url}/competitions/{competition_code}/standings"
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Error fetching standings: {str(e)}")
            return None
    
    def get_team_matches(self, team_id):
        """Get matches for a specific team"""
        url = f"{self.base_url}/teams/{team_id}/matches"
        params = {
            'limit': 10,
            'status': 'FINISHED'
        }
        try:
            response = requests.get(url, params=params, headers=self.headers)
            if response.status_code == 200:
                return response.json()['matches']
            return []
        except Exception as e:
            st.error(f"Error fetching team matches: {str(e)}")
            return []

def format_match_status(status):
    """Format match status for display"""
    status_map = {
        'SCHEDULED': '🟢 Scheduled',
        'LIVE': '🔴 Live',
        'IN_PLAY': '🔴 Live',
        'PAUSED': '🟡 Half Time',
        'FINISHED': '⚫ Finished',
        'POSTPONED': '🟠 Postponed',
        'SUSPENDED': '🟠 Suspended',
        'CANCELLED': '🔴 Cancelled'
    }
    return status_map.get(status, status)

def calculate_win_probability(odds):
    """Calculate implied probability from odds"""
    if odds and odds > 0:
        return (1 / odds) * 100
    return 0

def main():
    st.markdown('<h1 class="main-header">⚽ Football Data & Odds Analyzer</h1>', unsafe_allow_html=True)
    
    # Initialize API
    api = FootballDataAPI()
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # API key input
    api_key = st.sidebar.text_input(
        "Football-Data.org API Key", 
        value=api.api_key, 
        type="password",
        help="Get free API key from https://www.football-data.org/"
    )
    if api_key != api.api_key:
        api.api_key = api_key
        api.headers = {'X-Auth-Token': api_key}
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Live Matches", "🏆 Standings", "📊 Team Analysis", "ℹ️ Guide"])
    
    with tab1:
        st.header("📅 Live & Upcoming Matches")
        
        if not api.api_key or api.api_key == "your-free-api-key-here":
            st.warning("⚠️ Please enter your Football-Data.org API key in the sidebar")
            show_api_guide()
            return
        
        try:
            # Get competitions
            with st.spinner("Loading competitions..."):
                competitions = api.get_competitions()
            
            if not competitions:
                st.error("No competitions found. Please check your API key.")
                return
            
            # Competition selection
            comp_names = [f"{comp['name']} ({comp['code']})" for comp in competitions]
            comp_codes = [comp['code'] for comp in competitions]
            
            comp_dict = dict(zip(comp_names, comp_codes))
            
            selected_comp_name = st.selectbox(
                "Select Competition",
                options=comp_names,
                index=0
            )
            selected_comp_code = comp_dict[selected_comp_name]
            
            # Date range
            col1, col2 = st.columns(2)
            with col1:
                date_from = st.date_input("From Date", datetime.now().date())
            with col2:
                date_to = st.date_input("To Date", datetime.now().date() + timedelta(days=7))
            
            if st.button("Load Matches", type="primary"):
                with st.spinner("Fetching matches..."):
                    matches = api.get_matches(
                        selected_comp_code,
                        date_from.strftime('%Y-%m-%d'),
                        date_to.strftime('%Y-%m-%d')
                    )
                
                if not matches:
                    st.info("No matches found for the selected period.")
                    return
                
                display_matches(matches)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with tab2:
        st.header("🏆 Competition Standings")
        
        if api.api_key and api.api_key != "your-free-api-key-here":
            try:
                selected_comp_name_standings = st.selectbox(
                    "Select Competition for Standings",
                    options=comp_names,
                    key="standings_comp"
                )
                selected_comp_code_standings = comp_dict[selected_comp_name_standings]
                
                if st.button("Load Standings", key="load_standings"):
                    with st.spinner("Loading standings..."):
                        standings_data = api.get_standings(selected_comp_code_standings)
                    
                    if standings_data and 'standings' in standings_data:
                        display_standings(standings_data)
                    else:
                        st.error("No standings data available for this competition.")
            except Exception as e:
                st.error(f"Error loading standings: {str(e)}")
        else:
            st.warning("Please enter API key to view standings")
    
    with tab3:
        st.header("📊 Team Analysis")
        
        if api.api_key and api.api_key != "your-free-api-key-here":
            try:
                # For team analysis, we'd need to get teams first
                # This is a simplified version
                st.info("Team analysis feature - Select a competition first to see team data")
                
                if 'matches' in st.session_state:
                    teams = set()
                    for match in st.session_state.matches:
                        teams.add((match['homeTeam']['id'], match['homeTeam']['name']))
                        teams.add((match['awayTeam']['id'], match['awayTeam']['name']))
                    
                    team_list = [f"{name} ({id})" for id, name in teams]
                    selected_team = st.selectbox("Select Team", options=team_list)
                    
                    if st.button("Analyze Team"):
                        team_id = selected_team.split('(')[-1].replace(')', '')
                        team_matches = api.get_team_matches(team_id)
                        display_team_analysis(team_matches, selected_team)
                        
            except Exception as e:
                st.error(f"Error in team analysis: {str(e)}")
        else:
            st.warning("Please enter API key to view team analysis")
    
    with tab4:
        show_guide()

def display_matches(matches):
    """Display matches in a structured format"""
    
    # Group matches by status
    scheduled_matches = [m for m in matches if m['status'] in ['SCHEDULED', 'TIMED']]
    live_matches = [m for m in matches if m['status'] in ['LIVE', 'IN_PLAY', 'PAUSED']]
    finished_matches = [m for m in matches if m['status'] == 'FINISHED']
    
    if live_matches:
        st.subheader("🔴 Live Matches")
        for match in live_matches:
            display_match_card(match, True)
    
    if scheduled_matches:
        st.subheader("🟢 Upcoming Matches")
        for match in scheduled_matches:
            display_match_card(match, False)
    
    if finished_matches:
        st.subheader("⚫ Finished Matches")
        for match in finished_matches[-10:]:  # Show last 10 finished matches
            display_match_card(match, False)

def display_match_card(match, is_live=False):
    """Display individual match card"""
    
    home_team = match['homeTeam']['name']
    away_team = match['awayTeam']['name']
    status = format_match_status(match['status'])
    utc_date = match['utcDate']
    
    # Convert UTC to local time
    try:
        match_time = datetime.fromisoformat(utc_date.replace('Z', '+00:00'))
        formatted_time = match_time.strftime("%Y-%m-%d %H:%M")
    except:
        formatted_time = utc_date
    
    with st.container():
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.write(f"**{home_team}** vs **{away_team}**")
            st.caption(f"🕒 {formatted_time}")
            st.caption(f"**Status:** {status}")
            
            if match.get('score'):
                score = match['score']
                if score['winner']:
                    winner = "Home" if score['winner'] == 'HOME_TEAM' else "Away" if score['winner'] == 'AWAY_TEAM' else "Draw"
                    st.success(f"🏆 Winner: {winner}")
        
        with col2:
            if match.get('score') and match['score'].get('fullTime'):
                ft_score = match['score']['fullTime']
                if ft_score['home'] is not None and ft_score['away'] is not None:
                    st.metric(
                        "Full Time",
                        f"{ft_score['home']} - {ft_score['away']}",
                        delta="LIVE" if is_live else "FINISHED"
                    )
        
        with col3:
            if match.get('score') and match['score'].get('halfTime'):
                ht_score = match['score']['halfTime']
                if ht_score['home'] is not None and ht_score['away'] is not None:
                    st.metric(
                        "Half Time",
                        f"{ht_score['home']} - {ht_score['away']}",
                    )
        
        with col4:
            if is_live and match.get('score') and match['score'].get('duration'):
                st.info(f"⏱️ {match['score']['duration']}")
            
            # Show match day and stage if available
            if match.get('matchday'):
                st.caption(f"Matchday: {match['matchday']}")
        
        st.markdown("---")

def display_standings(standings_data):
    """Display competition standings"""
    
    competition = standings_data['competition']['name']
    st.subheader(f"🏆 {competition} Standings")
    
    for standing in standings_data['standings']:
        if standing['type'] == 'TOTAL':  # Main league table
            table_data = []
            for team in standing['table']:
                table_data.append({
                    'Position': team['position'],
                    'Team': team['team']['name'],
                    'Played': team['playedGames'],
                    'Won': team['won'],
                    'Drawn': team['draw'],
                    'Lost': team['lost'],
                    'GF': team['goalsFor'],
                    'GA': team['goalsAgainst'],
                    'GD': team['goalDifference'],
                    'Points': team['points'],
                    'Form': team.get('form', '')
                })
            
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
            
            # Add some visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Points distribution
                fig_points = px.bar(df.head(10), x='Team', y='Points', 
                                  title='Top 10 Teams - Points')
                st.plotly_chart(fig_points, use_container_width=True)
            
            with col2:
                # Goals difference
                fig_gd = px.bar(df, x='Team', y='GD', 
                              title='Goal Difference by Team')
                st.plotly_chart(fig_gd, use_container_width=True)

def display_team_analysis(team_matches, team_name):
    """Display team performance analysis"""
    
    st.subheader(f"📊 Analysis for {team_name}")
    
    if not team_matches:
        st.info("No recent match data available for this team.")
        return
    
    # Calculate team statistics
    wins = 0
    draws = 0
    losses = 0
    goals_for = 0
    goals_against = 0
    
    for match in team_matches:
        if match['status'] == 'FINISHED':
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            score = match['score']['fullTime']
            
            is_home = team_name.split('(')[0].strip() in home_team
            
            if is_home:
                goals_for += score['home'] or 0
                goals_against += score['away'] or 0
                if score['home'] > score['away']:
                    wins += 1
                elif score['home'] == score['away']:
                    draws += 1
                else:
                    losses += 1
            else:
                goals_for += score['away'] or 0
                goals_against += score['home'] or 0
                if score['away'] > score['home']:
                    wins += 1
                elif score['away'] == score['home']:
                    draws += 1
                else:
                    losses += 1
    
    total_matches = wins + draws + losses
    
    # Display stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Matches", total_matches)
    with col2:
        st.metric("Wins", wins)
    with col3:
        st.metric("Draws", draws)
    with col4:
        st.metric("Losses", losses)
    
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Goals For", goals_for)
    with col6:
        st.metric("Goals Against", goals_against)
    
    # Win rate
    if total_matches > 0:
        win_rate = (wins / total_matches) * 100
        st.metric("Win Rate", f"{win_rate:.1f}%")

def show_guide():
    """Display user guide"""
    
    st.header("📖 How to Use This App")
    
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Get API Key**: 
       - Go to [Football-Data.org](https://www.football-data.org/)
       - Register for a free account
       - Get your API key from the client area
    
    2. **Enter API Key**: 
       - Input your key in the sidebar
       - Free tier gives you 10 requests per minute
    
    3. **Explore Features**:
       - **Live Matches**: View upcoming and live matches
       - **Standings**: See league tables and statistics
       - **Team Analysis**: Analyze team performance
    
    ### 📊 Available Data
    
    **Free Tier Includes**:
    - All major European leagues (Premier League, La Liga, etc.)
    - Live scores and match details
    - League standings
    - Team information
    - Match statistics
    
    ### ⚠️ Important Notes
    
    - Free tier has rate limits (10 requests per minute)
    - Data updates in real-time
    - Some advanced features require premium access
    - Always check API status if data isn't loading
    """)
    
    st.info("""
    **💡 Pro Tip**: The free tier is perfect for personal use and small projects. 
    For commercial applications, consider upgrading to a paid plan.
    """)

def show_api_guide():
    """Show API guide when no key is entered"""
    
    st.info("""
    **How to get your free API key:**
    
    1. Visit [Football-Data.org](https://www.football-data.org/)
    2. Click "Sign Up" and create a free account
    3. Go to "Client Area" after logging in
    4. Find your API key in the account section
    5. Copy and paste it in the sidebar
    
    **Free Tier Limits:**
    - 10 requests per minute
    - All major competitions available
    - Perfect for personal use and testing
    """)

if __name__ == "__main__":
    main()
