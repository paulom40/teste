import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import re

# Configure the page
st.set_page_config(
    page_title="CornerProBet Live Stats",
    page_icon="⚽",
    layout="wide"
)

def debug_website_structure(url):
    """
    Debug function to understand website structure
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all potential game containers
        potential_containers = []
        
        # Common container classes for sports websites
        common_patterns = [
            'match', 'game', 'event', 'fixture', 'row', 'item',
            'live', 'inplay', 'container', 'card', 'table-row'
        ]
        
        for pattern in common_patterns:
            containers = soup.find_all(class_=re.compile(pattern, re.IGNORECASE))
            for container in containers:
                text = container.get_text(strip=True)
                if len(text) > 20 and any(word in text.lower() for word in ['corner', 'shot', 'target', 'goal']):
                    potential_containers.append({
                        'element': container.name,
                        'class': container.get('class', []),
                        'text': text[:100] + '...' if len(text) > 100 else text
                    })
        
        return potential_containers[:10]  # Return first 10 potential containers
        
    except Exception as e:
        return f"Error: {e}"

def scrape_cornerprobet_advanced():
    """
    Advanced scraping with multiple selector strategies
    """
    try:
        url = "https://cornerprobet.com/pt"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'pt-PT,pt;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        games_data = []
        
        # Strategy 1: Look for table rows
        table_rows = soup.find_all('tr')
        for row in table_rows:
            game_data = extract_from_table_row(row)
            if game_data and game_data.get('home_team'):
                games_data.append(game_data)
        
        # Strategy 2: Look for div containers with match data
        if not games_data:
            match_divs = soup.find_all('div', class_=True)
            for div in match_divs:
                classes = ' '.join(div.get('class', [])).lower()
                if any(keyword in classes for keyword in ['match', 'game', 'event', 'live']):
                    game_data = extract_from_div_container(div)
                    if game_data and game_data.get('home_team'):
                        games_data.append(game_data)
        
        # Strategy 3: Look for specific data attributes
        if not games_data:
            elements_with_data = soup.find_all(attrs={"data-type": True})
            for elem in elements_with_data:
                game_data = extract_from_data_attributes(elem)
                if game_data and game_data.get('home_team'):
                    games_data.append(game_data)
        
        return games_data[:15]  # Limit to 15 games
        
    except Exception as e:
        st.error(f"Scraping error: {e}")
        return []

def extract_from_table_row(row):
    """Extract game data from table row"""
    cells = row.find_all(['td', 'th'])
    if len(cells) < 3:
        return None
    
    game_data = {}
    full_text = row.get_text()
    
    # Try to extract teams (usually in first few cells)
    for i, cell in enumerate(cells[:3]):
        text = cell.get_text(strip=True)
        if text and len(text) > 2:
            if 'home_team' not in game_data:
                game_data['home_team'] = text
            else:
                game_data['away_team'] = text
                break
    
    # Extract score
    score_match = re.search(r'(\d+)[:\-](\d+)', full_text)
    if score_match:
        game_data['score'] = f"{score_match.group(1)}-{score_match.group(2)}"
    
    # Extract corners
    corner_match = re.search(r'(\d+)\s*[-\s]?\s*(\d+)\s*(?:corner|canto)', full_text, re.IGNORECASE)
    if corner_match:
        game_data['corners'] = f"{corner_match.group(1)}-{corner_match.group(2)}"
    
    # Extract shots on target
    shots_match = re.search(r'(\d+)\s*[-\s]?\s*(\d+)\s*(?:shot|chute|finaliza)', full_text, re.IGNORECASE)
    if shots_match:
        game_data['shots_on_target'] = f"{shots_match.group(1)}-{shots_match.group(2)}"
    
    game_data['last_updated'] = datetime.now().strftime("%H:%M:%S")
    
    return game_data

def extract_from_div_container(div):
    """Extract game data from div container"""
    game_data = {}
    full_text = div.get_text()
    
    # Look for team patterns (Team A vs Team B or Team A - Team B)
    team_pattern = r'([A-Za-z0-9\s\.]+)\s*(?:vs| versus|-|×)\s*([A-Za-z0-9\s\.]+)'
    team_match = re.search(team_pattern, full_text, re.IGNORECASE)
    
    if team_match:
        game_data['home_team'] = team_match.group(1).strip()
        game_data['away_team'] = team_match.group(2).strip()
    
    # Extract numeric data
    score_match = re.search(r'(\d+)[:\-](\d+)', full_text)
    if score_match:
        game_data['score'] = f"{score_match.group(1)}-{score_match.group(2)}"
    
    corner_match = re.search(r'(\d+)\s*[-\s]?\s*(\d+)\s*c', full_text)
    if corner_match:
        game_data['corners'] = f"{corner_match.group(1)}-{corner_match.group(2)}"
    
    game_data['last_updated'] = datetime.now().strftime("%H:%M:%S")
    
    return game_data

def extract_from_data_attributes(elem):
    """Extract game data from elements with data attributes"""
    # This would need to be customized based on the actual data attributes used
    game_data = {}
    
    # Example: if data-team-home and data-team-away attributes exist
    home_team = elem.get('data-team-home') or elem.get('data-home-team')
    away_team = elem.get('data-team-away') or elem.get('data-away-team')
    
    if home_team and away_team:
        game_data['home_team'] = home_team
        game_data['away_team'] = away_team
        
        # Extract other data attributes
        corners_home = elem.get('data-corners-home')
        corners_away = elem.get('data-corners-away')
        if corners_home and corners_away:
            game_data['corners'] = f"{corners_home}-{corners_away}"
        
        shots_home = elem.get('data-shots-home')
        shots_away = elem.get('data-shots-away')
        if shots_home and shots_away:
            game_data['shots_on_target'] = f"{shots_home}-{shots_away}"
        
        game_data['last_updated'] = datetime.now().strftime("%H:%M:%S")
        
        return game_data
    
    return None

def display_games_table(games_data):
    """Display games data in a table format"""
    if not games_data:
        st.warning("No live games data found. The website structure might have changed.")
        return
    
    st.header("⚽ Live Soccer Games - CornerProBet")
    
    # Create DataFrame for better display
    df_data = []
    for game in games_data:
        df_data.append({
            'Home Team': game.get('home_team', 'N/A'),
            'Away Team': game.get('away_team', 'N/A'),
            'Score': game.get('score', '0-0'),
            'Corners': game.get('corners', 'N/A'),
            'Shots on Target': game.get('shots_on_target', 'N/A'),
            'Last Updated': game.get('last_updated', 'N/A')
        })
    
    if df_data:
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        # Show raw data for debugging
        with st.expander("Show Raw Data"):
            st.json(games_data)
    else:
        st.error("No valid game data could be extracted")

def main():
    st.title("🔍 CornerProBet Live Tracker")
    st.markdown("Live soccer statistics from CornerProBet.com")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 30 seconds", value=False)
    refresh_btn = st.sidebar.button("Refresh Data")
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    if debug_mode:
        st.sidebar.subheader("Debug Tools")
        if st.sidebar.button("Analyze Website Structure"):
            with st.spinner("Analyzing website structure..."):
                debug_info = debug_website_structure("https://cornerprobet.com/pt")
                st.sidebar.write("Potential game containers found:")
                st.sidebar.json(debug_info)
    
    # Main content area
    data_placeholder = st.empty()
    
    if refresh_btn or auto_refresh:
        with st.spinner("Fetching live game data from CornerProBet..."):
            games_data = scrape_cornerprobet_advanced()
            
        with data_placeholder.container():
            display_games_table(games_data)
    
    # Initial load
    if not auto_refresh and not refresh_btn:
        st.info("Click 'Refresh Data' to load live games information")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Data Extracted:**\n"
        "• Team names\n"
        "• Current score\n"
        "• Corner kicks\n"
        "• Shots on target\n"
        "• Last update time"
    )

if __name__ == "__main__":
    main()
