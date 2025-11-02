import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import json

# Configure the page
st.set_page_config(
    page_title="Live Soccer Stats",
    page_icon="⚽",
    layout="wide"
)

def scrape_cornerprobet():
    """
    Scrape live soccer game data from CornerProBet
    """
    try:
        url = "https://cornerprobet.com/pt"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        games_data = []
        
        # Look for game containers - these selectors might need adjustment
        game_containers = soup.find_all('div', class_=lambda x: x and 'game' in x.lower()) or \
                         soup.find_all('div', class_=lambda x: x and 'match' in x.lower()) or \
                         soup.find_all('tr', class_=lambda x: x and 'match' in x.lower())
        
        if not game_containers:
            # Alternative approach: look for tables or specific patterns
            game_containers = soup.find_all('tr')[1:]  # Skip header row if it's a table
            
        for container in game_containers[:10]:  # Limit to first 10 games
            try:
                game_info = extract_game_info(container)
                if game_info:
                    games_data.append(game_info)
            except Exception as e:
                continue
                
        return games_data
        
    except requests.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return []
    except Exception as e:
        st.error(f"Error parsing data: {e}")
        return []

def extract_game_info(container):
    """
    Extract game information from a container element
    """
    try:
        # These selectors will need to be adjusted based on the actual website structure
        game_data = {}
        
        # Try to find team names
        teams = container.find_all('span', class_=lambda x: x and 'team' in x.lower()) or \
                container.find_all('td', class_=lambda x: x and 'team' in x.lower())
        
        if len(teams) >= 2:
            game_data['home_team'] = teams[0].get_text(strip=True)
            game_data['away_team'] = teams[1].get_text(strip=True)
        else:
            # Alternative approach: look for text patterns
            text = container.get_text()
            if ' - ' in text:
                parts = text.split(' - ')
                if len(parts) >= 2:
                    game_data['home_team'] = parts[0].strip()
                    game_data['away_team'] = parts[1].split()[0].strip()
        
        # Try to find score
        score_elements = container.find_all('span', class_=lambda x: x and 'score' in x.lower()) or \
                        container.find_all('b') or \
                        container.find_all('strong')
        
        for elem in score_elements:
            text = elem.get_text(strip=True)
            if ':' in text or '-' in text:
                game_data['score'] = text
                break
        
        # Try to find corners
        corners_text = container.get_text()
        if 'corner' in corners_text.lower():
            # Look for corner counts in the text
            import re
            corner_pattern = r'(\d+)\s*[-\s]?\s*(\d+)\s*corners?'
            corners_match = re.search(corner_pattern, corners_text, re.IGNORECASE)
            if corners_match:
                game_data['corners'] = f"{corners_match.group(1)}-{corners_match.group(2)}"
        
        # Try to find shots on target
        shots_text = container.get_text()
        if 'shot' in shots_text.lower() or 'target' in shots_text.lower():
            import re
            shots_pattern = r'(\d+)\s*[-\s]?\s*(\d+)\s*(?:shots?|target)'
            shots_match = re.search(shots_pattern, shots_text, re.IGNORECASE)
            if shots_match:
                game_data['shots_on_target'] = f"{shots_match.group(1)}-{shots_match.group(2)}"
        
        # Get current time
        game_data['last_updated'] = datetime.now().strftime("%H:%M:%S")
        
        return game_data if game_data else None
        
    except Exception as e:
        return None

def display_games_data(games_data):
    """
    Display the scraped games data in a nice format
    """
    if not games_data:
        st.warning("No live games data found. The website structure might have changed.")
        return
    
    st.header("⚽ Live Soccer Games Statistics")
    
    for i, game in enumerate(games_data):
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.subheader(game.get('home_team', 'Home Team'))
            
            with col2:
                score = game.get('score', '0-0')
                st.metric("Score", score)
                
                corners = game.get('corners', 'N/A')
                st.metric("Corners", corners)
                
                shots = game.get('shots_on_target', 'N/A')
                st.metric("Shots on Target", shots)
            
            with col3:
                st.subheader(game.get('away_team', 'Away Team'))
            
            st.markdown("---")

def main():
    st.title("🔍 Live Soccer Games Tracker")
    st.markdown("Live statistics from CornerProBet")
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 30 seconds", value=False)
    refresh_btn = st.sidebar.button("Refresh Data")
    
    # Placeholder for data
    data_placeholder = st.empty()
    
    if refresh_btn or auto_refresh:
        with st.spinner("Fetching live game data..."):
            games_data = scrape_cornerprobet()
            
        with data_placeholder.container():
            display_games_data(games_data)
    
    # Initial load
    if not auto_refresh and not refresh_btn:
        st.info("Click 'Refresh Data' to load live games information")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Add some information about the data
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This app scrapes live soccer game data including:\n"
        "• Current score\n"
        "• Corner kicks\n"
        "• Shots on target\n"
        "• Team names"
    )

if __name__ == "__main__":
    main()
