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

def debug_table_structure(url):
    """Debug function to analyze table structure"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all tables
        tables = soup.find_all('table')
        table_info = []
        
        for i, table in enumerate(tables[:5]):  # Analyze first 5 tables
            table_data = {
                'table_number': i + 1,
                'rows_count': 0,
                'sample_rows': []
            }
            
            rows = table.find_all('tr')
            table_data['rows_count'] = len(rows)
            
            # Show first 3 rows as example
            for j, row in enumerate(rows[:3]):
                cells = row.find_all(['td', 'th'])
                cell_texts = [cell.get_text(strip=True) for cell in cells]
                table_data['sample_rows'].append({
                    'row_number': j,
                    'cells_count': len(cells),
                    'cell_contents': cell_texts
                })
            
            table_info.append(table_data)
                
        return table_info
        
    except Exception as e:
        return f"Error: {e}"

def extract_from_table_row(row):
    """Extract game data from table row - Enhanced version"""
    cells = row.find_all(['td', 'th'])
    if len(cells) < 2:
        return None
    
    game_data = {}
    full_text = row.get_text(strip=True)
    
    # Debug: Show what we're working with
    if st.session_state.get('debug', False):
        st.sidebar.write(f"Row text: {full_text[:200]}...")
        st.sidebar.write(f"Number of cells: {len(cells)}")
    
    # Strategy 1: Look for team names in specific cell positions
    # Common patterns: teams are usually in first few cells
    for i, cell in enumerate(cells[:4]):  # Check first 4 cells
        cell_text = cell.get_text(strip=True)
        
        # Skip if it's clearly not a team name
        if not cell_text or len(cell_text) < 2 or cell_text.isdigit():
            continue
            
        # Skip common non-team text
        skip_patterns = ['vs', 'live', 'min', "'", 'finished', 'ht', 'ft', 'result', 'corner', 'shot']
        if any(pattern in cell_text.lower() for pattern in skip_patterns):
            continue
        
        # Check if this looks like a team name (contains letters, possibly with numbers and spaces)
        if re.search(r'[a-zA-Z]', cell_text) and len(cell_text) > 1:
            if 'home_team' not in game_data:
                game_data['home_team'] = cell_text
            elif 'away_team' not in game_data:
                game_data['away_team'] = cell_text
                break
    
    # Strategy 2: If we didn't find teams, try to split by common separators
    if 'home_team' not in game_data:
        # Common separators: vs, -, ×, —
        separators = [' vs ', ' - ', ' × ', ' — ', ' VS ']
        for separator in separators:
            if separator in full_text:
                parts = full_text.split(separator)
                if len(parts) >= 2:
                    # Take first meaningful text as home team
                    home_candidate = parts[0].strip()
                    away_candidate = parts[1].split()[0].strip() if parts[1] else ''
                    
                    if home_candidate and len(home_candidate) > 2:
                        game_data['home_team'] = home_candidate
                    if away_candidate and len(away_candidate) > 2:
                        game_data['away_team'] = away_candidate
                    break
    
    # Extract score - look for patterns like "1-0", "2:1", etc.
    score_patterns = [
        r'(\d+)[:\-](\d+)',  # Basic score 1-0, 2:1
        r'(\d+)\s*-\s*(\d+)',  # Score with spaces "1 - 0"
        r'(\d+)\s*:\s*(\d+)',  # Score with colon "1 : 0"
    ]
    
    for pattern in score_patterns:
        score_match = re.search(pattern, full_text)
        if score_match:
            game_data['score'] = f"{score_match.group(1)}-{score_match.group(2)}"
            break
    else:
        # If no score found, set default
        game_data['score'] = '0-0'
    
    # Extract corners - multiple patterns
    corner_patterns = [
        r'(\d+)\s*[-\s]?\s*(\d+)\s*corners?',  # "5-3 corners"
        r'corners?\s*(\d+)\s*[-\s]?\s*(\d+)',  # "corners 5-3"
        r'c\s*(\d+)\s*[-\s]?\s*(\d+)',  # "c 5-3"
        r'(\d+)\s*[-\s]?\s*(\d+)\s*c',  # "5-3 c"
        r'corner\s*(\d+)\s*[-\s]?\s*(\d+)',  # "corner 5-3"
        r'corners?:?\s*(\d+)\s*[-\s]?\s*(\d+)',  # "corners: 5-3"
    ]
    
    corners_found = False
    for pattern in corner_patterns:
        corner_match = re.search(pattern, full_text, re.IGNORECASE)
        if corner_match:
            game_data['corners'] = f"{corner_match.group(1)}-{corner_match.group(2)}"
            corners_found = True
            break
    
    if not corners_found:
        game_data['corners'] = 'N/A'
    
    # Extract shots on target - multiple patterns
    shots_patterns = [
        r'(\d+)\s*[-\s]?\s*(\d+)\s*shots?',  # "5-3 shots"
        r'shots?\s*(\d+)\s*[-\s]?\s*(\d+)',  # "shots 5-3"
        r'(\d+)\s*[-\s]?\s*(\d+)\s*s\s*\(?o?n?\)?\s*t',  # "5-3 s ot"
        r'shots?\s*on\s*target\s*(\d+)\s*[-\s]?\s*(\d+)',  # "shots on target 5-3"
        r'(\d+)\s*[-\s]?\s*(\d+)\s*sog',  # "5-3 sog" (shots on goal)
        r'target\s*(\d+)\s*[-\s]?\s*(\d+)',  # "target 5-3"
        r'shots?:?\s*(\d+)\s*[-\s]?\s*(\d+)',  # "shots: 5-3"
    ]
    
    shots_found = False
    for pattern in shots_patterns:
        shots_match = re.search(pattern, full_text, re.IGNORECASE)
        if shots_match:
            game_data['shots_on_target'] = f"{shots_match.group(1)}-{shots_match.group(2)}"
            shots_found = True
            break
    
    if not shots_found:
        game_data['shots_on_target'] = 'N/A'
    
    # If we found at least one team, consider it a valid game
    if game_data.get('home_team') or game_data.get('away_team'):
        game_data['last_updated'] = datetime.now().strftime("%H:%M:%S")
        return game_data
    
    return None

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
    else:
        game_data['score'] = '0-0'
    
    corner_match = re.search(r'(\d+)\s*[-\s]?\s*(\d+)\s*c', full_text)
    if corner_match:
        game_data['corners'] = f"{corner_match.group(1)}-{corner_match.group(2)}"
    else:
        game_data['corners'] = 'N/A'
    
    game_data['last_updated'] = datetime.now().strftime("%H:%M:%S")
    
    return game_data

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
        
        # Strategy 1: Look for table rows (most common)
        table_rows = soup.find_all('tr')
        if st.session_state.get('debug', False):
            st.sidebar.write(f"Found {len(table_rows)} table rows")
        
        for row in table_rows:
            game_data = extract_from_table_row(row)
            if game_data and game_data.get('home_team'):
                games_data.append(game_data)
        
        # Strategy 2: Look for div containers with match data
        if not games_data:
            match_divs = soup.find_all('div', class_=True)
            if st.session_state.get('debug', False):
                st.sidebar.write(f"Found {len(match_divs)} div elements with classes")
            
            for div in match_divs:
                classes = ' '.join(div.get('class', [])).lower()
                if any(keyword in classes for keyword in ['match', 'game', 'event', 'live']):
                    game_data = extract_from_div_container(div)
                    if game_data and game_data.get('home_team'):
                        games_data.append(game_data)
        
        return games_data[:15]  # Limit to 15 games
        
    except Exception as e:
        st.error(f"Scraping error: {e}")
        return []

def display_games_table(games_data):
    """Display games data in a table format"""
    if not games_data:
        st.warning("No live games data found. The website structure might have changed.")
        
        # Show troubleshooting tips
        with st.expander("Troubleshooting Tips"):
            st.markdown("""
            1. **Check the website structure**: Enable Debug Mode and analyze the table structure
            2. **Verify website accessibility**: Ensure https://cornerprobet.com/pt is accessible
            3. **Inspect elements**: Right-click on live games and inspect the HTML structure
            4. **Update selectors**: The CSS classes or HTML structure might have changed
            """)
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
        
        # Style the dataframe
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Home Team": st.column_config.TextColumn(width="large"),
                "Away Team": st.column_config.TextColumn(width="large"),
                "Score": st.column_config.TextColumn(width="small"),
                "Corners": st.column_config.TextColumn(width="small"),
                "Shots on Target": st.column_config.TextColumn(width="small"),
            }
        )
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Games", len(games_data))
        with col2:
            games_with_corners = sum(1 for game in games_data if game.get('corners') != 'N/A')
            st.metric("Games with Corners Data", games_with_corners)
        with col3:
            games_with_shots = sum(1 for game in games_data if game.get('shots_on_target') != 'N/A')
            st.metric("Games with Shots Data", games_with_shots)
        
        # Show raw data for debugging
        if st.session_state.get('debug', False):
            with st.expander("Debug: Show Raw Extracted Data"):
                st.json(games_data)
    else:
        st.error("No valid game data could be extracted")

def main():
    st.title("🔍 CornerProBet Live Tracker")
    st.markdown("Live soccer statistics from CornerProBet.com")
    
    # Initialize session state for debug
    if 'debug' not in st.session_state:
        st.session_state.debug = False
    
    # Sidebar controls
    st.sidebar.header("Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 30 seconds", value=False)
    refresh_btn = st.sidebar.button("Refresh Data")
    
    # Debug mode
    st.session_state.debug = st.sidebar.checkbox("Debug Mode", value=False)
    
    if st.session_state.debug:
        st.sidebar.subheader("Debug Tools")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("Analyze Website"):
                with st.spinner("Analyzing website structure..."):
                    debug_info = debug_website_structure("https://cornerprobet.com/pt")
                    st.sidebar.write("Potential game containers found:")
                    st.sidebar.json(debug_info)
        
        with col2:
            if st.button("Analyze Tables"):
                with st.spinner("Analyzing table structure..."):
                    table_info = debug_table_structure("https://cornerprobet.com/pt")
                    st.sidebar.write("Table structure analysis:")
                    st.sidebar.json(table_info)
    
    # Main content area
    data_placeholder = st.empty()
    
    if refresh_btn or auto_refresh:
        with st.spinner("Fetching live game data from CornerProBet..."):
            games_data = scrape_cornerprobet_advanced()
            
        with data_placeholder.container():
            display_games_table(games_data)
            
        if st.session_state.debug and games_data:
            st.sidebar.subheader("Extraction Results")
            st.sidebar.write(f"Games found: {len(games_data)}")
            for i, game in enumerate(games_data[:3]):  # Show first 3
                st.sidebar.write(f"Game {i+1}: {game.get('home_team', '?')} vs {game.get('away_team', '?')}")
    
    # Initial load
    if not auto_refresh and not refresh_btn:
        st.info("Click 'Refresh Data' to load live games information")
        
        # Show instructions
        with st.expander("How to use this app"):
            st.markdown("""
            1. **Click 'Refresh Data'** to fetch current live games
            2. **Enable Auto-refresh** to automatically update every 30 seconds
            3. **Use Debug Mode** if you're not getting results to analyze the website structure
            4. **Check the table** below for live scores, corners, and shots on target
            """)
    
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
        "• Last update time\n\n"
        "**Note:** This app scrapes public data from CornerProBet. "
        "Please respect their terms of service."
    )

if __name__ == "__main__":
    main()
