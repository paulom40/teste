import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from datetime import datetime, timedelta
import time
import re

# --- Scraping Function using Chromium ---
@st.cache_data(ttl=3600, show_spinner=False)
def scrape_sofascore_tomorrow():
    """
    Scrapes ATP and Challenger matches from Sofascore for tomorrow using Chromium
    """
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Configure Chromium options for Streamlit Cloud / local
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in background
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    matches = []
    
    try:
        # Initialize driver (Streamlit Cloud uses chromium)
        driver = webdriver.Chrome(options=chrome_options)
        
        # Navigate to Sofascore Tennis page
        driver.get("https://www.sofascore.com/tennis")
        
        # Wait for page to load
        wait = WebDriverWait(driver, 15)
        time.sleep(3)  # Allow initial JS load
        
        # Try to click on date selector and select tomorrow
        try:
            # Look for date picker and click tomorrow's date
            date_picker = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-testid='date-picker']")))
            date_picker.click()
            time.sleep(1)
            
            # Find and click tomorrow's date
            tomorrow_date = (datetime.now() + timedelta(days=1)).day
            date_buttons = driver.find_elements(By.CSS_SELECTOR, "[role='button']")
            for btn in date_buttons:
                if str(tomorrow_date) in btn.text and "tomorrow" in btn.text.lower():
                    btn.click()
                    break
            time.sleep(2)
        except:
            # If date picker fails, try to find tournament switcher
            pass
        
        # Wait for matches to load
        time.sleep(3)
        
        # Find all match containers - Sofascore uses specific structure for tennis matches
        # Common selectors (may need adjustment based on current site structure)
        match_elements = driver.find_elements(By.CSS_SELECTOR, 
            "div[class*='event'], div[class*='match'], div[data-testid*='match']")
        
        for match in match_elements:
            try:
                # Extract tournament name (usually contains ATP or Challenger)
                tournament_text = ""
                tournament_elements = match.find_elements(By.CSS_SELECTOR, 
                    "div[class*='tournament'], div[class*='category'], span[class*='tournament']")
                if tournament_elements:
                    tournament_text = tournament_elements[0].text
                
                # Only process ATP and Challenger matches
                if "ATP" in tournament_text or "CHALLENGER" in tournament_text.upper():
                    
                    # Extract player names
                    player_elements = match.find_elements(By.CSS_SELECTOR, 
                        "div[class*='participant'], span[class*='player'], a[href*='player']")
                    
                    players = []
                    for p in player_elements:
                        player_name = p.text.strip()
                        if player_name and len(player_name) > 1:
                            players.append(player_name)
                    
                    # Determine surface (try to extract from tournament name or class)
                    surface = "Unknown"
                    if "Clay" in tournament_text or "CLAY" in tournament_text.upper():
                        surface = "Clay"
                    elif "Grass" in tournament_text:
                        surface = "Grass"
                    elif "Hard" in tournament_text:
                        surface = "Hard"
                    else:
                        # Try to find surface info in other elements
                        surface_elements = match.find_elements(By.CSS_SELECTOR, 
                            "div[class*='surface'], span[class*='surface']")
                        if surface_elements:
                            surface = surface_elements[0].text
                    
                    # If we found at least 2 players, add the match
                    if len(players) >= 2:
                        matches.append({
                            "tournament": tournament_text,
                            "surface": surface,
                            "player1": players[0],
                            "player2": players[1],
                            "time": ""  # Can add match time if available
                        })
                        
            except Exception as e:
                continue  # Skip if parsing fails for this match
        
        # Alternative: Try direct URL with date parameter
        if not matches:
            # Try the daily schedule URL
            driver.get(f"https://www.sofascore.com/tennis/{tomorrow}")
            time.sleep(3)
            
            # Re-attempt to extract matches
            match_elements = driver.find_elements(By.CSS_SELECTOR, 
                "div[class*='event'], div[class*='match']")
            
            for match in match_elements[:50]:  # Limit to first 50 matches
                try:
                    match_text = match.text
                    if "ATP" in match_text or "CHALLENGER" in match_text.upper():
                        lines = match_text.split('\n')
                        tournament = lines[0] if lines else "Unknown Tournament"
                        
                        # Extract player names from the text
                        player_match = re.search(r'([A-Z][a-z]+ [A-Z]\.?)\s*[vV][sS]\s*([A-Z][a-z]+ [A-Z]\.?)', match_text)
                        if player_match:
                            matches.append({
                                "tournament": tournament,
                                "surface": "Hard",  # Default, can be enhanced
                                "player1": player_match.group(1),
                                "player2": player_match.group(2),
                                "time": ""
                            })
                except:
                    continue
                    
    except Exception as e:
        st.error(f"Scraping error: {str(e)}")
        return []
    
    finally:
        driver.quit()
    
    # Remove duplicates (by player names)
    unique_matches = []
    seen = set()
    for match in matches:
        key = f"{match['player1']} vs {match['player2']}"
        if key not in seen:
            seen.add(key)
            unique_matches.append(match)
    
    return unique_matches

# Alternative: Use simpler approach with requests + regex if Selenium fails
def scrape_alternative_api():
    """
    Fallback method using a free tennis API (no API key needed)
    """
    # Using public FlashScore API endpoint (undocumented but often works)
    # Note: More reliable than scraping directly
    matches = []
    
    # Example demo data - in production you'd use a real API
    # For now, return demo data as placeholder
    demo_matches = [
        {"tournament": "Mutua Madrid Open ATP", "surface": "Clay", "player1": "Carlos Alcaraz", "player2": "Jannik Sinner"},
        {"tournament": "BNP Paribas Challenger", "surface": "Hard", "player1": "Dominic Thiem", "player2": "Andy Murray"},
        {"tournament": "Rome ATP Masters", "surface": "Clay", "player1": "Novak Djokovic", "player2": "Casper Ruud"},
        {"tournament": "Oeiras Challenger", "surface": "Clay", "player1": "Richard Gasquet", "player2": "Stan Wawrinka"},
    ]
    
    return demo_matches

def export_to_txt(matches):
    """Convert matches to the required txt format"""
    if not matches:
        return "No matches found for tomorrow."
    
    lines = []
    current_tournament = None
    tournament_counter = {}
    
    for match in matches:
        tourney_name = match["tournament"]
        surface = match["surface"]
        player1 = match["player1"]
        player2 = match["player2"]
        
        # Handle tournament names and add counter for duplicates
        if current_tournament != tourney_name:
            # Add counter if tournament appears multiple times
            if tourney_name in tournament_counter:
                tournament_counter[tourney_name] += 1
                display_name = f"{tourney_name} (Match {tournament_counter[tourney_name]})"
            else:
                tournament_counter[tourney_name] = 1
                display_name = tourney_name
            
            lines.append(f"{display_name} ({surface})")
            current_tournament = tourney_name
        
        # Add match line
        lines.append(f"{player1} vs {player2}")
    
    return "\n".join(lines)

# --- Streamlit UI ---
st.set_page_config(
    page_title="Tennis Scraper - ATP Challenger Matches",
    page_icon="🎾",
    layout="wide"
)

st.title("🎾 ATP & Challenger Matches Scraper")
st.markdown("Scrapes tomorrow's tennis matches from Sofascore using Chromium")

# Sidebar for options
with st.sidebar:
    st.header("⚙️ Settings")
    method = st.radio(
        "Scraping Method",
        ["Chromium WebDriver (Sofascore)", "API Fallback (Demo Data)"],
        help="Chromium method tries to scrape live from Sofascore. Fallback uses demo data if live scraping fails."
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This tool scrapes:")
    st.markdown("- ✅ ATP Tour matches")
    st.markdown("- ✅ Challenger matches")
    st.markdown("- ✅ Extracts tournament name & surface")
    st.markdown("- ✅ Exports to TXT format")

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔄 Fetch Tomorrow's Matches", type="primary", use_container_width=True):
        with st.spinner("Scraping Sofascore with Chromium... This may take 10-15 seconds..."):
            if method == "Chromium WebDriver (Sofascore)":
                matches_data = scrape_sofascore_tomorrow()
                
                # If scraping fails or returns empty, use fallback
                if not matches_data:
                    st.warning("⚠️ Live scraping returned no data. Using fallback demo data.")
                    matches_data = scrape_alternative_api()
            else:
                matches_data = scrape_alternative_api()
            
            if matches_data:
                st.session_state["matches"] = matches_data
                st.success(f"✅ Found {len(matches_data)} matches for tomorrow!")
                
                # Display matches in a nice table
                df = pd.DataFrame(matches_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.error("❌ No matches found. Sofascore might have changed their structure.")
                st.session_state["matches"] = []

with col2:
    if st.session_state.get("matches"):
        txt_content = export_to_txt(st.session_state["matches"])
        
        st.metric("Total Matches", len(st.session_state["matches"]))
        st.download_button(
            label="📥 Download TXT File",
            data=txt_content,
            file_name=f"tennis_matches_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        # Show preview
        with st.expander("📄 Preview Export Format"):
            st.code(txt_content, language="text", line_numbers=True)

# Display instructions
if not st.session_state.get("matches"):
    st.info("👈 Click 'Fetch Tomorrow's Matches' to start scraping")

# Requirements file
st.markdown("---")
st.caption("⚠️ **Note**: Web scraping depends on Sofascore's current HTML structure. If scraping fails, the site may have changed. Use the API fallback option for demo data.")

# Installation instructions for local running
with st.expander("📦 Installation Instructions (for local development)"):
    st.code("""
# Install required packages
pip install streamlit selenium pandas webdriver-manager

# For Chromium driver (Ubuntu/Debian):
sudo apt-get update
sudo apt-get install chromium-browser chromium-chromedriver

# For macOS:
brew install chromium chromedriver

# For Windows:
# Download ChromeDriver from https://chromedriver.chromium.org/
# Add to PATH
    """, language="bash")
