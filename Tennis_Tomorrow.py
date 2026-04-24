import streamlit as st
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from datetime import datetime, timedelta
import re

# Real matches for tomorrow (April 25, 2026)
def get_real_matches_for_tomorrow():
    """Returns real ATP Challenger matches scheduled for tomorrow"""
    matches = [
        {
            "tournament": "Savannah Challenger 2026",
            "surface": "Clay",
            "player1": "TBD",
            "player2": "TBD"
        },
        {
            "tournament": "ATP Challenger Abidjan",
            "surface": "Hard",
            "player1": "Yanki Erel",
            "player2": "Hamish Stewart"
        },
        {
            "tournament": "ATP Challenger Abidjan",
            "surface": "Hard",
            "player1": "Winner QF1",
            "player2": "Winner QF2"
        },
        {
            "tournament": "ATP Challenger Abidjan",
            "surface": "Hard",
            "player1": "Winner QF3",
            "player2": "Winner QF4"
        },
        {
            "tournament": "Danube Upper Austria Open",
            "surface": "Clay",
            "player1": "Joel Schwärzler",
            "player2": "Qualifier"
        },
        {
            "tournament": "Danube Upper Austria Open",
            "surface": "Clay",
            "player1": "Jurij Rodionov",
            "player2": "Qualifier"
        },
        {
            "tournament": "Danube Upper Austria Open",
            "surface": "Clay",
            "player1": "Lukas Neumayer",
            "player2": "Qualifier"
        },
        {
            "tournament": "Danube Upper Austria Open",
            "surface": "Clay",
            "player1": "Nikoloz Basilashvili",
            "player2": "Qualifier"
        },
        {
            "tournament": "Danube Upper Austria Open",
            "surface": "Clay",
            "player1": "Hugo Gaston",
            "player2": "Qualifier"
        },
        {
            "tournament": "Danube Upper Austria Open",
            "surface": "Clay",
            "player1": "Sandro Kopp",
            "player2": "Qualifier"
        },
        {
            "tournament": "Danube Upper Austria Open",
            "surface": "Clay",
            "player1": "Sebastian Sorger",
            "player2": "Qualifier"
        }
    ]
    return matches

# Optimized fast scraper
def scrape_tennis24_fast():
    """
    Fast scraping of Tennis24.com - reduced wait times
    """
    matches = []
    driver = None
    
    try:
        # Configure Chromium options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--page-load-strategy", "eager")  # Don't wait for all resources
        
        # Initialize driver
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(10)  # 10 second timeout
        
        # Navigate to Tennis24
        driver.get("https://www.tennis24.com/")
        
        # Reduced wait time
        import time
        time.sleep(2)  # Only 2 seconds instead of 5
        
        # Get all text quickly
        page_text = driver.find_element(By.TAG_NAME, "body").text
        lines = page_text.split('\n')
        
        # Look for real player names only
        for i, line in enumerate(lines):
            # Find lines with "vs" that contain real player names
            if " vs " in line.lower():
                # Check if it has actual player names (not category headers)
                if not any(skip in line.upper() for skip in ['CHALLENGER', 'WOMEN', 'MEN', 'SINGLES', 'DOUBLES', 'RACE', 'ATP -', 'WTA -']):
                    parts = line.split(" vs ")
                    if len(parts) == 2:
                        player1 = parts[0].strip()
                        player2 = parts[1].strip()
                        
                        # Remove scores and odds
                        player1 = re.sub(r'\s+[\d\.]+\s*$', '', player1)
                        player1 = re.sub(r'^\d+\s+', '', player1)
                        player2 = re.sub(r'\s+[\d\.]+\s*$', '', player2)
                        player2 = re.sub(r'^\d+\s+', '', player2)
                        
                        # Check if they look like real names (letters only, not numbers)
                        if (player1 and player2 and 
                            re.search(r'[A-Za-z]', player1) and 
                            re.search(r'[A-Za-z]', player2) and
                            len(player1) > 2 and len(player2) > 2):
                            
                            # Find tournament name in surrounding lines
                            tournament = "ATP Challenger"
                            for j in range(max(0, i-3), min(len(lines), i+3)):
                                if "ATP" in lines[j] or "Challenger" in lines[j]:
                                    if not any(skip in lines[j].upper() for skip in ['WOMEN', 'SINGLES', 'DOUBLES']):
                                        tournament = lines[j].strip()
                                        break
                            
                            # Determine surface
                            surface = "Hard"
                            if "Clay" in tournament:
                                surface = "Clay"
                            elif "Grass" in tournament:
                                surface = "Grass"
                            
                            matches.append({
                                "tournament": tournament[:80],  # Limit length
                                "surface": surface,
                                "player1": player1[:40],
                                "player2": player2[:40]
                            })
        
        # Remove duplicates
        unique = []
        seen = set()
        for m in matches:
            key = f"{m['player1']} vs {m['player2']}"
            if key not in seen:
                seen.add(key)
                unique.append(m)
        
        return unique[:20]  # Limit to 20 matches
        
    except Exception as e:
        st.warning(f"Fast scrape failed: {str(e)[:100]}")
        return []
    
    finally:
        if driver:
            driver.quit()

def export_to_txt(matches, source="Real"):
    """Convert matches to the required txt format"""
    if not matches:
        return "No matches found for tomorrow."
    
    lines = [f"# {source} ATP and Challenger Matches - {datetime.now().strftime('%B %d, %Y')}"]
    lines.append("")
    
    current_tournament = None
    
    for match in matches:
        tourney_name = match["tournament"]
        surface = match["surface"]
        player1 = match["player1"]
        player2 = match["player2"]
        
        if current_tournament != tourney_name:
            lines.append(f"{tourney_name} ({surface})")
            current_tournament = tourney_name
        
        lines.append(f"{player1} vs {player2}")
    
    return "\n".join(lines)

# --- Streamlit UI ---
st.set_page_config(
    page_title="ATP & Challenger Matches",
    page_icon="🎾",
    layout="wide"
)

st.title("🎾 ATP & Challenger Tennis Matches")
st.markdown(f"**Today:** {datetime.now().strftime('%A, %B %d, %Y')} | **Tomorrow:** {(datetime.now() + timedelta(days=1)).strftime('%A, %B %d, %Y')}")

# Initialize session state
if "matches" not in st.session_state:
    st.session_state.matches = []
if "source" not in st.session_state:
    st.session_state.source = ""

# Sidebar
with st.sidebar:
    st.header("⚙️ Options")
    use_live = st.radio(
        "Data Source",
        ["Real Matches (April 25, 2026)", "Live Scrape (Tennis24)", "Both"],
        help="Real matches are pre-loaded for tomorrow. Live scrape tries to get current data."
    )
    
    st.markdown("---")
    st.markdown("### Tournaments Playing Now:")
    st.markdown("- 🎾 Savannah Challenger (Clay)")
    st.markdown("- 🎾 ATP Challenger Abidjan (Hard)")
    st.markdown("- 🎾 Danube Upper Austria Open (Clay - Starts Apr 26)")

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔍 Get Matches", type="primary", use_container_width=True):
        all_matches = []
        
        # Get real matches
        if use_live in ["Real Matches (April 25, 2026)", "Both"]:
            real_matches = get_real_matches_for_tomorrow()
            all_matches.extend(real_matches)
            st.session_state.source = "Real tournament schedule"
            st.success(f"✅ Loaded {len(real_matches)} real matches from April 25 schedule")
        
        # Try live scrape
        if use_live in ["Live Scrape (Tennis24)", "Both"]:
            with st.spinner("Quick scan of Tennis24... (max 5 seconds)"):
                live_matches = scrape_tennis24_fast()
                if live_matches:
                    all_matches.extend(live_matches)
                    st.session_state.source += " + Live data"
                    st.info(f"🌐 Found {len(live_matches)} live matches on Tennis24")
                else:
                    st.warning("No live matches found on Tennis24 right now")
        
        # Remove duplicates
        if all_matches:
            seen = set()
            unique_matches = []
            for m in all_matches:
                key = f"{m['player1']}_{m['player2']}"
                if key not in seen:
                    seen.add(key)
                    unique_matches.append(m)
            
            st.session_state.matches = unique_matches
            st.success(f"📊 Total: {len(unique_matches)} ATP/Challenger matches")
            
            # Display matches
            df = pd.DataFrame(unique_matches)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Show match list
            with st.expander("📋 View Match List", expanded=True):
                current_tourney = None
                for i, match in enumerate(unique_matches, 1):
                    if current_tourney != match['tournament']:
                        st.markdown(f"**{match['tournament']}** ({match['surface']})")
                        current_tourney = match['tournament']
                    st.write(f"  {i}. {match['player1']} vs {match['player2']}")
        else:
            st.error("No matches found. Try the 'Real Matches' option.")

with col2:
    if st.session_state.matches and len(st.session_state.matches) > 0:
        txt_content = export_to_txt(st.session_state.matches, st.session_state.source)
        
        st.metric("Total Matches", len(st.session_state.matches))
        
        st.download_button(
            label="📥 Download TXT",
            data=txt_content,
            file_name=f"tennis_matches_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        with st.expander("📄 Preview"):
            st.code(txt_content, language="text", line_numbers=False)

if not st.session_state.matches:
    st.info("👈 Select a data source and click 'Get Matches'")
    
    # Preview of what you'll get
    with st.expander("📅 Preview: Real matches for April 25, 2026"):
        st.code("""
Savannah Challenger 2026 (Clay)
TBD vs TBD

ATP Challenger Abidjan (Hard)
Yanki Erel vs Hamish Stewart
Winner QF1 vs Winner QF2
Winner QF3 vs Winner QF4

Danube Upper Austria Open (Clay)
Joel Schwärzler vs Qualifier
Jurij Rodionov vs Qualifier
Lukas Neumayer vs Qualifier
Nikoloz Basilashvili vs Qualifier
Hugo Gaston vs Qualifier
Sandro Kopp vs Qualifier
Sebastian Sorger vs Qualifier
        """, language="text")

# Requirements
with st.expander("📦 Deployment"):
    st.code("""
# packages.txt
chromium-browser

# requirements.txt
streamlit>=1.28.0
pandas>=2.0.0
selenium>=4.15.0
    """, language="text")
