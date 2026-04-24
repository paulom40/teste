import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from sofascore_wrapper.api import SofascoreAPI

# --- Async wrapper for Streamlit ---
def run_async(coro):
    """Run async function in Streamlit"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# --- Fetch matches using sofascore-wrapper ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tomorrow_matches():
    """Fetch ATP and Challenger matches for tomorrow using sofascore-wrapper"""
    tomorrow_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    matches = []
    
    async def get_matches():
        api = SofascoreAPI()
        try:
            # Get scheduled events for tennis (sport_id = 5 for tennis)
            # Search for events on tomorrow's date
            schedule = await api.get_schedule_by_date(sport_id=5, date=tomorrow_date)
            
            if schedule and 'events' in schedule:
                for event in schedule['events']:
                    tournament = event.get('tournament', {})
                    tournament_name = tournament.get('name', '')
                    
                    # Filter ATP and Challenger
                    if 'ATP' in tournament_name or 'Challenger' in tournament_name:
                        # Determine surface
                        surface = 'Hard'
                        if 'Clay' in tournament_name:
                            surface = 'Clay'
                        elif 'Grass' in tournament_name:
                            surface = 'Grass'
                        
                        # Get players
                        home = event.get('homeTeam', {}).get('name', '')
                        away = event.get('awayTeam', {}).get('name', '')
                        
                        if home and away:
                            matches.append({
                                'tournament': tournament_name,
                                'surface': surface,
                                'player1': home,
                                'player2': away
                            })
        finally:
            await api.close()
        return matches
    
    return run_async(get_matches())

# --- Export function ---
def export_to_txt(matches):
    if not matches:
        return "No matches found for tomorrow."
    
    lines = []
    current_tournament = None
    
    for match in matches:
        if current_tournament != match['tournament']:
            lines.append(f"{match['tournament']} ({match['surface']})")
            current_tournament = match['tournament']
        lines.append(f"{match['player1']} vs {match['player2']}")
    
    return '\n'.join(lines)

# --- Streamlit UI ---
st.set_page_config(page_title="ATP & Challenger Matches", page_icon="🎾", layout="wide")

st.title("🎾 ATP & Challenger Tennis Matches for Tomorrow")
st.markdown("Fetch real matches from Sofascore using the official wrapper")

if "matches" not in st.session_state:
    st.session_state.matches = []

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔍 Fetch Tomorrow's Matches", type="primary", use_container_width=True):
        with st.spinner("Fetching from Sofascore..."):
            matches_data = fetch_tomorrow_matches()
            
            if matches_data:
                st.session_state.matches = matches_data
                st.success(f"✅ Found {len(matches_data)} matches!")
                df = pd.DataFrame(matches_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("No ATP/Challenger matches found for tomorrow.")
                st.info("Check the official ATP schedule at atptour.com")

with col2:
    if st.session_state.matches:
        txt_content = export_to_txt(st.session_state.matches)
        st.metric("Total Matches", len(st.session_state.matches))
        st.download_button(
            label="📥 Download TXT",
            data=txt_content,
            file_name=f"tennis_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        with st.expander("Preview"):
            st.code(txt_content, language="text")

if not st.session_state.matches:
    st.info("👈 Click the button to fetch tomorrow's ATP and Challenger matches")
