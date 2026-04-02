import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time

st.set_page_config(page_title="Tennis Odds Scraper", page_icon="🎾", layout="wide")

st.title("🎾 Tennis24 Odds Scraper")
st.markdown("Daily tennis matches with betting odds")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def scrape_tennis_odds():
    """
    Scrape tennis matches and odds from tennis24.com
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        url = "https://www.tennis24.com/"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        matches = []
        
        # Find all match containers
        # Note: The exact selectors may need adjustment based on the actual HTML structure
        match_rows = soup.find_all('div', class_='event__match')
        
        for match in match_rows[:50]:  # Limit to first 50 matches
            try:
                # Extract match details
                home_team = match.find('div', class_='event__participant--home')
                away_team = match.find('div', class_='event__participant--away')
                
                # Extract odds if available
                odds_elements = match.find_all('div', class_='odds')
                
                match_data = {
                    'Home Player': home_team.text.strip() if home_team else 'N/A',
                    'Away Player': away_team.text.strip() if away_team else 'N/A',
                    'Home Odds': odds_elements[0].text.strip() if len(odds_elements) > 0 else 'N/A',
                    'Away Odds': odds_elements[1].text.strip() if len(odds_elements) > 1 else 'N/A',
                    'Time': match.find('div', class_='event__time').text.strip() if match.find('div', class_='event__time') else 'N/A',
                    'Tournament': 'N/A'  # Would need to extract from section headers
                }
                
                matches.append(match_data)
                
            except Exception as e:
                continue
        
        if not matches:
            # Fallback: Try alternative parsing strategy
            st.warning("Primary parsing method found no matches. Trying alternative approach...")
            
        return matches
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Scraping error: {str(e)}")
        return []

# Sidebar controls
st.sidebar.header("Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 minutes")
refresh_button = st.sidebar.button("🔄 Refresh Now")

if refresh_button:
    st.cache_data.clear()

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Today's Matches")

with col2:
    st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))

# Fetch data
with st.spinner("Fetching tennis odds..."):
    matches_data = scrape_tennis_odds()

if matches_data:
    df = pd.DataFrame(matches_data)
    
    # Display statistics
    st.markdown("---")
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Total Matches", len(df))
    with metric_cols[1]:
        matches_with_odds = df[(df['Home Odds'] != 'N/A') & (df['Away Odds'] != 'N/A')]
        st.metric("Matches with Odds", len(matches_with_odds))
    with metric_cols[2]:
        st.metric("Tournaments", df['Tournament'].nunique())
    
    st.markdown("---")
    
    # Filters
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        show_only_odds = st.checkbox("Show only matches with odds", value=False)
    
    # Apply filters
    filtered_df = df.copy()
    if show_only_odds:
        filtered_df = filtered_df[(filtered_df['Home Odds'] != 'N/A') & (filtered_df['Away Odds'] != 'N/A')]
    
    # Display table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Home Player": st.column_config.TextColumn("Home Player", width="medium"),
            "Away Player": st.column_config.TextColumn("Away Player", width="medium"),
            "Home Odds": st.column_config.NumberColumn("Home Odds", format="%.2f"),
            "Away Odds": st.column_config.NumberColumn("Away Odds", format="%.2f"),
        }
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f'tennis_odds_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv',
    )
    
else:
    st.warning("⚠️ No matches found. The website structure may have changed or there might be no live matches.")
    st.info("""
    **Troubleshooting tips:**
    1. The website may be blocking automated requests
    2. The HTML structure may have changed (requires code update)
    3. Check if the website is accessible
    4. Consider using the tennis24.com API if available
    """)

# Auto-refresh logic
if auto_refresh:
    st.markdown("🔄 Auto-refresh enabled")
    time.sleep(300)  # 5 minutes
    st.rerun()

# Footer
st.markdown("---")
st.caption("⚠️ **Disclaimer**: This is for educational purposes. Always check the website's Terms of Service and robots.txt before scraping.")
