import streamlit as st
import pandas as pd
from datetime import datetime
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import json

st.set_page_config(page_title="Tennis ATP & Challenger Scraper", page_icon="🎾", layout="wide")

st.title("🎾 ATP & Challenger Tennis Matches Scraper")
st.markdown("**Real-time data from Sofascore - ATP Singles & Challenger Tournaments**")

@st.cache_resource
def get_driver():
    """Initialize Selenium WebDriver with Chrome options"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        return driver
    except Exception as e:
        st.error(f"Failed to initialize Chrome driver: {e}")
        return None

@st.cache_data(ttl=1800)
def scrape_sofascore_selenium(_driver):
    """
    Scrape tennis data from Sofascore using Selenium
    """
    if _driver is None:
        return []
    
    matches = []
    
    try:
        # Load Sofascore tennis page
        _driver.get("https://www.sofascore.com/tennis")
        
        # Wait for matches to load
        wait = WebDriverWait(_driver, 10)
        
        # Wait for match elements to be present
        time.sleep(3)  # Give time for dynamic content to load
        
        # Find all match containers
        match_elements = _driver.find_elements(By.CLASS_NAME, "Box")
        
        for match_elem in match_elements[:50]:  # Limit to 50 matches
            try:
                # Extract match details
                match_text = match_elem.text
                
                # Parse the text to extract information
                lines = match_text.split('\n')
                
                if len(lines) >= 3:
                    matches.append({
                        'Tournament': lines[0] if len(lines) > 0 else 'Unknown',
                        'Player1': lines[1] if len(lines) > 1 else 'Unknown',
                        'Player2': lines[2] if len(lines) > 2 else 'Unknown',
                        'Score': lines[3] if len(lines) > 3 else '',
                        'Time': lines[4] if len(lines) > 4 else '',
                        'Raw_Data': match_text
                    })
            except Exception as e:
                continue
        
    except Exception as e:
        st.error(f"Selenium scraping error: {e}")
    
    return matches

@st.cache_data(ttl=1800)
def scrape_flashscore_api():
    """
    Try to access Flashscore's API endpoints directly
    """
    import requests
    
    matches = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://www.flashscore.com/'
    }
    
    # These are potential API endpoints (may need verification)
    api_urls = [
        'https://d.flashscore.com/x/feed/f_2_1_en_1',  # Tennis feed
    ]
    
    for url in api_urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                # Try to parse response
                data = response.text
                matches.append({
                    'Source': 'Flashscore API',
                    'Data': data[:500]  # First 500 chars
                })
        except Exception as e:
            continue
    
    return matches

def parse_atp_challenger_only(matches_df):
    """
    Filter only ATP and Challenger tournaments
    """
    if matches_df.empty:
        return matches_df
    
    # Keywords to identify ATP and Challenger tournaments
    atp_keywords = ['ATP', 'Grand Slam', 'Masters', 'Challenger', 'Australian Open', 
                     'French Open', 'Wimbledon', 'US Open', 'Roland Garros']
    
    # Filter rows
    filtered = matches_df[
        matches_df.apply(
            lambda row: any(keyword.lower() in str(row).lower() for keyword in atp_keywords),
            axis=1
        )
    ]
    
    return filtered

# Sidebar
st.sidebar.header("⚙️ Configuration")

scraping_method = st.sidebar.radio(
    "Scraping Method",
    ["Selenium (Recommended)", "API Attempt", "Both"]
)

filter_tournaments = st.sidebar.checkbox("Show only ATP & Challenger", value=True)
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)

if st.sidebar.button("🔄 Refresh Data Now"):
    st.cache_data.clear()
    st.rerun()

# Main content
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("⏰ Last Update", datetime.now().strftime("%H:%M:%S"))
with col2:
    st.metric("📊 Method", scraping_method.split(" ")[0])
with col3:
    status = "🟢 Active" if auto_refresh else "⏸️ Manual"
    st.metric("🔄 Status", status)

st.markdown("---")

# Scraping section
with st.spinner("🎾 Fetching tennis data..."):
    all_matches = []
    
    if scraping_method in ["Selenium (Recommended)", "Both"]:
        st.info("🔧 Initializing Selenium WebDriver...")
        driver = get_driver()
        
        if driver:
            selenium_data = scrape_sofascore_selenium(driver)
            all_matches.extend(selenium_data)
            st.success(f"✅ Selenium: Found {len(selenium_data)} matches")
    
    if scraping_method in ["API Attempt", "Both"]:
        api_data = scrape_flashscore_api()
        all_matches.extend(api_data)
        st.success(f"✅ API: Found {len(api_data)} entries")

# Display results
if all_matches:
    df = pd.DataFrame(all_matches)
    
    # Filter for ATP & Challenger if requested
    if filter_tournaments:
        original_count = len(df)
        df = parse_atp_challenger_only(df)
        st.info(f"🎯 Filtered: {len(df)} ATP/Challenger matches (from {original_count} total)")
    
    # Display statistics
    st.subheader("📊 Match Statistics")
    
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Total Matches", len(df))
    with metric_cols[1]:
        tournaments = df.get('Tournament', pd.Series()).nunique()
        st.metric("Tournaments", tournaments)
    with metric_cols[2]:
        live_matches = len(df[df.get('Score', '').astype(str).str.contains('-', na=False)])
        st.metric("Live/Completed", live_matches)
    with metric_cols[3]:
        upcoming = len(df) - live_matches
        st.metric("Upcoming", upcoming)
    
    st.markdown("---")
    
    # Data table
    st.subheader("📋 Match Data")
    st.dataframe(df, use_container_width=True, height=400)
    
    # Download section
    st.markdown("---")
    st.subheader("💾 Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f'tennis_atp_challenger_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
            mime='text/csv',
        )
    
    with col2:
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f'tennis_atp_challenger_{datetime.now().strftime("%Y%m%d_%H%M")}.json',
            mime='application/json',
        )
    
    with col3:
        excel_buffer = df.to_excel(index=False, engine='openpyxl')
        st.download_button(
            label="📥 Download Excel",
            data=excel_buffer,
            file_name=f'tennis_atp_challenger_{datetime.now().strftime("%Y%m%d_%H%M")}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )

else:
    st.error("❌ No data retrieved")
    
    st.warning("""
    ### Troubleshooting:
    
    1. **Selenium not working on Streamlit Cloud?**
       - Streamlit Cloud may not support Selenium by default
       - Consider using Streamlit Community Cloud with custom packages
       - Or deploy on Heroku/Railway with buildpacks
    
    2. **Alternative Approach:**
       - Use requests + BeautifulSoup for static content
       - Find sites with public APIs
       - Use browser automation locally, save to CSV, then upload
    
    3. **Recommended Free Sources:**
       - ATP Official Stats (atp-stats.com)
       - Tennis Abstract
       - Ultimate Tennis Statistics
    """)

# Information section
st.markdown("---")
with st.expander("ℹ️ About This Scraper"):
    st.markdown("""
    ### Features:
    - ✅ Scrapes ATP Singles matches
    - ✅ Scrapes Challenger tournaments
    - ✅ Real-time data updates
    - ✅ Multiple export formats (CSV, JSON, Excel)
    
    ### Data Sources:
    - Sofascore (primary)
    - Flashscore (API attempt)
    
    ### Limitations:
    - May not work on all hosting platforms (Selenium requirements)
    - Rate limited by source websites
    - No betting odds (most require paid APIs)
    
    ### For Production:
    Deploy on platforms that support Selenium:
    - Heroku (with Chrome buildpack)
    - Railway
    - DigitalOcean
    - AWS EC2
    """)

st.markdown("---")
st.caption("⚠️ Educational purposes only. Respect website ToS. Data accuracy not guaranteed.")

# Auto-refresh
if auto_refresh:
    time.sleep(300)  # 5 minutes
    st.rerun()
