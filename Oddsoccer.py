Here's a complete Streamlit app for scraping soccer odds from Oddsportal, merging with 2025/2026 season power ratings, and running predictive analysis. **Important: Scraping Oddsportal may violate their Terms of Service — use this responsibly for educational purposes only.**

```python
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="Soccer Odds Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Critical Disclaimer
st.warning("""
⚠️ **Important Disclaimer**
- Scraping https://www.oddsportal.com/ may violate their Terms of Service. Use this tool at your own risk.
- Rate limiting (1.5s delay between requests) is enabled to avoid overloading their servers.
- This is a demo for educational purposes only — real-world use requires compliance with data usage policies.
""")

# ------------------------------
# Sidebar Settings
# ------------------------------
st.sidebar.header("Settings")

# Date Picker
today = datetime.today().date()
selected_date = st.sidebar.date_input(
    "Select Match Date",
    value=today,
    min_value=today - timedelta(days=7),
    max_value=today + timedelta(days=7)
)

# League Filter
popular_leagues = [
    "All Leagues", "Premier League", "La Liga", "Bundesliga",
    "Serie A", "Ligue 1", "Eredivisie", "Primeira Liga"
]
selected_leagues = st.sidebar.multiselect(
    "Filter by League",
    options=popular_leagues,
    default=["All Leagues"]
)

# Power Ratings Upload (2025/2026 Season)
st.sidebar.subheader("2025/2026 Power Ratings")
power_rating_file = st.sidebar.file_uploader(
    "Upload Power Ratings CSV",
    type=["csv"],
    help="Required columns: Team, PowerRating, GoalsScoredPerGame, GoalsConcededPerGame, CornersPerGame, CornersConcededPerGame"
)

# ------------------------------
# Scraping Functions
# ------------------------------
def get_match_links(date: datetime) -> list:
    """Scrape match URLs and leagues from Oddsportal's daily soccer page"""
    date_str = date.strftime("%Y-%m-%d")
    base_url = f"https://www.oddsportal.com/matches/soccer/{date_str}/"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }

    try:
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        match_links = []
        # Find all match rows (active/deactivated)
        match_rows = soup.find_all("tr", class_=["deactivate", "active"])
        
        for row in match_rows:
            link_tag = row.find("a", class_="name")
            if link_tag:
                # Extract league from parent section
                league_tag = row.find_previous("th", class_="first2 tl")
                league = league_tag.text.strip() if league_tag else "Unknown"
                
                match_links.append({
                    "link": f"https://www.oddsportal.com{link_tag['href']}",
                    "league": league
                })
        
        return match_links

    except Exception as e:
        st.error(f"Failed to fetch match links: {str(e)}")
        return []

def scrape_match_details(match_link: str) -> dict:
    """Scrape detailed odds (1X2, O/U, Corners) from a single match page"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }

    try:
        response = requests.get(match_link, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        # Extract teams
        team_header = soup.find("h1").text.strip().split(" - ")
        home_team = team_header[0].strip()
        away_team = team_header[1].strip()

        # Extract 1X2 Odds
        x12_odds = {}
        x12_table = soup.find("table", class_="table-main odds-compare")
        if x12_table:
            odds_cells = x12_table.find_all("td", class_="odds-nowrp")
            if len(odds_cells) >= 3:
                x12_odds = {
                    "home_win": float(odds_cells[0].text.strip()),
                    "draw": float(odds_cells[1].text.strip()),
                    "away_win": float(odds_cells[2].text.strip())
                }

        # Extract Over/Under 2.5 Goals Odds
        ou_odds = {}
        ou_tab = soup.find("div", id="odds_ou")
        if ou_tab:
            ou_table = ou_tab.find("table", class_="table-main")
            if ou_table:
                for row in ou_table.find_all("tr"):
                    th_text = row.find("th").text.strip() if row.find("th") else ""
                    if "2.5" in th_text:
                        ou_cells = row.find_all("td", class_="odds-nowrp")
                        if len(ou_cells) >= 2:
                            ou_odds = {
                                "over_2.5": float(ou_cells[0].text.strip()),
                                "under_2.5": float(ou_cells[1].text.strip()),
                                "line": 2.5
                            }

        # Extract Corners Odds (typical line: 9.5/10.5)
        corners_odds = {}
        corners_tab = soup.find("div", id="odds_corners")
        if corners_tab:
            corners_table = corners_tab.find("table", class_="table-main")
            if corners_table:
                for row in corners_table.find_all("tr"):
                    th_text = row.find("th").text.strip() if row.find("th") else ""
                    if any(line in th_text for line in ["9.5", "10.5"]):
                        corners_cells = row.find_all("td", class_="odds-nowrp")
                        if len(corners_cells) >= 2:
                            corners_odds = {
                                "over": float(corners_cells[0].text.strip()),
                                "under": float(corners_cells[1].text.strip()),
                                "line": float(th_text.split()[-1])
                            }

        # Extract kickoff time
        kickoff = soup.find("p", class_="date").text.strip() if soup.find("p", class_="date") else "N/A"

        return {
            "home_team": home_team,
            "away_team": away_team,
            "kickoff": kickoff,
            "home_win_odds": x12_odds.get("home_win", np.nan),
            "draw_odds": x12_odds.get("draw", np.nan),
            "away_win_odds": x12_odds.get("away_win", np.nan),
            "ou_2.5_over": ou_odds.get("over_2.5", np.nan),
            "ou_2.5_under": ou_odds.get("under_2.5", np.nan),
            "ou_line": ou_odds.get("line", np.nan),
            "corners_over_odds": corners_odds.get("over", np.nan),
            "corners_under_odds": corners_odds.get("under", np.nan),
            "corners_line": corners_odds.get("line", np.nan)
        }

    except Exception as e:
        st.warning(f"Failed to scrape {match_link}: {str(e)}")
        return None
    
    finally:
        time.sleep(1.5)  # Rate limit to avoid blocking

# ------------------------------
# Main App Logic
# ------------------------------
st.header("Soccer Odds Scraper & Predictor")
st.subheader(f"Matches for {selected_date.strftime('%Y-%m-%d')}")

# Step 1: Scrape Match Data
if st.button("Start Scraping"):
    with st.spinner("Fetching match links..."):
        all_match_links = get_match_links(selected_date)

    if not all_match_links:
        st.info("No matches found for the selected date.")
    else:
        # Filter by selected leagues
        if "All Leagues" not in selected_leagues:
            filtered_links = [
                link for link in all_match_links
                if link["league"] in selected_leagues
            ]
        else:
            filtered_links = all_match_links

        st.success(f"Found {len(filtered_links)} matches to scrape.")
        
        # Progress bar for detail scraping
        progress_bar = st.progress(0)
        match_data = []

        for idx, link in enumerate(filtered_links):
            match_details = scrape_match_details(link["link"])
            if match_details:
                match_details["league"] = link["league"]
                match_data.append(match_details)
            progress_bar.progress((idx + 1) / len(filtered_links))

        # Store scraped data in session state
        st.session_state["scraped_data"] = pd.DataFrame(match_data)
        st.subheader("Scraped Match Data")
        st.dataframe(st.session_state["scraped_data"], use_container_width=True)

# Step 2: Load Power Ratings
if "scraped_data" in st.session_state and power_rating_file is not None:
    df_power = pd.read_csv(power_rating_file)
    
    # Validate power rating columns
    required_cols = [
        "Team", "PowerRating", "GoalsScoredPerGame", 
        "GoalsConcededPerGame", "CornersPerGame", "CornersConcededPerGame"
    ]
    
    if not all(col in df_power.columns for col in required_cols):
        st.error(f"Power Ratings CSV must include: {', '.join(required_cols)}")
    else:
        st.session_state["power_ratings"] = df_power
        st.subheader("2025/2026 Season Power Ratings")
        st.dataframe(df_power, use_container_width=True)

# Step 3: Merge Data & Create Features
if "scraped_data" in st.session_state and "power_ratings" in st.session_state:
    df_matches = st.session_state["scraped_data"]
    df_power = st.session_state["power_ratings"]

    # Merge home team power ratings
    df_merged = df_matches.merge(
        df_power, left_on="home_team", right_on="Team", 
        how="left", suffixes=("", "_home")
    )

    # Merge away team power ratings
    df_merged = df_merged.merge(
        df_power, left_on="away_team", right_on="Team",
        how="left", suffixes=("_home", "_away")
    )

    # Drop duplicate team columns
    df_merged = df_merged.drop(columns=["Team_home", "Team_away"])

    # Create predictive features
    df_merged["power_rating_diff"] = df_merged["PowerRating_home"] - df_merged["PowerRating_away"]
    df_merged["goals_diff"] = df_merged["GoalsScoredPerGame_home"] - df_merged["GoalsConcededPerGame_away"]
    df_merged["conceded_diff"] = df_merged["GoalsConcededPerGame_home"] - df_merged["GoalsScoredPerGame_away"]
    df_merged["corners_diff"] = df_merged["CornersPerGame_home"] - df_merged["CornersConcededPerGame_away"]

    # Remove rows with missing power ratings
    df_merged = df_merged.dropna(subset=["PowerRating_home", "PowerRating_away"])
    st.session_state["merged_data"] = df_merged

    st.subheader("Merged Odds + Power Ratings Data")
    st.dataframe(df_merged, use_container_width=True)

# Step 4: Train Model & Predict
if "merged_data" in st.session_state:
    df_merged = st.session_state["merged_data"]

    st.subheader("Prediction Model")
    
    # Define features and simulated targets (replace with real historical data in production)
    features = [
        "power_rating_diff", "goals_diff", "conceded_diff",
        "home_win_odds", "draw_odds", "away_win_odds"
    ]

    # Simulate targets (for demo only — use real match outcomes in production)
    if "target" not in df_merged.columns:
        df_merged["target"] = np.where(
            df_merged["power_rating_diff"] > 2, 0,  # 0 = Home Win
            np.where(df_merged["power_rating_diff"] < -2, 2, 1)  # 2 = Away Win, 1 = Draw
        )

    # Prepare training data
    X = df_merged[features].dropna()
    y = df_merged.loc[X.index, "target"]

    if len(X) < 5:
        st.warning("Not enough data to train the model (need at least 5 matches with complete data).")
    else:
        # Train logistic regression model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Model Trained | Test Accuracy: {accuracy:.2f}")

        # Predict outcomes for all matches
        df_merged["predicted_outcome"] = model.predict(df_merged[features])
        df_merged["prediction_confidence"] = model.predict_proba(df_merged[features]).max(axis=1)

        # Map numeric outcomes to labels
        outcome_labels = {0: "Home Win", 1: "Draw", 2: "Away Win"}
        df_merged["predicted_outcome"] = df_merged["predicted_outcome"].map(outcome_labels)

        # Predict Over/Under 2.5 Goals
        ou_features = [
            "power_rating_diff", "goals_diff", "conceded_diff",
            "ou_2.5_over", "ou_2.5_under"
        ]

        df_merged["ou_target"] = np.where(df_merged["goals_diff"] > 1, 1, 0)  # Simulated target
        X_ou = df_merged[ou_features].dropna()
        y_ou = df_merged.loc[X_ou.index, "ou_target"]

        if len(X_ou) >= 5:
            ou_model = LogisticRegression(max_iter=1000)
            ou_model.fit(X_ou, y_ou)
            df_merged["predicted_ou"] = ou_model.predict(df_merged[ou_features])
            df_merged["ou_confidence"] = ou_model.predict_proba(df_merged[ou_features]).max(axis=1)
            df_merged["predicted_ou"] = df_merged["predicted_ou"].map({1: "Over 2.5", 0: "Under 2.5"})

        # Display predictions
        st.subheader("Prediction Results")
        display_cols = [
            "league", "home_team", "away_team", "kickoff",
            "home_win_odds", "draw_odds", "away_win_odds",
            "predicted_outcome", "prediction_confidence",
            "predicted_ou", "ou_confidence"
        ]
        st.dataframe(
            df_merged[display_cols].sort_values("prediction_confidence", ascending=False),
            use_container_width=True
        )

        # Visualizations
        st.subheader("Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(
                df_merged,
                x="home_team",
                y="prediction_confidence",
                color="predicted_outcome"
            )

        with col2:
            st.scatter_chart(
                df_merged,
                x="power_rating_diff",
                y="prediction_confidence",
                color="predicted_outcome"
            )

# ------------------------------
# Notes & Limitations
# ------------------------------
st.subheader("Key Notes & Limitations")
st.markdown("""
- **Scraping Reliability**: Oddsportal frequently updates its HTML structure — you may need to adjust selectors
