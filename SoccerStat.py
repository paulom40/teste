import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time, random, json, os, asyncio
from typing import List, Dict, Any
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import logging
from scipy.stats import poisson

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
CACHE_FILE = "flashscore_value_cache.json"
CACHE_TTL = 300
MAX_RETRIES = 3
BASE_DELAY = 3
PROXIES = []
USE_PROXY = len(PROXIES) > 0
MAX_MATCHES = 60
MIN_EDGE = 0.03  # 3% edge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ua = UserAgent()

# --------------------------------------------------------------
# POWER RATINGS (ELO-style)
# --------------------------------------------------------------
INITIAL_RATINGS = {
    'manchester city': 95, 'liverpool': 93, 'arsenal': 91, 'chelsea': 88, 'manchester united': 87,
    'tottenham': 86, 'newcastle': 85, 'aston villa': 84, 'brighton': 83, 'west ham': 82,
    'real madrid': 96, 'barcelona': 94, 'atletico madrid': 90, 'sevilla': 86, 'real sociedad': 85,
    'bayern munich': 95, 'borussia dortmund': 90, 'rb leipzig': 89, 'bayer leverkusen': 88,
    'inter milan': 90, 'ac milan': 89, 'napoli': 91, 'juventus': 88, 'lazio': 85, 'roma': 85,
    'psg': 92, 'monaco': 86, 'marseille': 85, 'lyon': 84, 'lille': 83,
    'default': 70
}

# --------------------------------------------------------------
# CACHE
# --------------------------------------------------------------
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
                if data["timestamp"] > time.time() - CACHE_TTL:
                    return data["matches"]
        except: pass
    return None

def save_cache(matches):
    payload = {"timestamp": time.time(), "matches": matches}
    try: open(CACHE_FILE, "w").write(json.dumps(payload))
    except: pass

# --------------------------------------------------------------
# SELECTOR ENGINE
# --------------------------------------------------------------
class SelectorEngine:
    @staticmethod
    def match_rows(soup):
        patterns = ["div.event__match", "div[class*='event__match']", "div[data-eventid]"]
        for p in patterns:
            rows = soup.select(p)
            if rows: return rows, p
        return [], None

    @staticmethod
    def team(row, home): return (row.select_one(f".event__participant--{'home' if home else 'away'}") or {}).get_text(strip=True, default="?")
    @staticmethod
    def time(row): return (row.select_one(".event__time") or {}).get_text(strip=True, default="TBD")
    @staticmethod
    def score(row): 
        txt = (row.select_one(".event__score") or {}).get_text(strip=True, default="")
        return txt if ":" in txt else None
    @staticmethod
    def minute(row): return (row.select_one(".event__stage") or {}).get_text(strip=True, default="")
    @staticmethod
    def league(soup, row):
        prev = row.find_previous(["div", "h3"], class_=lambda x: x and any(k in x for k in ["header", "league"]))
        return prev.get_text(strip=True).split(" - ")[0] if prev else "?"

# --------------------------------------------------------------
# ODDS SCRAPER
# --------------------------------------------------------------
async def get_odds_for_match(page) -> Dict:
    odds = {
        "home": None, "draw": None, "away": None,
        "over_2_5": None, "under_2_5": None
    }
    try:
        await page.wait_for_selector("table.odds", timeout=8000)
        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")
        for tr in soup.select("table.odds tr"):
            cells = tr.select("td")
            if len(cells) < 2: continue
            label = cells[0].get_text(strip=True).lower()
            val = cells[1].get_text(strip=True).replace(",", ".")
            try: val = float(val)
            except: continue

            if "1" in label or "home" in label: odds["home"] = val
            elif "x" in label or "draw" in label: odds["draw"] = val
            elif "2" in label or "away" in label: odds["away"] = val
            elif "over 2.5" in label: odds["over_2_5"] = val
            elif "under 2.5" in label: odds["under_2_5"] = val
        return odds
    except: return odds

# --------------------------------------------------------------
# VALUE BET CALCULATOR
# --------------------------------------------------------------
def get_rating(team: str, ratings: dict) -> float:
    return ratings.get(team.lower(), INITIAL_RATINGS['default'])

def true_1x2_probs(home_rating: float, away_rating: float, home_adv: float = 3.0):
    total = home_rating + away_rating + home_adv
    p_home = (home_rating + home_adv) / total
    p_away = away_rating / total
    p_draw = 1 - p_home - p_away
    p_draw = max(p_draw, 0.22)
    total_p = p_home + p_away + p_draw
    return p_home / total_p, p_draw / total_p, p_away / total_p

def expected_goals(rating: float) -> float:
    return 0.02 * rating + 0.5  # Simple linear model

def true_over_under_prob(home_rating: float, away_rating: float, line: float = 2.5):
    home_goals = expected_goals(home_rating)
    away_goals = expected_goals(away_rating)
    total_goals = home_goals + away_goals
    return 1 - poisson.cdf(line, total_goals), poisson.cdf(line, total_goals)

def calculate_value_bets(matches: List[Dict], ratings: dict):
    value_bets = []
    for m in matches:
        home, away = m['home_team'], m['away_team']
        r_home = get_rating(home, ratings)
        r_away = get_rating(away, ratings)

        # 1X2
        p_home, p_draw, p_away = true_1x2_probs(r_home, r_away)
        odds = m['odds_home'], m['odds_draw'], m['odds_away']
        probs = [p_home, p_draw, p_away]
        labels = ['home', 'draw', 'away']

        # O/U
        p_over, p_under = true_over_under_prob(r_home, r_away)
        ou_odds = m['odds_over_2_5'], m['odds_under_2_5']
        ou_probs = [p_over, p_under]
        ou_labels = ['over_2_5', 'under_2_5']

        # Find best
        best_ev = -1
        best = None
        for i, (prob, odd, label) in enumerate(zip(probs + ou_probs, odds + ou_odds, labels + ou_labels)):
            if odd is None: continue
            ev = prob * (odd - 1) - (1 - prob)
            if ev > best_ev and ev > MIN_EDGE:
                best_ev = ev
                best = {
                    'match': f"{home} vs {away}",
                    'league': m['league'],
                    'time': m['match_time'],
                    'bet': 'Over 2.5' if 'over' in label else 'Under 2.5' if 'under' in label else label.title(),
                    'odds': odd,
                    'edge': round(ev * 100, 1),
                    'true_prob': round(prob * 100, 1),
                    'is_live': m['is_live']
                }
        if best:
            value_bets.append(best)

    return sorted(value_bets, key=lambda x: x['edge'], reverse=True)[:10]

# --------------------------------------------------------------
# MAIN SCRAPER
# --------------------------------------------------------------
async def scrape_flashscore() -> List[Dict]:
    proxy = random.choice(PROXIES) if USE_PROXY else None
    matches = []

    for attempt in range(MAX_RETRIES):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, args=[
                    "--no-sandbox", "--disable-blink-features=AutomationControlled"
                ])
                context = await browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent=ua.random
                )
                await context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => false});")

                page = await context.new_page()
                await page.goto("https://www.flashscore.com/football/", timeout=30000)
                await page.wait_for_load_state("networkidle")

                for _ in range(3):
                    await page.evaluate("window.scrollBy(0, 400)")
                    await asyncio.sleep(random.uniform(0.6, 1.4))

                content = await page.content()
                soup = BeautifulSoup(content, "html.parser")
                rows, _ = SelectorEngine.match_rows(soup)
                if not rows: continue

                for row in rows[:MAX_MATCHES]:
                    try:
                        home = SelectorEngine.team(row, True)
                        away = SelectorEngine.team(row, False)
                        if not home or not away or home == away: continue

                        time_str = SelectorEngine.time(row)
                        score_str = SelectorEngine.score(row)
                        is_live = bool(score_str)
                        home_score = away_score = 0
                        minute = ""

                        if is_live:
                            h, a = score_str.split(":")
                            home_score, away_score = int(h), int(a)
                            minute = SelectorEngine.minute(row)

                        league = SelectorEngine.league(soup, row)

                        # Get odds
                        odds = {"home": None, "draw": None, "away": None, "over_2_5": None, "under_2_5": None}
                        try:
                            link = row.select_one("a[href*='/match/']")
                            if link and "href" in link.attrs:
                                await page.goto("https://www.flashscore.com" + link["href"], timeout=20000)
                                await page.wait_for_load_state("networkidle")
                                odds = await get_odds_for_match(page)
                                await page.go_back()
                        except: pass

                        matches.append({
                            "home_team": home, "away_team": away, "league": league,
                            "match_time": time_str, "is_live": is_live,
                            "home_score": home_score, "away_score": away_score, "minute": minute,
                            "odds_home": odds["home"], "odds_draw": odds["draw"], "odds_away": odds["away"],
                            "odds_over_2_5": odds["over_2_5"], "odds_under_2_5": odds["under_2_5"],
                            "timestamp": datetime.now().isoformat()
                        })
                    except: continue

                await browser.close()
                if matches:
                    logger.info(f"Scraped {len(matches)} matches")
                    return matches
        except Exception as e:
            logger.error(f"Attempt {attempt+1}: {e}")
        await asyncio.sleep(BASE_DELAY * (2 ** attempt) + random.uniform(0, 3))
    return []

# --------------------------------------------------------------
# CACHED ENTRY
# --------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def get_matches_with_odds():
    cached = load_cache()
    if cached: return cached
    matches = asyncio.run(scrape_flashscore())
    if matches: save_cache(matches)
    return matches or fallback_matches()

def fallback_matches():
    return [{
        "home_team": "Man City", "away_team": "Arsenal", "league": "Premier League",
        "match_time": "20:00", "is_live": False, "home_score": 0, "away_score": 0,
        "minute": "", "odds_home": 1.75, "odds_draw": 3.80, "odds_away": 4.50,
        "odds_over_2_5": 1.90, "odds_under_2_5": 1.95,
        "timestamp": datetime.now().isoformat()
    }]

# --------------------------------------------------------------
# UI
# --------------------------------------------------------------
def main():
    st.set_page_config(page_title="FlashScore Value Bets", page_icon="money", layout="wide")
    st.markdown("""
    <style>
        .value-card {background: linear-gradient(135deg, #11998e, #38ef7d); color: white; padding: 16px; border-radius: 12px; margin: 10px 0; border: 2px solid #28a745;}
        .odds {font-weight:bold; color:#ffd700;}
        .edge {font-weight:bold; color:#ff6b6b;}
        .inplay {background:#ff4b4b; color:white; padding:12px; border-radius:8px;}
        .upcoming {background:#1f77b4; color:white; padding:12px; border-radius:8px;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center;'>FlashScore Value Bets</h1>", unsafe_allow_html=True)

    # Init ratings
    if 'ratings' not in st.session_state:
        st.session_state.ratings = INITIAL_RATINGS.copy()

    with st.sidebar:
        st.header("Controls")
        if st.button("Force Refresh"):
            st.cache_data.clear()
            st.rerun()
        st.metric("Min Edge", f"{MIN_EDGE*100}%")

    matches = get_matches_with_odds()
    value_bets = calculate_value_bets(matches, st.session_state.ratings)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Top Value Bets")
        if value_bets:
            for b in value_bets[:5]:
                live = "LIVE" if b['is_live'] else b['time']
                st.markdown(f"""
                <div class="value-card">
                    <strong>{b['match']}</strong><br>
                    <small>{live} • {b['league']}</small><br>
                    <strong>Bet:</strong> {b['bet']} @ <span class="odds">{b['odds']}</span><br>
                    <strong>Edge:</strong> <span class="edge">+{b['edge']}%</span> (True: {b['true_prob']}%)</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No value bets found. Market is efficient.")

    with col2:
        st.subheader("All Matches")
        live = [m for m in matches if m['is_live']]
        upcoming = [m for m in matches if not m['is_live']]

        for m in live[:5]:
            st.markdown(f"<div class='inplay'><strong>{m['home_team']} {m['home_score']}–{m['away_score']} {m['away_team']}</strong><br>{m['minute']} • {m['league']}</div>", unsafe_allow_html=True)
        for m in upcoming[:5]:
            st.markdown(f"<div class='upcoming'><strong>{m['home_team']} vs {m['away_team']}</strong><br>{m['match_time']} • {m['league']}</div>", unsafe_allow_html=True)

    # Export
    if matches:
        df = pd.DataFrame(matches)
        # Add value edges
        edges = {f"edge_{k.split('_')[-1]}": None for k in ['home', 'draw', 'away', 'over_2_5', 'under_2_5']}
        for vb in value_bets:
            bet_type = vb['bet'].lower().replace(' ', '_')
            df.loc[df['home_team'] + ' vs ' + df['away_team'] == vb['match'], f'edge_{bet_type}'] = vb['edge']
        csv = df.to_csv(index=False).encode()
        st.download_button("Download Full Data + Edges", csv, "flashscore_value_bets.csv", "text/csv")

if __name__ == "__main__":
    main()
