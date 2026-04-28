import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import time
import re
import shutil
import subprocess

# =====================================================
# Selenium + Chromium Driver Setup
# =====================================================
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromiumService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# webdriver_manager handles Chromium driver download automatically
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType


# ---------- NORMALIZATION ----------
def normalize_name(name: str) -> str:
    """Normalize player name for matching."""
    if not name:
        return ""
    name = name.lower().strip()
    # Remove common suffixes like (1), [5], nationality codes
    name = re.sub(r"\s*[\(\[\{].*?[\)\]\}]", "", name)
    name = name.replace(".", "").replace(",", "").replace("-", " ")
    return " ".join(name.split())


def find_player(name: str, all_players: list) -> str | None:
    """
    Match a scraped name against the known player list.
    Strategy: exact → last-name match → first+last combo.
    """
    name_norm = normalize_name(name)
    tokens = name_norm.split()

    if not tokens:
        return None

    # 1) Exact match
    for p in all_players:
        if normalize_name(p) == name_norm:
            return p

    # 2) Last-name match (last token)
    last_name = tokens[-1]
    candidates = [
        p for p in all_players
        if normalize_name(p).split()[-1] == last_name
    ]
    if len(candidates) == 1:
        return candidates[0]

    # 3) First-token + last-token combo
    if len(tokens) >= 2:
        first = tokens[0]
        for p in all_players:
            p_tokens = normalize_name(p).split()
            if len(p_tokens) >= 2 and p_tokens[0] == first and p_tokens[-1] == last_name:
                return p

    return None


# ---------- LOAD PLAYERS ----------
@st.cache_data
def load_players_from_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)
    required = {"winner_name", "loser_name"}
    if not required.issubset(df.columns):
        st.error("Excel must contain columns 'winner_name' and 'loser_name'.")
        return []
    winners = df["winner_name"].dropna().unique().tolist()
    losers = df["loser_name"].dropna().unique().tolist()
    return sorted(set(winners + losers))


# ---------- DUMMY MODEL (replace with real trained model!) ----------
class DummyModel:
    """Always predicts 50/50 — REPLACE with your trained model."""
    def predict_proba(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.tile(np.array([[0.5, 0.5]]), (X.shape[0], 1))


model = DummyModel()


# ---------- INFER SURFACE FROM TOURNAMENT NAME ----------
SURFACE_KEYWORDS = {
    "clay": [
        "roland garros", "rome", "madrid", "barcelona", "monte carlo",
        "buenos aires", "rio", "lyon", "hamburg", "kitzbuhel",
        "bastad", "umag", "gstaad", "bucharest", "marrakech", "clay",
    ],
    "grass": [
        "wimbledon", "halle", "queens", "stuttgart", "eastbourne",
        "mallorca", "s-hertogenbosch", "grass",
    ],
}


def infer_surface(tournament_name: str) -> str:
    """Guess surface from tournament name. Default = Hard."""
    t = tournament_name.lower()
    for surface, keywords in SURFACE_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return surface.capitalize()
    return "Hard"


# ---------- MATCH SUMMARY (feature vector) ----------
def build_match_summary(player1, player2, surface):
    features = {
        "surface_hard": 1 if surface.lower() == "hard" else 0,
        "surface_clay": 1 if surface.lower() == "clay" else 0,
        "surface_grass": 1 if surface.lower() == "grass" else 0,
        "dummy_strength": 0.5,  # TODO: replace with real ELO / stats
    }
    return pd.Series(features)


# =====================================================
# CHROMIUM DRIVER — works locally AND on Streamlit Cloud
# =====================================================
def _find_chromium_binary() -> str | None:
    """
    Locate the Chromium browser binary on the system.
    Checks common names across Linux, macOS, and Windows.
    """
    # Common binary names across platforms
    candidates = [
        "chromium-browser",   # Debian/Ubuntu apt package
        "chromium",           # Arch, Alpine, snap, macOS brew
        "google-chrome",      # If Chrome is installed instead
        "google-chrome-stable",
    ]
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path

    # Fallback: check common fixed paths
    import os
    fixed_paths = [
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
        "/snap/bin/chromium",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
    ]
    for p in fixed_paths:
        if os.path.isfile(p):
            return p

    return None


def get_chromium_driver():
    """
    Create a headless Chromium WebDriver instance.
    Uses webdriver_manager with ChromeType.CHROMIUM for automatic
    driver binary management.
    """
    options = Options()

    # ---- Headless & stability flags ----
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    # ---- Point to Chromium binary if not in default Chrome location ----
    chromium_path = _find_chromium_binary()
    if chromium_path:
        options.binary_location = chromium_path

    # ---- Let webdriver_manager fetch the correct chromedriver for Chromium ----
    service = ChromiumService(
        ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()
    )

    driver = webdriver.Chrome(service=service, options=options)
    return driver


# =====================================================
# FLASHSCORE SCRAPER (Selenium + Chromium)
# =====================================================
def scrape_flashscore_tennis():
    """
    Scrape today's tennis matches from FlashScore using headless Chromium.
    Returns (list[dict], list[str]) = (matches, logs).
    """
    logs = ["📅 Searching today's matches on FlashScore (tennis)..."]
    matches = []
    driver = None

    try:
        driver = get_chromium_driver()
        logs.append("✅ Chromium driver initialized successfully")

        driver.get("https://www.flashscore.com/tennis/")
        logs.append("🌐 Page loaded, waiting for match data to render...")

        # Wait for match rows to appear
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".event__match"))
        )
        # Extra wait for all JS to finish rendering
        time.sleep(4)

        # --- Extract tournament headers and match rows ---
        elements = driver.find_elements(
            By.CSS_SELECTOR,
            ".event__header, .event__match"
        )

        current_tournament = "Unknown"
        logs.append(f"📋 Found {len(elements)} DOM elements to parse")

        for el in elements:
            class_attr = el.get_attribute("class") or ""

            # ---- Tournament header ----
            if "event__header" in class_attr:
                try:
                    cat_el = el.find_elements(By.CSS_SELECTOR, ".event__title--type")
                    name_el = el.find_elements(By.CSS_SELECTOR, ".event__title--name")
                    cat = cat_el[0].text.strip() if cat_el else ""
                    name = name_el[0].text.strip() if name_el else ""
                    current_tournament = f"{cat} {name}".strip()
                except Exception:
                    pass
                continue

            # ---- Match row ----
            if "event__match" in class_attr:
                try:
                    # Filter: only ATP / Challenger / ITF Men
                    tournament_upper = current_tournament.upper()
                    if not any(k in tournament_upper for k in [
                        "ATP", "CHALLENGER", "ITF MEN"
                    ]):
                        continue

                    # Players
                    participants = el.find_elements(
                        By.CSS_SELECTOR,
                        "[class*='participant']"
                    )
                    if len(participants) < 2:
                        continue
                    p1 = participants[0].text.strip()
                    p2 = participants[1].text.strip()

                    if not p1 or not p2:
                        continue

                    # Odds (FlashScore sometimes shows inline odds)
                    odds_els = el.find_elements(
                        By.CSS_SELECTOR,
                        "[class*='odds'], .odds__odd"
                    )
                    odd1 = None
                    odd2 = None
                    if len(odds_els) >= 2:
                        try:
                            odd1 = float(odds_els[0].text.strip())
                        except (ValueError, AttributeError):
                            pass
                        try:
                            odd2 = float(odds_els[1].text.strip())
                        except (ValueError, AttributeError):
                            pass

                    surface = infer_surface(current_tournament)

                    matches.append({
                        "tournament": current_tournament,
                        "player1": p1,
                        "player2": p2,
                        "surface": surface,
                        "odd1": odd1,
                        "odd2": odd2,
                    })

                except Exception as e:
                    logs.append(f"⚠️  Error parsing match row: {e}")
                    continue

        logs.append(f"🎾 TOTAL: {len(matches)} ATP/Challenger matches found")

    except Exception as e:
        logs.append(f"💥 Chromium scraper error: {e}")

    finally:
        if driver:
            driver.quit()
            logs.append("🔒 Chromium driver closed")

    return matches, logs


# ---------- EXPORT EXCEL ----------
def export_daily_predictions_to_excel(df: pd.DataFrame) -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    output.seek(0)
    return output


# ---------- PREPARE & PREDICT ----------
def prepare_match_for_prediction(match, all_players):
    p1_hist = find_player(match["player1"], all_players)
    p2_hist = find_player(match["player2"], all_players)

    if not p1_hist or not p2_hist:
        missing = []
        if not p1_hist:
            missing.append(match["player1"])
        if not p2_hist:
            missing.append(match["player2"])
        return None, f"❌ Not in history: {', '.join(missing)}"

    summary = build_match_summary(p1_hist, p2_hist, match["surface"])
    return summary, f"✅ Matched: {match['player1']} vs {match['player2']} ({match['surface']})"


def run_daily_predictions(all_players):
    matches, logs = scrape_flashscore_tennis()

    st.subheader("🔍 Scraper Logs")
    for log in logs:
        st.text(log)

    if not matches:
        st.error("❌ No matches found today.")
        return

    resultados = []

    for match in matches:
        summary, log_match = prepare_match_for_prediction(match, all_players)
        st.text(log_match)

        if summary is None:
            continue

        X = summary.values.reshape(1, -1)
        prob = model.predict_proba(X)[0][1]

        odd1 = match["odd1"]
        odd2 = match["odd2"]
        ev1 = (prob * odd1 - 1) if odd1 else None
        ev2 = ((1 - prob) * odd2 - 1) if odd2 else None
        value_bet = "✅ YES" if ev1 and ev1 > 0 else "❌ NO"

        resultados.append({
            "Tournament": match["tournament"],
            "Surface": match["surface"],
            "Player 1": match["player1"],
            "Player 2": match["player2"],
            "Odd P1": odd1,
            "Odd P2": odd2,
            "Prob P1": round(prob, 4),
            "EV P1": round(ev1, 4) if ev1 is not None else None,
            "EV P2": round(ev2, 4) if ev2 is not None else None,
            "VALUE BET P1": value_bet,
        })

    if not resultados:
        st.warning("⚠️  Matches found but none matched your player history.")
        return

    df_res = pd.DataFrame(resultados)

    st.subheader("📊 Today's Predictions")
    st.dataframe(df_res, use_container_width=True)

    # Stats row
    col1, col2, col3 = st.columns(3)
    col1.metric("Matches Analyzed", len(df_res))
    col2.metric("Value Bets", len(df_res[df_res["VALUE BET P1"] == "✅ YES"]))
    col3.metric("Avg Prob P1", f"{df_res['Prob P1'].mean():.2%}")

    excel_file = export_daily_predictions_to_excel(df_res)
    st.download_button(
        label="📥 Download Predictions (Excel)",
        data=excel_file,
        file_name="predictions_flashscore.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ---------- MAIN ----------
def main():
    st.set_page_config(page_title="Tennis Predictions", layout="wide")
    st.title("🎾 ATP / Challenger — Tennis Predictions (FlashScore)")

    # Show Chromium status in sidebar
    chromium_path = _find_chromium_binary()
    if chromium_path:
        st.sidebar.success(f"🌐 Chromium found: `{chromium_path}`")
    else:
        st.sidebar.warning(
            "⚠️ Chromium not found. Install it:\n\n"
            "**Ubuntu/Debian:**\n"
            "```\nsudo apt install chromium-browser\n```\n"
            "**macOS:**\n"
            "```\nbrew install --cask chromium\n```"
        )

    uploaded_file = st.sidebar.file_uploader(
        "Upload player history (Excel .xlsx with winner_name / loser_name)",
        type=["xlsx"],
    )

    if uploaded_file is None:
        st.warning("⬆️ Upload an Excel file with match history to get started.")
        return

    all_players = load_players_from_excel(uploaded_file)
    if not all_players:
        return

    st.sidebar.success(f"✅ {len(all_players)} players loaded.")

    if st.button("🔍 Fetch today's matches & predict"):
        with st.spinner("Starting Chromium & scraping FlashScore..."):
            run_daily_predictions(all_players)


if __name__ == "__main__":
    main()
