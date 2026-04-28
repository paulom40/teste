import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import time
import re
import os
import shutil
import subprocess
import logging

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================
# CHROMIUM DRIVER — Streamlit Cloud Compatible
# Uses SYSTEM-INSTALLED chromium & chromedriver
# via packages.txt (NO webdriver-manager needed)
# =====================================================

@st.cache_resource
def get_chromium_info():
    """
    Detect system-installed Chromium and chromedriver paths.
    On Streamlit Cloud (Debian), these are installed via packages.txt.
    Returns dict with paths and version info.
    """
    info = {
        "chromium_path": None,
        "chromedriver_path": None,
        "chromium_version": None,
        "chromedriver_version": None,
        "platform": "unknown",
    }

    # --- Find Chromium binary ---
    chromium_candidates = [
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/snap/bin/chromium",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
    ]
    for path in chromium_candidates:
        if os.path.isfile(path):
            info["chromium_path"] = path
            break
    if not info["chromium_path"]:
        info["chromium_path"] = shutil.which("chromium") or shutil.which("chromium-browser")

    # --- Find chromedriver binary ---
    chromedriver_candidates = [
        "/usr/bin/chromedriver",
        "/usr/lib/chromium/chromedriver",
        "/usr/lib/chromium-browser/chromedriver",
        "/snap/bin/chromium.chromedriver",
    ]
    for path in chromedriver_candidates:
        if os.path.isfile(path):
            info["chromedriver_path"] = path
            break
    if not info["chromedriver_path"]:
        info["chromedriver_path"] = shutil.which("chromedriver")

    # --- Get versions ---
    try:
        if info["chromium_path"]:
            result = subprocess.run(
                [info["chromium_path"], "--version"],
                capture_output=True, text=True, timeout=10
            )
            info["chromium_version"] = result.stdout.strip()
    except Exception:
        pass

    try:
        if info["chromedriver_path"]:
            result = subprocess.run(
                [info["chromedriver_path"], "--version"],
                capture_output=True, text=True, timeout=10
            )
            info["chromedriver_version"] = result.stdout.strip()
    except Exception:
        pass

    # Detect platform
    if os.path.exists("/home/appuser"):
        info["platform"] = "streamlit_cloud"
    else:
        info["platform"] = "local"

    return info


def create_driver():
    """
    Create a headless Chromium WebDriver.
    Uses system-installed binaries (via packages.txt on Streamlit Cloud).
    Fully compatible with Streamlit Cloud Debian runtime.
    """
    chromium_info = get_chromium_info()

    if not chromium_info["chromedriver_path"]:
        raise FileNotFoundError(
            "chromedriver not found! Make sure packages.txt contains:\n"
            "chromium\nchromium-driver"
        )

    options = Options()

    # --- Point to system Chromium binary ---
    if chromium_info["chromium_path"]:
        options.binary_location = chromium_info["chromium_path"]

    # --- Mandatory headless flags for Streamlit Cloud ---
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # --- Stability flags for container environments ---
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-default-apps")
    options.add_argument("--disable-sync")
    options.add_argument("--disable-translate")
    options.add_argument("--disable-features=VizDisplayCompositor")
    options.add_argument("--single-process")
    options.add_argument("--no-zygote")

    # --- Memory optimization (Streamlit Cloud has limited RAM) ---
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--window-size=1920,1080")

    # --- Realistic user agent ---
    options.add_argument(
        "user-agent=Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    # --- Logging ---
    options.add_argument("--log-level=3")
    options.add_argument("--silent")

    # --- Use system-installed chromedriver ---
    service = Service(executable_path=chromium_info["chromedriver_path"])

    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(30)
    driver.implicitly_wait(5)

    return driver


# =====================================================
# NAME NORMALIZATION & PLAYER MATCHING
# =====================================================

def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower().strip()
    name = re.sub(r"\s*[\(\[\{].*?[\)\]\}]", "", name)
    name = re.sub(r"\s*\d+$", "", name)  # remove trailing seed numbers
    name = name.replace(".", "").replace(",", "").replace("-", " ")
    return " ".join(name.split())


def find_player(name: str, all_players: list) -> str | None:
    name_norm = normalize_name(name)
    tokens = name_norm.split()
    if not tokens:
        return None

    # 1) Exact match
    for p in all_players:
        if normalize_name(p) == name_norm:
            return p

    # 2) Last-name match
    last_name = tokens[-1]
    candidates = [p for p in all_players if normalize_name(p).split()[-1] == last_name]
    if len(candidates) == 1:
        return candidates[0]

    # 3) First + last combo
    if len(tokens) >= 2:
        first = tokens[0]
        for p in all_players:
            p_tokens = normalize_name(p).split()
            if len(p_tokens) >= 2 and p_tokens[0] == first and p_tokens[-1] == last_name:
                return p

    # 4) Partial last-name containment
    for p in all_players:
        p_norm = normalize_name(p)
        if last_name in p_norm.split():
            return p

    return None


# =====================================================
# LOAD PLAYERS FROM EXCEL
# =====================================================

@st.cache_data
def load_players_from_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)
    required = {"winner_name", "loser_name"}
    if not required.issubset(df.columns):
        st.error("Excel must contain columns: 'winner_name' and 'loser_name'")
        return []
    winners = df["winner_name"].dropna().unique().tolist()
    losers = df["loser_name"].dropna().unique().tolist()
    return sorted(set(winners + losers))


# =====================================================
# DUMMY MODEL — REPLACE WITH YOUR TRAINED MODEL
# =====================================================

class DummyModel:
    """Always returns 50/50. Replace with your real model!"""
    def predict_proba(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.tile(np.array([[0.5, 0.5]]), (X.shape[0], 1))


model = DummyModel()


# =====================================================
# SURFACE INFERENCE
# =====================================================

SURFACE_MAP = {
    "clay": [
        "roland garros", "rome", "madrid", "barcelona", "monte carlo",
        "buenos aires", "rio", "lyon", "hamburg", "kitzbuhel",
        "bastad", "umag", "gstaad", "bucharest", "marrakech",
        "geneva", "parma", "cagliari", "belgrade", "clay",
    ],
    "grass": [
        "wimbledon", "halle", "queens", "stuttgart", "eastbourne",
        "mallorca", "s-hertogenbosch", "newport", "grass",
    ],
}


def infer_surface(tournament: str) -> str:
    t = tournament.lower()
    for surface, keywords in SURFACE_MAP.items():
        if any(kw in t for kw in keywords):
            return surface.capitalize()
    return "Hard"


# =====================================================
# FEATURE BUILDER
# =====================================================

def build_match_summary(player1, player2, surface):
    features = {
        "surface_hard": 1 if surface.lower() == "hard" else 0,
        "surface_clay": 1 if surface.lower() == "clay" else 0,
        "surface_grass": 1 if surface.lower() == "grass" else 0,
        "dummy_strength": 0.5,  # TODO: replace with ELO, H2H stats, etc.
    }
    return pd.Series(features)


# =====================================================
# FLASHSCORE SCRAPER (Selenium + System Chromium)
# =====================================================

def scrape_flashscore_tennis():
    """
    Scrape today's ATP/Challenger tennis matches from FlashScore.
    Uses system-installed Chromium (headless) via Selenium.
    Returns (matches: list[dict], logs: list[str])
    """
    logs = []
    matches = []
    driver = None

    try:
        # ---- Show Chromium info ----
        info = get_chromium_info()
        logs.append(f"🖥️  Platform: {info['platform']}")
        logs.append(f"🌐 Chromium: {info['chromium_path']}")
        logs.append(f"🔧 ChromeDriver: {info['chromedriver_path']}")
        if info["chromium_version"]:
            logs.append(f"📋 Chromium version: {info['chromium_version']}")
        if info["chromedriver_version"]:
            logs.append(f"📋 ChromeDriver version: {info['chromedriver_version']}")

        # ---- Create driver ----
        logs.append("⏳ Starting headless Chromium...")
        driver = create_driver()
        logs.append("✅ Chromium driver created successfully")

        # ---- Load FlashScore ----
        logs.append("🌐 Loading FlashScore tennis page...")
        driver.get("https://www.flashscore.com/tennis/")

        # ---- Wait for content ----
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".event__match"))
            )
            logs.append("✅ Match data detected on page")
        except Exception as e:
            logs.append(f"⏰ Timeout waiting for matches: {e}")
            # Try alternate: maybe the page loaded but with different structure
            page_title = driver.title
            logs.append(f"📄 Page title: {page_title}")
            page_len = len(driver.page_source)
            logs.append(f"📄 Page source length: {page_len} chars")

            # Check if we got a 403 / blocked page
            if page_len < 5000:
                logs.append("⚠️  Page seems blocked (very short HTML). "
                            "FlashScore may use GeoIP blocking on Streamlit Cloud IPs.")
                logs.append("💡 TIP: Consider using a proxy or an API-based approach.")
            return matches, logs

        # Extra wait for JS rendering
        time.sleep(4)

        # ---- Parse elements ----
        elements = driver.find_elements(
            By.CSS_SELECTOR,
            ".event__header, .event__match"
        )
        logs.append(f"📋 Found {len(elements)} DOM elements to parse")

        current_tournament = "Unknown"
        skipped_tournaments = set()

        for el in elements:
            class_attr = el.get_attribute("class") or ""

            # ---- Tournament header ----
            if "event__header" in class_attr:
                try:
                    cat_els = el.find_elements(By.CSS_SELECTOR, ".event__title--type")
                    name_els = el.find_elements(By.CSS_SELECTOR, ".event__title--name")
                    cat = cat_els[0].text.strip() if cat_els else ""
                    name = name_els[0].text.strip() if name_els else ""
                    current_tournament = f"{cat} {name}".strip() if (cat or name) else "Unknown"
                except Exception:
                    pass
                continue

            # ---- Match row ----
            if "event__match" in class_attr:
                try:
                    tournament_upper = current_tournament.upper()

                    # Filter: only ATP / Challenger / ITF Men
                    if not any(k in tournament_upper for k in ["ATP", "CHALLENGER", "ITF MEN"]):
                        skipped_tournaments.add(current_tournament)
                        continue

                    # ---- Extract player names ----
                    # FlashScore uses various class names for participants
                    home_els = el.find_elements(By.CSS_SELECTOR,
                        ".event__participant--home, [class*='homeParticipant']"
                    )
                    away_els = el.find_elements(By.CSS_SELECTOR,
                        ".event__participant--away, [class*='awayParticipant']"
                    )

                    # Fallback: generic participant selector
                    if not home_els or not away_els:
                        parts = el.find_elements(By.CSS_SELECTOR, "[class*='participant']")
                        if len(parts) >= 2:
                            home_els = [parts[0]]
                            away_els = [parts[1]]

                    if not home_els or not away_els:
                        continue

                    p1 = home_els[0].text.strip()
                    p2 = away_els[0].text.strip()

                    if not p1 or not p2:
                        continue

                    # Skip doubles matches (contain " / " in name)
                    if " / " in p1 or " / " in p2:
                        continue

                    # ---- Extract odds (if visible) ----
                    odd1, odd2 = None, None
                    odds_els = el.find_elements(By.CSS_SELECTOR,
                        ".odds__odd, [class*='odds'], [class*='bookmaker']"
                    )
                    if len(odds_els) >= 2:
                        try:
                            text1 = odds_els[0].text.strip()
                            if text1 and text1.replace(".", "").isdigit():
                                odd1 = float(text1)
                        except (ValueError, AttributeError):
                            pass
                        try:
                            text2 = odds_els[1].text.strip()
                            if text2 and text2.replace(".", "").isdigit():
                                odd2 = float(text2)
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
                    logs.append(f"⚠️  Error parsing match: {e}")
                    continue

        if skipped_tournaments:
            logs.append(f"⏭️  Skipped {len(skipped_tournaments)} non-ATP/Challenger tournaments")

        logs.append(f"🎾 TOTAL: {len(matches)} ATP/Challenger matches found")

    except FileNotFoundError as e:
        logs.append(f"❌ {e}")
    except Exception as e:
        logs.append(f"💥 Scraper error: {type(e).__name__}: {e}")
    finally:
        if driver:
            try:
                driver.quit()
                logs.append("🔒 Chromium driver closed")
            except Exception:
                pass

    return matches, logs


# =====================================================
# EXCEL EXPORT
# =====================================================

def export_to_excel(df: pd.DataFrame) -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    output.seek(0)
    return output


# =====================================================
# PREDICTION PIPELINE
# =====================================================

def prepare_match(match, all_players):
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
    return summary, f"✅ {match['player1']} vs {match['player2']} ({match['surface']})"


def run_predictions(all_players):
    matches, logs = scrape_flashscore_tennis()

    # ---- Display scraper logs ----
    with st.expander("🔍 Scraper Logs", expanded=True):
        for log in logs:
            st.text(log)

    if not matches:
        st.error("❌ No matches found today.")
        st.info(
            "💡 **Possible reasons:**\n"
            "- No ATP/Challenger matches scheduled today\n"
            "- FlashScore may block Streamlit Cloud IPs (GeoIP)\n"
            "- Page structure may have changed\n\n"
            "**Workaround:** Try using a sports data API instead."
        )
        return

    # ---- Process matches ----
    results = []
    match_logs = []

    for match in matches:
        summary, log_msg = prepare_match(match, all_players)
        match_logs.append(log_msg)

        if summary is None:
            continue

        X = summary.values.reshape(1, -1)
        prob = model.predict_proba(X)[0][1]

        odd1 = match["odd1"]
        odd2 = match["odd2"]
        ev1 = (prob * odd1 - 1) if odd1 else None
        ev2 = ((1 - prob) * odd2 - 1) if odd2 else None
        value_bet = "✅ YES" if ev1 and ev1 > 0 else "❌ NO"

        results.append({
            "Tournament": match["tournament"],
            "Surface": match["surface"],
            "Player 1": match["player1"],
            "Player 2": match["player2"],
            "Odd P1": odd1,
            "Odd P2": odd2,
            "Prob P1 (%)": round(prob * 100, 2),
            "EV P1": round(ev1, 4) if ev1 is not None else None,
            "EV P2": round(ev2, 4) if ev2 is not None else None,
            "VALUE BET": value_bet,
        })

    # ---- Show matching logs ----
    with st.expander("🔗 Player Matching Logs", expanded=False):
        matched = sum(1 for l in match_logs if l.startswith("✅"))
        failed = sum(1 for l in match_logs if l.startswith("❌"))
        st.markdown(f"**Matched:** {matched} | **Failed:** {failed}")
        for log in match_logs:
            st.text(log)

    if not results:
        st.warning("⚠️ Matches found but none matched your player history.")
        return

    # ---- Display results ----
    df_res = pd.DataFrame(results)

    st.subheader("📊 Today's Predictions")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Matches", len(df_res))
    col2.metric("Value Bets", len(df_res[df_res["VALUE BET"] == "✅ YES"]))
    col3.metric("Avg Prob P1", f"{df_res['Prob P1 (%)'].mean():.1f}%")
    tournaments = df_res["Tournament"].nunique()
    col4.metric("Tournaments", tournaments)

    # Color-coded table
    st.dataframe(
        df_res.style.applymap(
            lambda v: "background-color: #d4edda" if v == "✅ YES" else
                      "background-color: #f8d7da" if v == "❌ NO" else "",
            subset=["VALUE BET"]
        ),
        use_container_width=True,
        height=min(len(df_res) * 40 + 50, 600),
    )

    # Excel download
    excel_file = export_to_excel(df_res)
    st.download_button(
        label="📥 Download Predictions (Excel)",
        data=excel_file,
        file_name="predictions_flashscore.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# =====================================================
# MAIN APP
# =====================================================

def main():
    st.set_page_config(
        page_title="🎾 Tennis Predictions",
        page_icon="🎾",
        layout="wide",
    )

    st.title("🎾 ATP / Challenger — Tennis Predictions")
    st.caption("Scrapes FlashScore with headless Chromium on Streamlit Cloud")

    # ---- Sidebar: System Info ----
    with st.sidebar:
        st.header("⚙️ System Status")
        info = get_chromium_info()

        if info["chromium_path"] and info["chromedriver_path"]:
            st.success("✅ Chromium + ChromeDriver found")
            with st.expander("Details"):
                st.code(f"""
Platform:    {info['platform']}
Chromium:    {info['chromium_path']}
ChromeDriver:{info['chromedriver_path']}
Version:     {info['chromium_version'] or 'N/A'}
Driver ver:  {info['chromedriver_version'] or 'N/A'}
""")
        else:
            st.error("❌ Chromium/ChromeDriver NOT found")
            st.markdown(
                "**To fix:** Make sure `packages.txt` in your repo root contains:\n"
                "```\nchromium\nchromium-driver\n```"
            )
            if not info["chromium_path"]:
                st.warning("Chromium binary not found")
            if not info["chromedriver_path"]:
                st.warning("ChromeDriver binary not found")

        st.divider()

        # ---- Sidebar: Upload ----
        st.header("📂 Player History")
        uploaded_file = st.file_uploader(
            "Upload Excel (.xlsx) with winner_name / loser_name",
            type=["xlsx"],
        )

    # ---- Main content ----
    if uploaded_file is None:
        st.info(
            "⬆️ **Upload an Excel file** with match history to get started.\n\n"
            "The file must contain columns:\n"
            "- `winner_name`\n"
            "- `loser_name`"
        )

        # Show a quick test button even without data
        st.divider()
        if st.button("🧪 Test Chromium (no predictions)"):
            with st.spinner("Testing Chromium driver..."):
                try:
                    driver = create_driver()
                    driver.get("https://www.flashscore.com/tennis/")
                    time.sleep(3)
                    title = driver.title
                    source_len = len(driver.page_source)
                    driver.quit()
                    st.success(f"✅ Chromium works! Page title: '{title}', HTML size: {source_len} chars")
                except Exception as e:
                    st.error(f"❌ Chromium failed: {type(e).__name__}: {e}")
        return

    # ---- Load players ----
    all_players = load_players_from_excel(uploaded_file)
    if not all_players:
        return

    st.sidebar.success(f"✅ {len(all_players)} players loaded")

    # ---- Run predictions ----
    if st.button("🔍 Fetch today's matches & predict", type="primary"):
        with st.spinner("🌐 Starting Chromium & scraping FlashScore..."):
            run_predictions(all_players)


if __name__ == "__main__":
    main()
