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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================
# CHROMIUM BINARY DETECTION
# This is the KEY fix for "cannot find Chrome binary"
# =====================================================

@st.cache_resource
def get_chromium_info():
    """
    Detect system-installed Chromium and chromedriver.
    On Streamlit Cloud, these come from packages.txt.
    On local, they come from your system install.
    """
    info = {
        "chromium_path": None,
        "chromedriver_path": None,
        "chromium_version": None,
        "chromedriver_version": None,
        "is_cloud": os.path.exists("/home/appuser"),
    }

    # --- Find Chromium binary ---
    # Try shutil.which first (respects PATH)
    for name in ["chromium", "chromium-browser", "google-chrome", "google-chrome-stable"]:
        path = shutil.which(name)
        if path:
            info["chromium_path"] = path
            break

    # Fallback: check fixed paths (Debian/Ubuntu)
    if not info["chromium_path"]:
        for path in [
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/snap/bin/chromium",
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
        ]:
            if os.path.isfile(path):
                info["chromium_path"] = path
                break

    # --- Find chromedriver binary ---
    info["chromedriver_path"] = shutil.which("chromedriver")
    if not info["chromedriver_path"]:
        for path in [
            "/usr/bin/chromedriver",
            "/usr/lib/chromium/chromedriver",
            "/usr/lib/chromium-browser/chromedriver",
        ]:
            if os.path.isfile(path):
                info["chromedriver_path"] = path
                break

    # --- Get versions (for debug logging) ---
    if info["chromium_path"]:
        try:
            result = subprocess.run(
                [info["chromium_path"], "--version"],
                capture_output=True, text=True, timeout=10
            )
            info["chromium_version"] = result.stdout.strip()
        except Exception:
            pass

    if info["chromedriver_path"]:
        try:
            result = subprocess.run(
                [info["chromedriver_path"], "--version"],
                capture_output=True, text=True, timeout=10
            )
            info["chromedriver_version"] = result.stdout.strip()
        except Exception:
            pass

    return info


def create_driver():
    """
    Create a headless Chromium driver.
    - Uses SYSTEM-INSTALLED chromium + chromedriver (from packages.txt)
    - Explicitly sets binary_location (fixes "cannot find Chrome binary")
    - Uses correct headless flags for container environments
    """
    info = get_chromium_info()

    # ---- Validate that both binaries exist ----
    if not info["chromium_path"]:
        raise FileNotFoundError(
            "Chromium binary NOT found!\n"
            "Make sure packages.txt contains:\n"
            "  chromium\n"
            "  chromium-driver\n"
            "And that it is in the ROOT of your GitHub repo."
        )
    if not info["chromedriver_path"]:
        raise FileNotFoundError(
            "chromedriver binary NOT found!\n"
            "Make sure packages.txt contains:\n"
            "  chromium\n"
            "  chromium-driver\n"
            "And that it is in the ROOT of your GitHub repo."
        )

    options = Options()

    # ========================================================
    # THIS IS THE KEY LINE — tells Selenium WHERE Chromium is
    # Without this, you get "cannot find Chrome binary"
    # ========================================================
    options.binary_location = info["chromium_path"]

    # ---- Mandatory headless flags for Streamlit Cloud ----
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # ---- Stability flags for container / low-resource environments ----
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-default-apps")
    options.add_argument("--disable-sync")
    options.add_argument("--disable-translate")
    options.add_argument("--single-process")
    options.add_argument("--no-zygote")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--log-level=3")
    options.add_argument("--silent")

    # ---- Realistic user agent ----
    options.add_argument(
        "user-agent=Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    # ---- Create service pointing to the FOUND chromedriver ----
    service = Service(executable_path=info["chromedriver_path"])

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
    name = re.sub(r"\s*\d+$", "", name)
    name = name.replace(".", "").replace(",", "").replace("-", " ")
    return " ".join(name.split())


def find_player(name: str, all_players: list):
    name_norm = normalize_name(name)
    tokens = name_norm.split()
    if not tokens:
        return None

    # Exact
    for p in all_players:
        if normalize_name(p) == name_norm:
            return p

    # Last-name
    last_name = tokens[-1]
    candidates = [p for p in all_players if normalize_name(p).split()[-1] == last_name]
    if len(candidates) == 1:
        return candidates[0]

    # First+last combo
    if len(tokens) >= 2:
        first = tokens[0]
        for p in all_players:
            p_tokens = normalize_name(p).split()
            if len(p_tokens) >= 2 and p_tokens[0] == first and p_tokens[-1] == last_name:
                return p

    # Partial last-name
    for p in all_players:
        if last_name in normalize_name(p).split():
            return p

    return None


# =====================================================
# LOAD PLAYERS
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
# DUMMY MODEL — REPLACE WITH YOUR TRAINED MODEL!
# =====================================================

class DummyModel:
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
        "dummy_strength": 0.5,
    }
    return pd.Series(features)


# =====================================================
# FLASHSCORE SCRAPER (Selenium + System Chromium)
# =====================================================

def scrape_flashscore_tennis():
    logs = []
    matches = []
    driver = None

    try:
        info = get_chromium_info()
        logs.append(f"🖥️  Cloud: {info['is_cloud']}")
        logs.append(f"🌐 Chromium: {info['chromium_path'] or '❌ NOT FOUND'}")
        logs.append(f"🔧 Driver:   {info['chromedriver_path'] or '❌ NOT FOUND'}")
        if info["chromium_version"]:
            logs.append(f"📋 Version:  {info['chromium_version']}")
        if info["chromedriver_version"]:
            logs.append(f"📋 Driver:   {info['chromedriver_version']}")

        logs.append("⏳ Starting headless Chromium...")
        driver = create_driver()
        logs.append("✅ Chromium started successfully!")

        logs.append("🌐 Loading FlashScore tennis...")
        driver.get("https://www.flashscore.com/tennis/")

        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".event__match"))
            )
            logs.append("✅ Match data detected")
        except Exception as e:
            page_len = len(driver.page_source)
            logs.append(f"⏰ Timeout: {e}")
            logs.append(f"📄 Page source: {page_len} chars")
            if page_len < 5000:
                logs.append("⚠️  Likely GeoIP blocked. Try a proxy or API approach.")
            return matches, logs

        time.sleep(4)

        elements = driver.find_elements(
            By.CSS_SELECTOR,
            ".event__header, .event__match"
        )
        logs.append(f"📋 {len(elements)} DOM elements found")

        current_tournament = "Unknown"
        skipped = set()

        for el in elements:
            class_attr = el.get_attribute("class") or ""

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

            if "event__match" in class_attr:
                try:
                    t_upper = current_tournament.upper()
                    if not any(k in t_upper for k in ["ATP", "CHALLENGER", "ITF MEN"]):
                        skipped.add(current_tournament)
                        continue

                    home_els = el.find_elements(By.CSS_SELECTOR,
                        ".event__participant--home, [class*='homeParticipant']")
                    away_els = el.find_elements(By.CSS_SELECTOR,
                        ".event__participant--away, [class*='awayParticipant']")

                    if not home_els or not away_els:
                        parts = el.find_elements(By.CSS_SELECTOR, "[class*='participant']")
                        if len(parts) >= 2:
                            home_els = [parts[0]]
                            away_els = [parts[1]]

                    if not home_els or not away_els:
                        continue

                    p1 = home_els[0].text.strip()
                    p2 = away_els[0].text.strip()

                    if not p1 or not p2 or " / " in p1 or " / " in p2:
                        continue

                    odd1, odd2 = None, None
                    odds_els = el.find_elements(By.CSS_SELECTOR,
                        ".odds__odd, [class*='odds']")
                    if len(odds_els) >= 2:
                        try:
                            t1 = odds_els[0].text.strip()
                            if t1 and t1.replace(".", "").isdigit():
                                odd1 = float(t1)
                        except (ValueError, AttributeError):
                            pass
                        try:
                            t2 = odds_els[1].text.strip()
                            if t2 and t2.replace(".", "").isdigit():
                                odd2 = float(t2)
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
                    logs.append(f"⚠️  Parse error: {e}")
                    continue

        if skipped:
            logs.append(f"⏭️  Skipped {len(skipped)} non-ATP/Challenger tournaments")
        logs.append(f"🎾 TOTAL: {len(matches)} ATP/Challenger matches")

    except FileNotFoundError as e:
        logs.append(f"❌ {e}")
    except Exception as e:
        logs.append(f"💥 Error: {type(e).__name__}: {e}")
    finally:
        if driver:
            try:
                driver.quit()
                logs.append("🔒 Chromium closed")
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

    with st.expander("🔍 Scraper Logs", expanded=True):
        for log in logs:
            st.text(log)

    if not matches:
        st.error("❌ No matches found today.")
        st.info(
            "💡 **Possible reasons:**\n"
            "- No ATP/Challenger matches today\n"
            "- FlashScore GeoIP blocking Streamlit Cloud IPs\n"
            "- Page structure changed\n\n"
            "**Workaround:** Use a sports data API."
        )
        return

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

    with st.expander("🔗 Player Matching", expanded=False):
        matched = sum(1 for l in match_logs if l.startswith("✅"))
        failed = sum(1 for l in match_logs if l.startswith("❌"))
        st.markdown(f"**Matched:** {matched} | **Failed:** {failed}")
        for log in match_logs:
            st.text(log)

    if not results:
        st.warning("⚠️ Matches found but none matched your player history.")
        return

    df_res = pd.DataFrame(results)
    st.subheader("📊 Today's Predictions")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Matches", len(df_res))
    col2.metric("Value Bets", len(df_res[df_res["VALUE BET"] == "✅ YES"]))
    col3.metric("Avg Prob P1", f"{df_res['Prob P1 (%)'].mean():.1f}%")
    col4.metric("Tournaments", df_res["Tournament"].nunique())

    st.dataframe(df_res, use_container_width=True)

    excel_file = export_to_excel(df_res)
    st.download_button(
        label="📥 Download Predictions (Excel)",
        data=excel_file,
        file_name="predictions_flashscore.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# =====================================================
# MAIN
# =====================================================

def main():
    st.set_page_config(page_title="🎾 Tennis Predictions", page_icon="🎾", layout="wide")
    st.title("🎾 ATP / Challenger — Tennis Predictions")

    # ---- Sidebar: System diagnostics ----
    with st.sidebar:
        st.header("⚙️ System Status")
        info = get_chromium_info()

        if info["chromium_path"] and info["chromedriver_path"]:
            st.success("✅ Chromium + ChromeDriver FOUND")
            with st.expander("Binary Details"):
                st.code(
                    f"Cloud:       {info['is_cloud']}\n"
                    f"Chromium:    {info['chromium_path']}\n"
                    f"ChromeDriver:{info['chromedriver_path']}\n"
                    f"Version:     {info['chromium_version'] or 'N/A'}\n"
                    f"Driver ver:  {info['chromedriver_version'] or 'N/A'}"
                )
        else:
            st.error("❌ Chromium/ChromeDriver NOT FOUND")
            if not info["chromium_path"]:
                st.warning("Chromium binary missing")
            if not info["chromedriver_path"]:
                st.warning("ChromeDriver binary missing")
            st.markdown(
                "**Fix:** Make sure `packages.txt` in your repo ROOT contains:\n"
                "```\nchromium\nchromium-driver\n```"
            )

        st.divider()
        st.header("📂 Player History")
        uploaded_file = st.file_uploader(
            "Upload Excel (.xlsx) with winner_name / loser_name",
            type=["xlsx"],
        )

    # ---- Test Chromium button (works without Excel) ----
    if uploaded_file is None:
        st.info(
            "⬆️ **Upload an Excel file** in the sidebar.\n\n"
            "Required columns: `winner_name`, `loser_name`"
        )
        st.divider()
        if st.button("🧪 Test Chromium (no predictions)"):
            with st.spinner("Testing headless Chromium..."):
                try:
                    driver = create_driver()
                    driver.get("https://www.google.com")
                    time.sleep(2)
                    title = driver.title
                    src_len = len(driver.page_source)
                    driver.quit()
                    st.success(
                        f"✅ Chromium works!\n\n"
                        f"- Page title: `{title}`\n"
                        f"- HTML size: `{src_len}` chars"
                    )
                except Exception as e:
                    st.error(f"❌ Chromium FAILED: `{type(e).__name__}: {e}`")
        return

    all_players = load_players_from_excel(uploaded_file)
    if not all_players:
        return

    st.sidebar.success(f"✅ {len(all_players)} players loaded")

    if st.button("🔍 Fetch today's matches & predict", type="primary"):
        with st.spinner("🌐 Scraping FlashScore with Chromium..."):
            run_predictions(all_players)


if __name__ == "__main__":
    main()
