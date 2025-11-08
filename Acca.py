import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------
# App configuration
# -------------------------------------------------
st.set_page_config(page_title="Multi-Tipster Soccer Dashboard", layout="wide")
st.title("Multi-Tipster Soccer Dashboard – Free Verified Experts")
st.markdown("**Track Top Free Tipsters from ProTipster.com | 100% Free Tips | Updated Daily**")
st.markdown("---")

# -------------------------------------------------
# Sidebar – tipster selection
# -------------------------------------------------
st.sidebar.header("Select Tipsters")
tipsters_available = ["NorthSea", "Rush 641", "GoalMaster", "ValueHunter"]
selected_tipsters = st.sidebar.multiselect(
    "Choose tipsters to display:",
    options=tipsters_available,
    default=tipsters_available[:2]
)

# -------------------------------------------------
# Tipster data (static demo – replace with API later)
# -------------------------------------------------
tipsters_data = {
    "NorthSea": {
        "subtitle": "BTTS & Over/Under Specialist",
        "stats": {"Profit": "+£3,381", "Strike Rate": "77%", "Tips": "196"},
        "tips": {
            'Date': ['2025-10-25', '2025-10-22', '2025-10-19', '2025-10-15', '2025-10-12',
                     '2025-10-08', '2025-10-05', '2025-11-08', '2025-11-09', '2025-11-10'],
            'Match': ['Man City vs Southampton (PL)', 'PSG vs Marseille (Ligue 1)', 'Real Madrid vs Villarreal (La Liga)',
                      'Juventus vs Inter (Serie A)', 'Bayern vs Dortmund (Bundesliga)', 'Liverpool vs Chelsea (PL)',
                      'Lyon vs Monaco (Ligue 1)', 'Union Berlin vs Bayern (Bundesliga)', 'AC Milan vs Juventus (Serie A)',
                      'Barcelona vs Real Sociedad (La Liga)'],
            'Selection': ['BTTS - Yes', 'Over 2.5 Goals', 'BTTS - Yes', 'Under 2.5 Goals', 'Over 3.5 Goals',
                          'BTTS & Over 2.5', 'BTTS - Yes', 'BTTS - Yes', 'Over 2.5 Goals', 'BTTS - Yes'],
            'Market': ['BTTS', 'Over/Under', 'BTTS', 'Over/Under', 'Over/Under', 'BTTS + O/U', 'BTTS',
                       'BTTS', 'Over/Under', 'BTTS'],
            'Odds': [1.85, 1.70, 1.95, 1.90, 2.20, 2.40, 1.80, 2.10, 1.95, 1.85],
            'Outcome': ['Win', 'Win', 'Win', 'Loss', 'Win', 'Win', 'Win', 'Pending', 'Pending', 'Pending'],
            'Reasoning': [
                'Both teams scoring in 6/7 recent games; City leaky at back.',
                'Derby fireworks: Over in 8/10 h2h; high attack ratings.',
                'Madrid concede on counters; Villarreal score 70% aways.',
                'Tactical battle expected low-scoring; defied with late goals.',
                'Klassiker always goals: 5/6 Over 3.5; explosive attacks.',
                'End-to-end rivalry; BTTS in 9/10 meetings.',
                'Both leaky defenses; scoring form in 80% games.',
                'Bayern dominate but concede; Union home scorers.',
                'Milan attack firing; Juve counters—goals likely.',
                'Barca creative; Sociedad potent at home—mutual threats.'
            ]
        }
    },
    "Rush 641": {
        "subtitle": "Over/Under & Asian Handicap Expert",
        "stats": {"Profit": "+£2,697", "Strike Rate": "72%", "Tips": "89"},
        "tips": {
            'Date': ['2025-10-26', '2025-10-23', '2025-10-20', '2025-10-16', '2025-10-13',
                     '2025-10-09', '2025-10-06', '2025-11-08', '2025-11-09', '2025-11-10'],
            'Match': ['Ajax vs PSV (Eredivisie)', 'Porto vs Benfica (Primeira Liga)', 'Atalanta vs Lazio (Serie A)',
                      'Leipzig vs Stuttgart (Bundesliga)', 'Sevilla vs Valencia (La Liga)', 'Celtic vs Rangers (Scottish PL)',
                      'Monaco vs Nice (Ligue 1)', 'Union Berlin vs Bayern (Bundesliga)', 'AC Milan vs Juventus (Serie A)',
                      'Barcelona vs Real Sociedad (La Liga)'],
            'Selection': ['Over 2.5 Goals', 'Under 2.5 Goals', 'Asian Handicap -1.5 Atalanta', 'Over 3.0 Goals',
                          'Over 1.5 Goals', 'Asian Handicap 0 Rangers', 'Under 2.5 Goals', 'Over 2.5 Goals',
                          'Asian Handicap -0.5 Milan', 'Over 2.5 Goals'],
            'Market': ['Over/Under', 'Over/Under', 'Asian Handicap', 'Over/Under', 'Over/Under',
                       'Asian Handicap', 'Over/Under', 'Over/Under', 'Asian Handicap', 'Over/Under'],
            'Odds': [1.75, 1.85, 2.10, 1.95, 1.40, 1.90, 1.80, 1.70, 2.00, 1.85],
            'Outcome': ['Win', 'Win', 'Win', 'Loss', 'Win', 'Win', 'Win', 'Pending', 'Pending', 'Pending'],
            'Reasoning': [
                'Dutch top clash: Over in 7/8 h2h; both attacks firing.',
                'Tight rivalry: Under in 5/6 recent derbies; defensive setups.',
                'Atalanta home form (W4/5); Lazio poor aways, concede heavily.',
                'Expected goals fest, but low-scoring tactical battle.',
                'Both leaky: Over 1.5 in 9/10 combined games.',
                'Rangers edge in form; Celtic vulnerable at home.',
                "Cote d'Azur derby: Low goals in 6/7 h2h.",
                'Bayern firepower; Union counter threats—goals expected.',
                'Milan streak intact; Juve travel woes continue.',
                'Barca home dominance; Sociedad open playstyle.'
            ]
        }
    },
    # -----------------------------------------------------------------
    # Add GoalMaster & ValueHunter exactly as in the previous version
    # -----------------------------------------------------------------
    "GoalMaster": { ... },   # <-- paste the dict from the previous answer
    "ValueHunter": { ... }   # <-- paste the dict from the previous answer
}
# (For brevity the two extra tipsters are omitted here – copy them from the previous reply.)

# -------------------------------------------------
# Helper: styling functions
# -------------------------------------------------
def highlight_outcome(val):
    if val == 'Win':
        return 'background-color: #ccffcc; color: green; font-weight: bold'
    if val == 'Loss':
        return 'background-color: #ffcccc; color: red; font-weight: bold'
    if val == 'Pending':
        return 'background-color: #fff3cd; color: #d58b00; font-weight: bold'
    return ''

def highlight_market(val):
    if 'BTTS' in val:
        return 'background-color: #e6f7ff; font-weight: bold'
    if 'Over' in val:
        return 'background-color: #f0e6ff; font-weight: bold'
    if 'Under' in val:
        return 'background-color: #fff0f0; font-weight: bold'
    if 'Asian' in val or 'Handicap' in val:
        return 'background-color: #fff4e6; font-weight: bold'
    if '1X2' in val or 'Win' in val:
        return 'background-color: #f0fff0; font-weight: bold'
    return ''

# -------------------------------------------------
# MAIN CONTENT
# -------------------------------------------------
if not selected_tipsters:
    st.warning("Select at least one tipster from the sidebar to view their predictions!")
else:
    # ---------- Overall summary ----------
    st.header("Overall Performance Summary")
    summary_rows = []
    for tipster in selected_tipsters:
        df_tip = pd.DataFrame(tipsters_data[tipster]["tips"])
        wins = len(df_tip[df_tip['Outcome'] == 'Win'])
        resolved = len(df_tip[df_tip['Outcome'] != 'Pending'])
        strike = (wins / resolved * 100) if resolved else 0
        profit = sum((o - 1) * 10 for o in df_tip.loc[df_tip['Outcome'] == 'Win', 'Odds']) \
                 - 10 * len(df_tip[df_tip['Outcome'] == 'Loss'])
        summary_rows.append({
            "Tipster": tipster,
            "Total Tips": len(df_tip),
            "Strike Rate": f"{strike:.1f}%",
            "Est. Profit (£10 stakes)": f"£{profit:.0f}"
        })
    st.table(pd.DataFrame(summary_rows))

    # ---------- Individual tipster tabs ----------
    st.header("Individual Tipster Predictions")
    tabs = st.tabs(selected_tipsters)

    for idx, tipster in enumerate(selected_tipsters):
        with tabs[idx]:
            info = tipsters_data[tipster]
            st.subheader(f"{tipster} – {info['subtitle']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("YTD Profit", info['stats']['Profit'])
            c2.metric("Strike Rate", info['stats']['Strike Rate'])
            c3.metric("Total Tips", info['stats']['Tips'])

            df = pd.DataFrame(info["tips"])

            styled = df.style \
                .applymap(highlight_outcome, subset=['Outcome']) \
                .applymap(highlight_market, subset=['Market']) \
                .format({'Odds': '{:.2f}'})

            st.subheader("Prediction History")
            st.dataframe(styled, use_container_width=True)

            # ---- Acca builder for this tipster ----
            st.subheader("Build Acca from Future Tips")
            future = df[df['Outcome'] == 'Pending'].copy()
            if not future.empty:
                sel = st.multiselect(
                    f"Select legs for **{tipster}** acca:",
                    options=future['Match'].tolist(),
                    default=future['Match'].tolist()[:3],
                    key=f"{tipster}_acca"
                )
                acca = future[future['Match'].isin(sel)]
                if not acca.empty:
                    odds_prod = np.prod(acca['Odds'])
                    col1, col2 = st.columns(2)
                    with col1:
                        stake = st.number_input(f"Stake for {tipster} (£)", 1.0, 100.0, 10.0, 1.0,
                                                key=f"{tipster}_stake")
                    with col2:
                        st.metric("Combined Odds", f"{odds_prod:.2f}")
                        st.metric("Potential Return", f"£{stake * odds_prod:.2f}")
                    st.write("**Acca Legs:**")
                    st.table(acca[['Match', 'Selection', 'Odds']].reset_index(drop=True))
            else:
                st.info("No upcoming tips yet – check back tomorrow!")

    # ---------- Cross-tipster mega acca ----------
    st.header("Cross-Tipster Mega Acca Builder")
    all_future = pd.DataFrame()
    for tipster in selected_tipsters:
        df_tip = pd.DataFrame(tipsters_data[tipster]["tips"])
        fut = df_tip[df_tip['Outcome'] == 'Pending'].copy()
        fut['Tipster'] = tipster
        all_future = pd.concat([all_future, fut], ignore_index=True)

    if all_future.empty:
        st.info("No future tips across the selected tipsters.")
    else:
        # ---- create a nicely formatted label column ----
        all_future['Label'] = all_future['Match'] + " (" + all_future['Tipster'] + ")"

        mega_sel = st.multiselect(
            "Pick any legs from **all** tipsters:",
            options=all_future['Label'].tolist(),
            default=all_future['Label'].head(5).tolist(),
            key="mega_acca"
        )
        # ---- extract the rows that were chosen ----
        mega_df = all_future[all_future['Label'].isin(mega_sel)].copy()
        if not mega_df.empty:
            mega_odds = np.prod(mega_df['Odds'])
            col1, col2 = st.columns(2)
            with col1:
                mega_stake = st.number_input("Mega Acca Stake (£)", 1.0, 100.0, 5.0, 1.0, key="mega_stake")
            with col2:
                st.metric("Mega Combined Odds", f"{mega_odds:.2f}")
                st.metric("Potential Mega Return", f"£{mega_stake * mega_odds:.2f}")
            st.write("**Mega Acca Legs:**")
            st.table(mega_df[['Tipster', 'Match', 'Selection', 'Odds']].reset_index(drop=True))
        else:
            st.info("Select at least one leg to see the acca.")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Data sourced from ProTipster.com – all tips 100% free & verified. "
           "App uses static 2025 demo data. Gamble responsibly. 18+ only.")
st.markdown("[**Visit ProTipster**](https://www.protipster.com/betting-tips/football)")
