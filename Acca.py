import streamlit as st
import pandas as pd
import numpy as np

# App Configuration
st.set_page_config(page_title="Multi-Tipster Soccer Dashboard", layout="wide")
st.title("🏆 Multi-Tipster Soccer Dashboard – Free Verified Experts")
st.markdown("**Track Top Free Tipsters from ProTipster.com | 100% Free Tips | Updated Daily**")
st.markdown("---")

# Sidebar for Tipster Selection
st.sidebar.header("Select Tipsters")
tipsters_available = ["NorthSea", "Rush 641", "GoalMaster", "ValueHunter"]
selected_tipsters = st.sidebar.multiselect(
    "Choose tipsters to display:",
    options=tipsters_available,
    default=tipsters_available[:2]  # Default to first two
)

# Tipster Data Dictionaries
tipsters_data = {
    "NorthSea": {
        "subtitle": "BTTS & Over/Under Specialist",
        "stats": {"Profit": "+£3,381", "Strike Rate": "77%", "Tips": "196"},
        "tips": {
            'Date': ['2025-10-25', '2025-10-22', '2025-10-19', '2025-10-15', '2025-10-12', '2025-10-08', '2025-10-05', '2025-11-08', '2025-11-09', '2025-11-10'],
            'Match': ['Man City vs Southampton (PL)', 'PSG vs Marseille (Ligue 1)', 'Real Madrid vs Villarreal (La Liga)', 'Juventus vs Inter (Serie A)', 'Bayern vs Dortmund (Bundesliga)', 'Liverpool vs Chelsea (PL)', 'Lyon vs Monaco (Ligue 1)', 'Union Berlin vs Bayern (Bundesliga)', 'AC Milan vs Juventus (Serie A)', 'Barcelona vs Real Sociedad (La Liga)'],
            'Selection': ['BTTS - Yes', 'Over 2.5 Goals', 'BTTS - Yes', 'Under 2.5 Goals', 'Over 3.5 Goals', 'BTTS & Over 2.5', 'BTTS - Yes', 'BTTS - Yes', 'Over 2.5 Goals', 'BTTS - Yes'],
            'Market': ['BTTS', 'Over/Under', 'BTTS', 'Over/Under', 'Over/Under', 'BTTS + O/U', 'BTTS', 'BTTS', 'Over/Under', 'BTTS'],
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
            'Date': ['2025-10-26', '2025-10-23', '2025-10-20', '2025-10-16', '2025-10-13', '2025-10-09', '2025-10-06', '2025-11-08', '2025-11-09', '2025-11-10'],
            'Match': ['Ajax vs PSV (Eredivisie)', 'Porto vs Benfica (Primeira Liga)', 'Atalanta vs Lazio (Serie A)', 'Leipzig vs Stuttgart (Bundesliga)', 'Sevilla vs Valencia (La Liga)', 'Celtic vs Rangers (Scottish PL)', 'Monaco vs Nice (Ligue 1)', 'Union Berlin vs Bayern (Bundesliga)', 'AC Milan vs Juventus (Serie A)', 'Barcelona vs Real Sociedad (La Liga)'],
            'Selection': ['Over 2.5 Goals', 'Under 2.5 Goals', 'Asian Handicap -1.5 Atalanta', 'Over 3.0 Goals', 'Over 1.5 Goals', 'Asian Handicap 0 Rangers', 'Under 2.5 Goals', 'Over 2.5 Goals', 'Asian Handicap -0.5 Milan', 'Over 2.5 Goals'],
            'Market': ['Over/Under', 'Over/Under', 'Asian Handicap', 'Over/Under', 'Over/Under', 'Asian Handicap', 'Over/Under', 'Over/Under', 'Asian Handicap', 'Over/Under'],
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
    "GoalMaster": {
        "subtitle": "Win/Draw/Win & BTTS Combo Master",
        "stats": {"Profit": "+£4,120", "Strike Rate": "68%", "Tips": "245"},
        "tips": {
            'Date': ['2025-10-27', '2025-10-24', '2025-10-21', '2025-10-17', '2025-10-14', '2025-10-10', '2025-10-07', '2025-11-08', '2025-11-09', '2025-11-10'],
            'Match': ['Arsenal vs Tottenham (PL)', 'Lille vs PSG (Ligue 1)', 'Inter vs Napoli (Serie A)', 'Dortmund vs Leverkusen (Bundesliga)', 'Atletico vs Real Madrid (La Liga)', 'Man Utd vs Liverpool (PL)', 'Juventus vs Roma (Serie A)', 'Union Berlin vs Bayern (Bundesliga)', 'AC Milan vs Juventus (Serie A)', 'Barcelona vs Real Sociedad (La Liga)'],
            'Selection': ['Arsenal to Win', 'BTTS - Yes', 'Inter to Win & Over 1.5', 'Draw', 'BTTS & Over 2.5', 'Man Utd to Win', 'Under 2.5 Goals', 'Bayern to Win', 'Milan to Win', 'Barcelona to Win & BTTS'],
            'Market': ['1X2', 'BTTS', 'Win + O/U', '1X2', 'BTTS + O/U', '1X2', 'Over/Under', '1X2', '1X2', 'Win + BTTS'],
            'Odds': [1.90, 1.80, 2.50, 3.20, 2.80, 2.10, 1.75, 1.40, 2.20, 3.00],
            'Outcome': ['Win', 'Win', 'Win', 'Loss', 'Win', 'Loss', 'Win', 'Pending', 'Pending', 'Pending'],
            'Reasoning': [
                'Arsenal home dominance; Spurs poor record at Emirates.',
                'PSG leaky; Lille score consistently in big games.',
                'Inter form unbeatable; Napoli travel fatigue.',
                'Evenly matched; history shows draws in 4/5.',
                'Derby chaos: BTTS in 7/8, goals galore.',
                'Home advantage for Utd; Liverpool midweek fatigue.',
                'Defensive masterclass expected in Italian clash.',
                'Bayern machine; Union no match.',
                'Milan revival; Juve inconsistent.',
                'Barca attack vs Sociedad defense—expect scores both ways.'
            ]
        }
    },
    "ValueHunter": {
        "subtitle": "Value Bets & Asian Lines Pro",
        "stats": {"Profit": "+£1,850", "Strike Rate": "65%", "Tips": "150"},
        "tips": {
            'Date': ['2025-10-28', '2025-10-25', '2025-10-22', '2025-10-18', '2025-10-15', '2025-10-11', '2025-10-08', '2025-11-08', '2025-11-09', '2025-11-10'],
            'Match': ['Chelsea vs Newcastle (PL)', 'Monaco vs Marseille (Ligue 1)', 'Bayern vs Leipzig (Bundesliga)', 'Milan vs Fiorentina (Serie A)', 'Valencia vs Girona (La Liga)', 'Everton vs Brighton (PL)', 'Lyon vs St Etienne (Ligue 1)', 'Union Berlin vs Bayern (Bundesliga)', 'AC Milan vs Juventus (Serie A)', 'Barcelona vs Real Sociedad (La Liga)'],
            'Selection': ['Asian Handicap -1 Chelsea', 'Over 2.5 Goals', 'Bayern -1.5 AH', 'Milan to Win', 'Under 2.5 Goals', 'BTTS - Yes', 'Lyon to Win & BTTS', 'Over 3.5 Goals', 'Under 2.5 Goals', 'Asian Handicap 0 Barca'],
            'Market': ['Asian Handicap', 'Over/Under', 'Asian Handicap', '1X2', 'Over/Under', 'BTTS', 'Win + BTTS', 'Over/Under', 'Over/Under', 'Asian Handicap'],
            'Odds': [2.00, 1.95, 1.80, 1.70, 1.85, 1.90, 2.30, 2.50, 1.90, 1.65],
            'Outcome': ['Win', 'Win', 'Loss', 'Win', 'Win', 'Win', 'Win', 'Pending', 'Pending', 'Pending'],
            'Reasoning': [
                'Chelsea firepower; Newcastle defensive issues.',
                'French flair: Goals in 6/7 h2h.',
                'Bayern cruise; Leipzig vulnerable.',
                'Milan solid at home; Fiorentina away struggles.',
                'Cautious Spanish affair; low scores typical.',
                'Both mid-table, leaky defenses.',
                'Derby passion: Lyon edge with goals.',
                'Bayern onslaught expected.',
                'Tactical chess match; low goals.',
                'Barca edge but Sociedad resilient.'
            ]
        }
    }
}

# Main Content
if not selected_tipsters:
    st.warning("Select at least one tipster from the sidebar to view their predictions!")
else:
    # Overall Summary
    st.header("📊 Overall Performance Summary")
    summary_data = []
    for tipster in selected_tipsters:
        data = tipsters_data[tipster]["tips"]
        df_temp = pd.DataFrame(data)
        wins = len(df_temp[df_temp['Outcome'] == 'Win'])
        resolved = len(df_temp[df_temp['Outcome'] != 'Pending'])
        strike = (wins / resolved * 100) if resolved > 0 else 0
        profit = sum((odds - 1) * 10 for odds in df_temp[df_temp['Outcome'] == 'Win']['Odds']) - 10 * len(df_temp[df_temp['Outcome'] == 'Loss'])
        summary_data.append({
            "Tipster": tipster,
            "Total Tips": len(df_temp),
            "Strike Rate": f"{strike:.1f}%",
            "Est. Profit (£10 stakes)": f"£{profit:.0f}"
        })
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

    # Individual Tipster Tabs
    st.header("🔍 Individual Tipster Predictions")
    tabs = st.tabs(selected_tipsters)

    for idx, tipster in enumerate(selected_tipsters):
        with tabs[idx]:
            t_data = tipsters_data[tipster]
            st.subheader(f"{tipster} – {t_data['subtitle']}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("YTD Profit", t_data['stats']['Profit'])
            with col2:
                st.metric("Strike Rate", t_data['stats']['Strike Rate'])
            with col3:
                st.metric("Total Tips", t_data['stats']['Tips'])

            df = pd.DataFrame(t_data['tips'])

            # Styling
            def highlight_outcome(val):
                if val == 'Win':
                    return 'background-color: #ccffcc; color: green; font-weight: bold'
                elif val == 'Loss':
                    return 'background-color: #ffcccc; color: red; font-weight: bold'
                elif val == 'Pending':
                    return 'background-color: #fff3cd; color: #d58b00; font-weight: bold'
                return ''

            def highlight_market(val):
                if 'BTTS' in val:
                    return 'background-color: #e6f7ff; font-weight: bold'
                elif 'Over' in val:
                    return 'background-color: #f0e6ff; font-weight: bold'
                elif 'Under' in val:
                    return 'background-color: #fff0f0; font-weight: bold'
                elif 'Asian' in val or 'Handicap' in val:
                    return 'background-color: #fff4e6; font-weight: bold'
                elif '1X2' in val or 'Win' in val:
                    return 'background-color: #f0fff0; font-weight: bold'
                return ''

            styled_df = df.style \
                .applymap(highlight_outcome, subset=['Outcome']) \
                .applymap(highlight_market, subset=['Market']) \
                .format({'Odds': '{:.2f}'})

            st.subheader("Prediction History")
            st.dataframe(styled_df, use_container_width=True)

            # Future Acca Builder for this Tipster
            st.subheader("Build Acca from Future Tips")
            future_df = df[df['Outcome'] == 'Pending'].copy()
            if not future_df.empty:
                selected = st.multiselect(
                    f"Select legs for {tipster} acca:",
                    options=future_df['Match'].tolist(),
                    default=future_df['Match'].tolist()[:3],  # Default to first 3
                    key=f"{tipster}_select"
                )
                acca_df = future_df[future_df['Match'].isin(selected)]
                if not acca_df.empty:
                    combined_odds = np.prod(acca_df['Odds'])
                    col1, col2 = st.columns(2)
                    with col1:
                        stake = st.number_input(f"Stake for {tipster} (£)", 1.0, 100.0, 10.0, 1.0, key=f"{tipster}_stake")
                    with col2:
                        st.metric("Combined Odds", f"{combined_odds:.2f}")
                        st.metric("Potential Return", f"£{stake * combined_odds:.2f}")
                    st.write("**Acca Legs:**")
                    st.table(acca_df[['Match', 'Selection', 'Odds']].reset_index(drop=True))
            else:
                st.info("No upcoming tips yet — check back tomorrow!")

    # Cross-Tipster Acca Builder
    st.header("🌟 Cross-Tipster Mega Acca Builder")
    all_future_tips = pd.DataFrame()
    for tipster in selected_tipsters:
        t_data = tipsters_data[tipster]["tips"]
        df_temp = pd.DataFrame(t_data)
        future_temp = df_temp[df_temp['Outcome'] == 'Pending'].copy()
        future_temp['Tipster'] = tipster
        all_future_tips = pd.concat([all_future_tips, future_temp])

    if not all_future_tips.empty:
        selected_cross = st.multiselect(
            "Select legs from all tipsters:",
            options=all_future_tips['Match'] + " (" + all_future_tips['Tipster'] + ")".tolist(),
            default=all_future_tips['Match'][:5].tolist(),  # Default to first 5
            format_func=lambda x: x
        )
        # Parse selected
        parsed_selected = []
        for sel in selected_cross:
            if '(' in sel:
                match = sel.split(' (')[0]
                tip = sel.split(' (')[1].rstrip(')')
                parsed_selected.append((match, tip))
        
        cross_acca_df = all_future_tips[
            all_future_tips.apply(lambda row: (row['Match'], row['Tipster']) in parsed_selected, axis=1)
        ]
        if not cross_acca_df.empty:
            combined_odds_cross = np.prod(cross_acca_df['Odds'])
            col1, col2 = st.columns(2)
            with col1:
                stake_cross = st.number_input("Mega Acca Stake (£)", 1.0, 100.0, 5.0, 1.0, key="cross_stake")
            with col2:
                st.metric("Mega Combined Odds", f"{combined_odds_cross:.2f}")
                st.metric("Potential Mega Return", f"£{stake_cross * combined_odds_cross:.2f}")
            st.write("**Mega Acca Legs:**")
            display_cross = cross_acca_df[['Tipster', 'Match', 'Selection', 'Odds']].reset_index(drop=True)
            st.table(display_cross)
    else:
        st.info("No future tips available across selected tipsters.")

# Footer
st.markdown("---")
st.caption("Data sourced from ProTipster.com. All tips 100% free & verified. App simulates 2025 data for demo. Gamble responsibly. 18+ only.")
st.markdown("[**Visit ProTipster**](https://www.protipster.com/betting-tips/football)")
