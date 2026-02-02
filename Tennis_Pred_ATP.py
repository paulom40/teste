import streamlit as st
import pandas as pd
import math
import numpy as np
from collections import defaultdict

# Set page config
st.set_page_config(
    page_title="Tennis ELO Rating Calculator",
    page_icon="🎾",
    layout="wide"
)

# Title
st.title("🎾 Surface-Aware Tennis ELO Rating Calculator")
st.markdown("Calculate ELO ratings for tennis players with separate ratings for each surface type")

# Initialize session state
if 'elo_ratings' not in st.session_state:
    st.session_state.elo_ratings = {}  # Now will be dict of dicts: {player_id: {surface: rating}}
if 'player_names' not in st.session_state:
    st.session_state.player_names = {}
if 'global_elo' not in st.session_state:
    st.session_state.global_elo = {}  # Global rating for players without surface-specific data

# Function to compute surface-aware ELO ratings
def compute_surface_elo_from_csv(df, k_factor=32, initial_elo=1500):
    # Sort by tourney_date and match_num for chronological order
    if 'tourney_date' in df.columns and 'match_num' in df.columns:
        df = df.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)
    
    # Get all unique player ids
    players = set(df['winner_id'].unique()).union(set(df['loser_id'].unique()))
    
    # Store player names if available
    if 'winner_name' in df.columns and 'loser_name' in df.columns:
        for _, row in df.iterrows():
            st.session_state.player_names[row['winner_id']] = row['winner_name']
            st.session_state.player_names[row['loser_id']] = row['loser_name']
    
    # Initialize ELO ratings structure
    # Each player has a dict: {'Hard': rating, 'Clay': rating, 'Grass': rating, 'Carpet': rating, 'Global': rating}
    elo_ratings = {}
    global_ratings = {}
    
    # Standard surfaces in tennis
    surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']
    
    for player in players:
        elo_ratings[player] = {}
        for surface in surfaces:
            elo_ratings[player][surface] = initial_elo
        # Also track a global rating for fallback
        global_ratings[player] = initial_elo
    
    # Function to calculate expected score
    def expected_score(rating_a, rating_b):
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))
    
    # Process each match
    for index, row in df.iterrows():
        winner = row['winner_id']
        loser = row['loser_id']
        
        # Get surface, default to Hard if not specified
        surface = row.get('surface', 'Hard')
        if pd.isna(surface) or surface not in surfaces:
            surface = 'Hard'
        
        # Get ratings for this surface (use global if no surface-specific rating yet)
        rating_w = elo_ratings[winner].get(surface, global_ratings[winner])
        rating_l = elo_ratings[loser].get(surface, global_ratings[loser])
        
        # Expected scores
        exp_w = expected_score(rating_w, rating_l)
        exp_l = expected_score(rating_l, rating_w)
        
        # Update surface-specific ratings
        elo_ratings[winner][surface] = rating_w + k_factor * (1 - exp_w)
        elo_ratings[loser][surface] = rating_l + k_factor * (0 - exp_l)
        
        # Also update global ratings
        global_ratings[winner] = global_ratings[winner] + k_factor * (1 - exp_w)
        global_ratings[loser] = global_ratings[loser] + k_factor * (0 - exp_l)
    
    return elo_ratings, global_ratings

# Function to predict a match on a specific surface
def predict_surface_match(player_a_id, player_b_id, surface, elo_ratings, global_ratings):
    if player_a_id not in elo_ratings or player_b_id not in elo_ratings:
        return None, None, "One or both players not found."
    
    # Get ratings for the specified surface, fall back to global rating
    if surface in elo_ratings[player_a_id]:
        rating_a = elo_ratings[player_a_id][surface]
    else:
        rating_a = global_ratings.get(player_a_id, 1500)
    
    if surface in elo_ratings[player_b_id]:
        rating_b = elo_ratings[player_b_id][surface]
    else:
        rating_b = global_ratings.get(player_b_id, 1500)
    
    prob_a = 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))
    prob_b = 1 - prob_a
    
    # Get player names if available
    player_a_name = st.session_state.player_names.get(player_a_id, player_a_id)
    player_b_name = st.session_state.player_names.get(player_b_id, player_b_id)
    
    return prob_a, prob_b, player_a_name, player_b_name, rating_a, rating_b

# Function to get player's surface specialization
def get_surface_specialization(player_id, elo_ratings, global_ratings):
    if player_id not in elo_ratings:
        return "No data available"
    
    player_ratings = elo_ratings[player_id]
    surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']
    
    # Filter surfaces with data
    available_surfaces = [s for s in surfaces if s in player_ratings]
    
    if not available_surfaces:
        return "No surface data"
    
    # Find best and worst surfaces
    best_surface = max(available_surfaces, key=lambda s: player_ratings[s])
    worst_surface = min(available_surfaces, key=lambda s: player_ratings[s])
    
    # Calculate specialization strength
    best_rating = player_ratings[best_surface]
    worst_rating = player_ratings[worst_surface]
    
    if best_rating - worst_rating > 100:
        strength = "Strong specialist"
    elif best_rating - worst_rating > 50:
        strength = "Moderate specialist"
    else:
        strength = "Balanced player"
    
    return f"{strength}: Best on {best_surface} ({best_rating:.0f}), Worst on {worst_surface} ({worst_rating:.0f})"

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose a section:",
    ["📊 Upload & Calculate ELO", "🎯 Match Prediction", "👤 Player Analysis", "📈 View All Ratings"]
)

# Main content area
if app_mode == "📊 Upload & Calculate ELO":
    st.header("Upload Match Data and Calculate Surface-Aware ELO Ratings")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with match data",
            type=['csv'],
            help="CSV should contain columns: winner_id, loser_id, surface (and optionally tourney_date, match_num, winner_name, loser_name)"
        )
    
    with col2:
        # ELO parameters
        st.subheader("ELO Parameters")
        initial_elo = st.number_input("Initial ELO", min_value=1000, max_value=2000, value=1500)
        k_factor = st.number_input("K-factor", min_value=10, max_value=50, value=32)
        surface_k_factor = st.number_input("Surface K-factor", min_value=10, max_value=50, value=24,
                                          help="K-factor for surface-specific adjustments")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Display preview
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Show required columns
            required_cols = ['winner_id', 'loser_id']
            if all(col in df.columns for col in required_cols):
                st.success("✅ Required columns found!")
                
                # Check for surface column
                if 'surface' not in df.columns:
                    st.warning("⚠️ No 'surface' column found. All matches will be treated as Hard court.")
                    df['surface'] = 'Hard'
                else:
                    st.success("✅ Surface column found!")
                    # Show surface distribution
                    surface_counts = df['surface'].value_counts()
                    st.write("Surface distribution:")
                    st.dataframe(surface_counts)
                
                # Calculate ELO button
                if st.button("Calculate Surface-Aware ELO Ratings", type="primary"):
                    with st.spinner("Calculating surface-specific ELO ratings..."):
                        elo_ratings, global_ratings = compute_surface_elo_from_csv(df, k_factor, initial_elo)
                        st.session_state.elo_ratings = elo_ratings
                        st.session_state.global_elo = global_ratings
                    
                    st.success(f"✅ Calculated ELO ratings for {len(elo_ratings)} players across surfaces!")
                    
                    # Show top players by surface
                    st.subheader("Top Players by Surface")
                    
                    surfaces_to_show = ['Hard', 'Clay', 'Grass']
                    cols = st.columns(len(surfaces_to_show))
                    
                    for idx, surface in enumerate(surfaces_to_show):
                        with cols[idx]:
                            st.markdown(f"**{surface} Courts**")
                            # Get top 5 players for this surface
                            surface_ratings = []
                            for player, ratings in elo_ratings.items():
                                if surface in ratings:
                                    surface_ratings.append((player, ratings[surface]))
                            
                            if surface_ratings:
                                surface_ratings.sort(key=lambda x: x[1], reverse=True)
                                for i, (player_id, rating) in enumerate(surface_ratings[:5]):
                                    player_name = st.session_state.player_names.get(player_id, player_id)
                                    st.write(f"{i+1}. {player_name}: {rating:.0f}")
                            else:
                                st.write("No data")
                    
            else:
                st.error(f"Missing required columns. CSV must contain: {required_cols}")
                
        except Exception as e:
            st.error(f"Error reading file: {e}")

elif app_mode == "🎯 Match Prediction":
    st.header("Surface-Specific Match Prediction")
    
    if not st.session_state.elo_ratings:
        st.warning("Please upload data and calculate ELO ratings first in the 'Upload & Calculate ELO' section.")
    else:
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Surface selection
            surface = st.selectbox(
                "Select Court Surface",
                options=['Hard', 'Clay', 'Grass', 'Carpet'],
                help="Choose the surface for this match prediction"
            )
            
            # Player A selection
            player_options = list(st.session_state.elo_ratings.keys())
            player_a_id = st.selectbox(
                "Select Player A",
                options=player_options,
                format_func=lambda x: f"{st.session_state.player_names.get(x, x)} ({x})" if st.session_state.player_names.get(x) else x
            )
        
        with col2:
            # Player B selection
            player_b_options = [p for p in player_options if p != player_a_id]
            player_b_id = st.selectbox(
                "Select Player B",
                options=player_b_options,
                format_func=lambda x: f"{st.session_state.player_names.get(x, x)} ({x})" if st.session_state.player_names.get(x) else x
            )
            
            # Show surface specialization
            if player_a_id and player_b_id:
                st.markdown("**Surface Specialization**")
                specialization_a = get_surface_specialization(player_a_id, st.session_state.elo_ratings, st.session_state.global_elo)
                specialization_b = get_surface_specialization(player_b_id, st.session_state.elo_ratings, st.session_state.global_elo)
                
                st.write(f"**Player A:** {specialization_a}")
                st.write(f"**Player B:** {specialization_b}")
        
        with col3:
            # Prediction button
            if st.button("Predict Match", type="primary", use_container_width=True):
                # Make prediction
                prob_a, prob_b, player_a_name, player_b_name, rating_a, rating_b = predict_surface_match(
                    player_a_id, player_b_id, surface, st.session_state.elo_ratings, st.session_state.global_elo
                )
                
                if prob_a is not None:
                    # Display ratings
                    st.subheader("Surface Ratings")
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric(
                            label=f"{player_a_name}",
                            value=f"{rating_a:.0f}",
                            delta=f"{surface} Rating"
                        )
                    
                    with col_b:
                        st.metric(
                            label=f"{player_b_name}",
                            value=f"{rating_b:.0f}",
                            delta=f"{surface} Rating"
                        )
                    
                    # Display probabilities
                    st.subheader("Win Probabilities")
                    col_proba, col_probb = st.columns(2)
                    
                    with col_proba:
                        st.progress(prob_a)
                        st.markdown(f"### {prob_a*100:.1f}%")
                        st.caption(f"**{player_a_name}**")
                    
                    with col_probb:
                        st.progress(prob_b)
                        st.markdown(f"### {prob_b*100:.1f}%")
                        st.caption(f"**{player_b_name}**")
                    
                    # Prediction verdict
                    st.subheader("Prediction Verdict")
                    if prob_a > 0.7:
                        st.success(f"🎯 **Strong favorite on {surface}:** {player_a_name} is predicted to win!")
                    elif prob_a > 0.55:
                        st.info(f"⚖️ **Favorite on {surface}:** {player_a_name} is predicted to win!")
                    elif prob_b > 0.7:
                        st.success(f"🎯 **Strong favorite on {surface}:** {player_b_name} is predicted to win!")
                    elif prob_b > 0.55:
                        st.info(f"⚖️ **Favorite on {surface}:** {player_b_name} is predicted to win!")
                    else:
                        st.warning(f"🤔 **Too close to call on {surface}!**")

elif app_mode == "👤 Player Analysis":
    st.header("Player Surface Analysis")
    
    if not st.session_state.elo_ratings:
        st.warning("Please upload data and calculate ELO ratings first in the 'Upload & Calculate ELO' section.")
    else:
        # Player selection
        player_options = list(st.session_state.elo_ratings.keys())
        selected_player = st.selectbox(
            "Select Player",
            options=player_options,
            format_func=lambda x: f"{st.session_state.player_names.get(x, x)} ({x})" if st.session_state.player_names.get(x) else x
        )
        
        if selected_player:
            # Get player ratings
            player_ratings = st.session_state.elo_ratings[selected_player]
            player_name = st.session_state.player_names.get(selected_player, selected_player)
            
            # Display player info
            st.subheader(f"Surface Ratings for {player_name}")
            
            # Create surface rating chart
            surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']
            ratings = []
            labels = []
            
            for surface in surfaces:
                if surface in player_ratings:
                    ratings.append(player_ratings[surface])
                    labels.append(f"{surface}\n{player_ratings[surface]:.0f}")
            
            if ratings:
                # Create a simple bar chart using Streamlit
                chart_data = pd.DataFrame({
                    'Surface': labels,
                    'ELO Rating': ratings
                })
                
                st.bar_chart(chart_data.set_index('Surface'))
                
                # Display specialization
                specialization = get_surface_specialization(selected_player, st.session_state.elo_ratings, st.session_state.global_elo)
                st.info(f"**Surface Specialization:** {specialization}")
                
                # Detailed ratings table
                st.subheader("Detailed Ratings")
                ratings_table = []
                for surface in surfaces:
                    if surface in player_ratings:
                        ratings_table.append({
                            'Surface': surface,
                            'ELO Rating': f"{player_ratings[surface]:.0f}",
                            'Difference from Global': f"{player_ratings[surface] - st.session_state.global_elo.get(selected_player, 1500):+.0f}"
                        })
                
                if ratings_table:
                    st.table(pd.DataFrame(ratings_table))
            else:
                st.warning("No surface-specific ratings available for this player.")

elif app_mode == "📈 View All Ratings":
    st.header("All Player ELO Ratings by Surface")
    
    if not st.session_state.elo_ratings:
        st.warning("Please upload data and calculate ELO ratings first in the 'Upload & Calculate ELO' section.")
    else:
        # Surface selection for viewing
        selected_surface = st.selectbox(
            "Filter by Surface",
            options=['All', 'Hard', 'Clay', 'Grass', 'Carpet', 'Global']
        )
        
        # Create dataframe with all ratings
        data = []
        for player_id, ratings in st.session_state.elo_ratings.items():
            player_name = st.session_state.player_names.get(player_id, player_id)
            
            if selected_surface == 'All':
                # Show all surfaces
                for surface, rating in ratings.items():
                    data.append({
                        'Player ID': player_id,
                        'Player Name': player_name,
                        'Surface': surface,
                        'ELO Rating': rating
                    })
            elif selected_surface == 'Global':
                # Show global ratings
                data.append({
                    'Player ID': player_id,
                    'Player Name': player_name,
                    'Surface': 'Global',
                    'ELO Rating': st.session_state.global_elo.get(player_id, 1500)
                })
            elif selected_surface in ratings:
                # Show specific surface
                data.append({
                    'Player ID': player_id,
                    'Player Name': player_name,
                    'Surface': selected_surface,
                    'ELO Rating': ratings[selected_surface]
                })
        
        if data:
            ratings_df = pd.DataFrame(data)
            
            if selected_surface != 'All':
                # Sort by rating for single surface view
                ratings_df = ratings_df.sort_values('ELO Rating', ascending=False).reset_index(drop=True)
                ratings_df.insert(0, 'Rank', range(1, len(ratings_df) + 1))
            
            # Display
            st.dataframe(ratings_df, use_container_width=True)
            
            # Download button
            csv = ratings_df.to_csv(index=False)
            st.download_button(
                label=f"Download {selected_surface} Ratings as CSV",
                data=csv,
                file_name=f"elo_ratings_{selected_surface.lower()}.csv",
                mime="text/csv"
            )
            
            # Statistics
            if selected_surface != 'All':
                st.subheader(f"{selected_surface} Court Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Players", len(ratings_df))
                
                with col2:
                    st.metric("Highest ELO", f"{ratings_df['ELO Rating'].max():.0f}")
                
                with col3:
                    st.metric("Lowest ELO", f"{ratings_df['ELO Rating'].min():.0f}")
                
                with col4:
                    st.metric("Average ELO", f"{ratings_df['ELO Rating'].mean():.0f}")
        else:
            st.info(f"No ratings available for {selected_surface} surface.")

# Footer
st.markdown("---")
st.markdown(
    """
    **How Surface-Aware ELO Works:**
    1. Players have **separate ratings for each surface type** (Hard, Clay, Grass, Carpet)
    2. Matches only affect the rating for that specific surface
    3. Players also maintain a **global rating** as a fallback
    4. The system identifies **surface specialists** vs **all-round players**
    
    **Required CSV columns:** `winner_id`, `loser_id`, `surface`
    **Recommended columns:** `tourney_date`, `match_num`, `winner_name`, `loser_name`
    """
)
