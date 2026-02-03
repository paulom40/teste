import streamlit as st
import pandas as pd
import numpy as np
import math
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# Try to import XGBoost, show instructions if not available
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Tennis Prediction System",
    page_icon="🎾",
    layout="wide"
)

# Initialize session state
if 'elo_ratings' not in st.session_state:
    st.session_state.elo_ratings = {}
if 'player_names' not in st.session_state:
    st.session_state.player_names = {}
if 'global_elo' not in st.session_state:
    st.session_state.global_elo = {}
if 'xgb_model' not in st.session_state:
    st.session_state.xgb_model = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = []
if 'match_data' not in st.session_state:
    st.session_state.match_data = None

# Function to compute surface-aware ELO ratings
def compute_surface_elo_from_csv(df, k_factor=32, initial_elo=1500):
    # Sort chronologically
    if 'tourney_date' in df.columns and 'match_num' in df.columns:
        df = df.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)
    
    # Get all unique player ids
    players = set(df['winner_id'].unique()).union(set(df['loser_id'].unique()))
    
    # Store player names
    if 'winner_name' in df.columns and 'loser_name' in df.columns:
        for _, row in df.iterrows():
            st.session_state.player_names[row['winner_id']] = row['winner_name']
            st.session_state.player_names[row['loser_id']] = row['loser_name']
    
    # Initialize ELO structure
    elo_ratings = {}
    global_ratings = {}
    surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']
    
    for player in players:
        elo_ratings[player] = {}
        for surface in surfaces:
            elo_ratings[player][surface] = initial_elo
        global_ratings[player] = initial_elo
    
    # Process matches
    for index, row in df.iterrows():
        winner = row['winner_id']
        loser = row['loser_id']
        
        # Get surface
        surface = row.get('surface', 'Hard')
        if pd.isna(surface) or surface not in surfaces:
            surface = 'Hard'
        
        # Get ratings
        rating_w = elo_ratings[winner].get(surface, global_ratings[winner])
        rating_l = elo_ratings[loser].get(surface, global_ratings[loser])
        
        # Update ELO
        exp_w = 1 / (1 + math.pow(10, (rating_l - rating_w) / 400))
        exp_l = 1 - exp_w
        
        elo_ratings[winner][surface] = rating_w + k_factor * (1 - exp_w)
        elo_ratings[loser][surface] = rating_l + k_factor * (0 - exp_l)
        
        global_ratings[winner] = global_ratings[winner] + k_factor * (1 - exp_w)
        global_ratings[loser] = global_ratings[loser] + k_factor * (0 - exp_l)
    
    return elo_ratings, global_ratings

# Function to prepare features for XGBoost
def prepare_features(df, elo_ratings, global_ratings):
    """Prepare match data with ELO features for machine learning"""
    features_list = []
    labels = []
    match_info = []
    
    # Define statistical features to use (from the CSV data)
    stat_features = [
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
        'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon'
    ]
    
    for idx, row in df.iterrows():
        try:
            winner_id = str(row['winner_id'])
            loser_id = str(row['loser_id'])
            surface = str(row.get('surface', 'Hard'))
            
            # Get ELO ratings
            winner_elo = elo_ratings.get(winner_id, {}).get(surface, global_ratings.get(winner_id, 1500))
            loser_elo = elo_ratings.get(loser_id, {}).get(surface, global_ratings.get(loser_id, 1500))
            
            # Calculate ELO difference
            elo_diff = winner_elo - loser_elo
            elo_expected = 1 / (1 + math.pow(10, (-elo_diff) / 400))
            
            # Create feature vector
            features = {
                'elo_diff': float(elo_diff),
                'winner_elo': float(winner_elo),
                'loser_elo': float(loser_elo),
                'elo_expected': float(elo_expected),
                
                # Surface encoding
                'is_hard': 1 if surface == 'Hard' else 0,
                'is_clay': 1 if surface == 'Clay' else 0,
                'is_grass': 1 if surface == 'Grass' else 0,
                
                # Player rankings if available
                'winner_rank': float(row.get('winner_rank', 100)),
                'loser_rank': float(row.get('loser_rank', 100)),
            }
            
            # Add statistical features from CSV if available
            for feat in stat_features:
                if feat in row and pd.notna(row[feat]):
                    features[feat] = float(row[feat])
                else:
                    # Use average values as defaults
                    if feat.startswith('w_'):
                        features[feat] = 5.0 if 'ace' in feat else 2.0 if 'df' in feat else 60.0 if 'svpt' in feat else 30.0
                    else:
                        features[feat] = 5.0 if 'ace' in feat else 2.0 if 'df' in feat else 60.0 if 'svpt' in feat else 30.0
            
            # Add derived statistics
            if features['w_svpt'] > 0:
                features['winner_1st_serve_pct'] = float(features['w_1stIn'] / features['w_svpt'])
                features['winner_1st_serve_won_pct'] = float(features['w_1stWon'] / max(1, features['w_1stIn']))
                features['winner_2nd_serve_won_pct'] = float(features['w_2ndWon'] / max(1, features['w_svpt'] - features['w_1stIn']))
            
            if features['l_svpt'] > 0:
                features['loser_1st_serve_pct'] = float(features['l_1stIn'] / features['l_svpt'])
                features['loser_1st_serve_won_pct'] = float(features['l_1stWon'] / max(1, features['l_1stIn']))
                features['loser_2nd_serve_won_pct'] = float(features['l_2ndWon'] / max(1, features['l_svpt'] - features['l_1stIn']))
            
            features_list.append(features)
            labels.append(1)  # 1 for winner (player A perspective)
            match_info.append({
                'winner_id': winner_id,
                'loser_id': loser_id,
                'surface': surface,
                'match_idx': idx
            })
            
            # Also add reverse perspective for more data
            features_reverse = features.copy()
            features_reverse['elo_diff'] = float(-elo_diff)
            features_reverse['winner_elo'], features_reverse['loser_elo'] = float(loser_elo), float(winner_elo)
            features_reverse['elo_expected'] = float(1 - elo_expected)
            features_reverse['winner_rank'], features_reverse['loser_rank'] = float(features['loser_rank']), float(features['winner_rank'])
            
            # Swap statistical features
            stat_swap_pairs = [
                ('w_ace', 'l_ace'), ('w_df', 'l_df'), ('w_svpt', 'l_svpt'),
                ('w_1stIn', 'l_1stIn'), ('w_1stWon', 'l_1stWon'), ('w_2ndWon', 'l_2ndWon')
            ]
            for w_feat, l_feat in stat_swap_pairs:
                features_reverse[w_feat], features_reverse[l_feat] = float(features[l_feat]), float(features[w_feat])
            
            features_list.append(features_reverse)
            labels.append(0)  # 0 for loser (player A perspective)
            
        except Exception as e:
            continue
    
    features_df = pd.DataFrame(features_list)
    return features_df, np.array(labels), match_info

# Train XGBoost model
def train_xgboost_model(features_df, labels, params=None):
    """Train XGBoost classifier on match features"""
    if params is None:
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train model
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, features_df.columns.tolist()

# Hybrid prediction function (simplified - no match statistics input)
def hybrid_prediction(player_a_id, player_b_id, surface, elo_ratings, global_ratings, xgb_model, feature_columns):
    """Make prediction using both ELO and XGBoost"""
    
    # Get ELO-based prediction
    winner_elo = elo_ratings.get(player_a_id, {}).get(surface, global_ratings.get(player_a_id, 1500))
    loser_elo = elo_ratings.get(player_b_id, {}).get(surface, global_ratings.get(player_b_id, 1500))
    
    elo_diff = float(winner_elo - loser_elo)
    elo_prob = float(1 / (1 + math.pow(10, (-elo_diff) / 400)))
    
    # Prepare features for XGBoost using historical averages
    features = {
        'elo_diff': elo_diff,
        'winner_elo': float(winner_elo),
        'loser_elo': float(loser_elo),
        'elo_expected': elo_prob,
        'is_hard': 1 if surface == 'Hard' else 0,
        'is_clay': 1 if surface == 'Clay' else 0,
        'is_grass': 1 if surface == 'Grass' else 0,
        
        # Use average statistics from historical data
        'w_ace': 5.0, 'w_df': 2.0, 'w_svpt': 60.0, 'w_1stIn': 36.0, 'w_1stWon': 24.0, 'w_2ndWon': 12.0,
        'l_ace': 5.0, 'l_df': 2.0, 'l_svpt': 60.0, 'l_1stIn': 36.0, 'l_1stWon': 24.0, 'l_2ndWon': 12.0,
        'winner_rank': 50.0, 'loser_rank': 50.0,
        
        # Derived statistics
        'winner_1st_serve_pct': 0.6,
        'winner_1st_serve_won_pct': 0.7,
        'winner_2nd_serve_won_pct': 0.5,
        'loser_1st_serve_pct': 0.6,
        'loser_1st_serve_won_pct': 0.7,
        'loser_2nd_serve_won_pct': 0.5,
    }
    
    # Create feature DataFrame
    features_df = pd.DataFrame([features])
    
    # Ensure all training columns are present
    for col in feature_columns:
        if col not in features_df.columns:
            features_df[col] = 0.0
    
    features_df = features_df[feature_columns]
    
    # Get XGBoost prediction
    if xgb_model is not None:
        xgb_prob = float(xgb_model.predict_proba(features_df)[0, 1])
        
        # Weighted combination
        elo_weight = 0.3
        xgb_weight = 0.7
        
        final_prob = float(elo_weight * elo_prob + xgb_weight * xgb_prob)
    else:
        xgb_prob = None
        final_prob = elo_prob
    
    return {
        'elo_probability': elo_prob,
        'xgb_probability': xgb_prob,
        'final_probability': final_prob,
        'player_a_elo': winner_elo,
        'player_b_elo': loser_elo,
        'elo_difference': elo_diff
    }

# Streamlit App Interface
st.title("🎾 Tennis Prediction System: ELO + XGBoost")
st.markdown("Combining traditional ELO ratings with machine learning for match predictions")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose a section:",
    ["📊 Data & Model Training", "🎯 Match Prediction", "🤖 Model Analysis", "📈 Player Rankings"]
)

# Check XGBoost availability
if not XGB_AVAILABLE and app_mode in ["🤖 Model Analysis", "🎯 Match Prediction"]:
    st.error("""
    **XGBoost not installed!**
    
    Please install XGBoost to use machine learning features:
    ```
    pip install xgboost scikit-learn
    ```
    """)

# Main app logic
if app_mode == "📊 Data & Model Training":
    st.header("Data Upload & Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload ATP Match Data CSV",
            type=['csv'],
            help="Upload match data with player IDs, surface, and match statistics"
        )
    
    with col2:
        st.subheader("Model Parameters")
        
        elo_k = st.slider("ELO K-factor", 10, 50, 32)
        initial_elo = st.slider("Initial ELO", 1200, 1800, 1500)
        
        if XGB_AVAILABLE:
            use_xgb = st.checkbox("Train XGBoost Model", value=True)
            xgb_depth = st.slider("XGBoost Max Depth", 3, 10, 6)
            xgb_estimators = st.slider("Number of Trees", 50, 200, 100)
        else:
            use_xgb = False
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.match_data = df
            
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            required_cols = ['winner_id', 'loser_id']
            if all(col in df.columns for col in required_cols):
                st.success(f"✅ Data loaded: {len(df)} matches")
                
                # Calculate ELO ratings
                if st.button("Calculate ELO Ratings & Train Models", type="primary"):
                    with st.spinner("Calculating ELO ratings..."):
                        elo_ratings, global_ratings = compute_surface_elo_from_csv(
                            df, k_factor=elo_k, initial_elo=initial_elo
                        )
                        st.session_state.elo_ratings = elo_ratings
                        st.session_state.global_elo = global_ratings
                    
                    st.success(f"✅ ELO calculated for {len(elo_ratings)} players")
                    
                    # Show top players
                    top_players = sorted(
                        [(pid, st.session_state.global_elo.get(pid, 1500)) 
                         for pid in elo_ratings.keys()],
                        key=lambda x: x[1], reverse=True
                    )[:10]
                    
                    st.subheader("Top 10 Players (Global ELO)")
                    top_df = pd.DataFrame(top_players, columns=['Player ID', 'ELO Rating'])
                    top_df['Player Name'] = top_df['Player ID'].map(st.session_state.player_names)
                    st.dataframe(top_df)
                    
                    # Train XGBoost model
                    if XGB_AVAILABLE and use_xgb:
                        with st.spinner("Training XGBoost model..."):
                            # Prepare features
                            features_df, labels, match_info = prepare_features(
                                df, elo_ratings, global_ratings
                            )
                            
                            # Train model
                            xgb_params = {
                                'max_depth': xgb_depth,
                                'n_estimators': xgb_estimators,
                                'learning_rate': 0.1,
                                'objective': 'binary:logistic',
                                'random_state': 42
                            }
                            
                            model, accuracy, feature_cols = train_xgboost_model(
                                features_df, labels, xgb_params
                            )
                            
                            st.session_state.xgb_model = model
                            st.session_state.feature_columns = feature_cols
                            
                            st.success(f"✅ XGBoost trained! Accuracy: {accuracy:.2%}")
                            
                            # Show feature importance
                            st.subheader("Top Feature Importances")
                            importance_df = pd.DataFrame({
                                'feature': feature_cols,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=False).head(15)
                            
                            st.bar_chart(importance_df.set_index('feature')['importance'])
                            
            else:
                st.error(f"Missing required columns: {required_cols}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif app_mode == "🎯 Match Prediction":
    st.header("Match Prediction")
    
    if not st.session_state.elo_ratings:
        st.warning("Please upload data and train models first in the 'Data & Model Training' section.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            # Surface selection
            surface = st.selectbox("Court Surface", ['Hard', 'Clay', 'Grass', 'Carpet'])
            
            # Player A selection
            player_options = list(st.session_state.elo_ratings.keys())
            player_a = st.selectbox(
                "Player A",
                options=player_options,
                format_func=lambda x: f"{st.session_state.player_names.get(x, x)} ({x})"
            )
        
        with col2:
            # Player B selection
            player_b_options = [p for p in player_options if p != player_a]
            player_b = st.selectbox(
                "Player B",
                options=player_b_options,
                format_func=lambda x: f"{st.session_state.player_names.get(x, x)} ({x})"
            )
            
            # Prediction options
            st.subheader("Prediction Options")
            use_xgb = st.checkbox("Use XGBoost (if trained)", 
                                 value=st.session_state.xgb_model is not None,
                                 help="Use machine learning model for more accurate predictions")
            show_details = st.checkbox("Show detailed breakdown", value=True)
            
            # Prediction button
            if st.button("Run Prediction", type="primary", use_container_width=True):
                # Store prediction in session state
                st.session_state.prediction_result = hybrid_prediction(
                    player_a, player_b, surface,
                    st.session_state.elo_ratings,
                    st.session_state.global_elo,
                    st.session_state.xgb_model if use_xgb else None,
                    st.session_state.feature_columns
                )
                st.session_state.last_prediction_players = (player_a, player_b, surface)
        
        # Display results if prediction exists
        if hasattr(st.session_state, 'prediction_result') and st.session_state.prediction_result:
            player_a_name = st.session_state.player_names.get(player_a, player_a)
            player_b_name = st.session_state.player_names.get(player_b, player_b)
            
            prediction = st.session_state.prediction_result
            
            st.subheader(f"Prediction: {player_a_name} vs {player_b_name}")
            st.markdown(f"**Surface:** {surface}")
            
            # Show probabilities
            col_prob_a, col_prob_b = st.columns(2)
            
            with col_prob_a:
                prob_a = float(prediction['final_probability'])
                st.metric(
                    label=player_a_name,
                    value=f"{prob_a:.1%}",
                    delta=f"Win Probability"
                )
                st.progress(min(1.0, max(0.0, prob_a)))
            
            with col_prob_b:
                prob_b = float(1 - prob_a)
                st.metric(
                    label=player_b_name,
                    value=f"{prob_b:.1%}",
                    delta=f"Win Probability"
                )
                st.progress(min(1.0, max(0.0, prob_b)))
            
            # Detailed breakdown
            if show_details:
                st.subheader("Prediction Breakdown")
                
                if prediction['xgb_probability'] is not None:
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("ELO Probability", f"{float(prediction['elo_probability']):.1%}")
                    with cols[1]:
                        st.metric("XGBoost Probability", f"{float(prediction['xgb_probability']):.1%}")
                    with cols[2]:
                        st.metric("Final Probability", f"{float(prediction['final_probability']):.1%}")
                else:
                    st.info("Using ELO-only prediction (XGBoost not available or not selected)")
                
                # ELO ratings
                st.markdown("**ELO Ratings**")
                elo_cols = st.columns(2)
                with elo_cols[0]:
                    st.write(f"{player_a_name}: {float(prediction['player_a_elo']):.0f}")
                with elo_cols[1]:
                    st.write(f"{player_b_name}: {float(prediction['player_b_elo']):.0f}")
                    st.write(f"ELO Difference: {float(prediction['elo_difference']):.0f}")
                
                # Prediction verdict
                st.subheader("Verdict")
                if prob_a > 0.70:
                    st.success(f"🎯 **Strong favorite:** {player_a_name} is predicted to win on {surface}!")
                elif prob_a > 0.60:
                    st.info(f"⚖️ **Moderate favorite:** {player_a_name} is predicted to win on {surface}!")
                elif prob_a > 0.55:
                    st.info(f"📈 **Slight favorite:** {player_a_name} is predicted to win on {surface}!")
                elif prob_b > 0.70:
                    st.success(f"🎯 **Strong favorite:** {player_b_name} is predicted to win on {surface}!")
                elif prob_b > 0.60:
                    st.info(f"⚖️ **Moderate favorite:** {player_b_name} is predicted to win on {surface}!")
                elif prob_b > 0.55:
                    st.info(f"📈 **Slight favorite:** {player_b_name} is predicted to win on {surface}!")
                else:
                    st.warning("🤔 **Too close to call!** This could go either way.")
        else:
            st.info("👈 Select players and click 'Run Prediction' to see results")

elif app_mode == "🤖 Model Analysis":
    st.header("Model Performance Analysis")
    
    if st.session_state.xgb_model is None:
        st.warning("No XGBoost model trained yet. Please train a model in the 'Data & Model Training' section.")
    else:
        st.success("✅ XGBoost model is ready!")
        
        # Model information
        st.subheader("Model Configuration")
        st.json({
            "n_estimators": st.session_state.xgb_model.n_estimators,
            "max_depth": st.session_state.xgb_model.max_depth,
            "learning_rate": st.session_state.xgb_model.learning_rate,
            "n_features": len(st.session_state.feature_columns)
        })
        
        # Feature importance
        st.subheader("Feature Importance")
        
        importance_dict = st.session_state.xgb_model.get_booster().get_score(importance_type="weight")
        
        if importance_dict:
            importance_df = pd.DataFrame(
                list(importance_dict.items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False)
            
            # Display top 15 features
            top_features = importance_df.head(15)
            
            col_chart, col_table = st.columns([2, 1])
            
            with col_chart:
                st.bar_chart(top_features.set_index('Feature'))
            
            with col_table:
                st.dataframe(top_features)
        
        # Model explanation
        st.subheader("How the Model Works")
        st.markdown("""
        **Hybrid Prediction System:**
        
        1. **ELO Rating System** (30% weight)
           - Traditional rating system based on match outcomes
           - Surface-specific ratings (Hard, Clay, Grass, Carpet)
           - Simple and interpretable
        
        2. **XGBoost Classifier** (70% weight)
           - Machine learning model trained on historical match data
           - Uses features from the uploaded CSV (aces, double faults, serve percentages, etc.)
           - Learns complex patterns from historical data
        
        **Final Prediction = 0.3 × ELO_Probability + 0.7 × XGBoost_Probability**
        
        The system automatically extracts features from your match data to train the XGBoost model.
        """)

elif app_mode == "📈 Player Rankings":
    st.header("Player Rankings & Analysis")
    
    if not st.session_state.elo_ratings:
        st.warning("Please upload data first in the 'Data & Model Training' section.")
    else:
        # Surface selection
        selected_surface = st.selectbox(
            "Select Surface for Rankings",
            ['Global', 'Hard', 'Clay', 'Grass', 'Carpet']
        )
        
        # Number of players to show
        num_players = st.slider("Number of players to display", 10, 50, 20)
        
        # Prepare rankings data
        rankings_data = []
        
        for player_id in st.session_state.elo_ratings.keys():
            player_name = st.session_state.player_names.get(player_id, player_id)
            
            if selected_surface == 'Global':
                rating = st.session_state.global_elo.get(player_id, 1500)
            else:
                rating = st.session_state.elo_ratings[player_id].get(selected_surface, 
                                                                   st.session_state.global_elo.get(player_id, 1500))
            
            rankings_data.append({
                'Player ID': player_id,
                'Player Name': player_name,
                'Rating': float(rating)
            })
        
        rankings_df = pd.DataFrame(rankings_data)
        rankings_df = rankings_df.sort_values('Rating', ascending=False).reset_index(drop=True)
        rankings_df.insert(0, 'Rank', range(1, len(rankings_df) + 1))
        
        # Display rankings
        st.subheader(f"{selected_surface} Court Rankings")
        st.dataframe(rankings_df.head(num_players), use_container_width=True)
        
        # Download option
        csv = rankings_df.to_csv(index=False)
        st.download_button(
            label=f"Download {selected_surface} Rankings as CSV",
            data=csv,
            file_name=f"tennis_rankings_{selected_surface.lower()}.csv",
            mime="text/csv"
        )
        
        # Surface specialization analysis
        if selected_surface == 'Global':
            st.subheader("Surface Specialists")
            
            specialization_data = []
            
            for player_id, ratings in st.session_state.elo_ratings.items():
                if len(ratings) >= 2:
                    surface_ratings = [float(ratings[s]) for s in ratings.keys()]
                    
                    if surface_ratings:
                        max_rating = max(surface_ratings)
                        min_rating = min(surface_ratings)
                        diff = max_rating - min_rating
                        
                        if diff > 50:  # Only show players with significant differences
                            best_surface = [s for s, r in ratings.items() if float(r) == max_rating][0]
                            worst_surface = [s for s, r in ratings.items() if float(r) == min_rating][0]
                            
                            specialization_data.append({
                                'Player': st.session_state.player_names.get(player_id, player_id),
                                'Specialization Score': float(diff),
                                'Best Surface': best_surface,
                                'Worst Surface': worst_surface,
                                'Best Rating': max_rating,
                                'Worst Rating': min_rating
                            })
            
            if specialization_data:
                spec_df = pd.DataFrame(specialization_data)
                spec_df = spec_df.sort_values('Specialization Score', ascending=False).head(10)
                
                st.dataframe(spec_df)

# Footer
st.markdown("---")
st.markdown(
    """
    **Tennis Prediction System Features:**
    - **Surface-Aware ELO Ratings**: Separate ratings for each court type
    - **XGBoost Machine Learning**: Trained on historical match statistics
    - **Hybrid Predictions**: Combines ELO and ML for better accuracy
    - **Player Rankings**: View rankings by surface
    
    **Required CSV columns:** `winner_id`, `loser_id`, `surface`
    **Recommended columns:** Player statistics for better ML predictions
    """
)
