# Add this to your main app section - RIGHT AFTER THE IMPORTS AND CONFIG

# ================================
# CSV UPLOAD & DATA LOADING
# ================================
st.sidebar.header("📁 Upload Match Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV File", type=["csv"], 
                                         help="Upload your football match data CSV file")

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = load_csv(uploaded_file.read())
        if df.empty:
            st.error("The uploaded CSV file is empty.")
        else:
            st.success(f"✅ Successfully loaded {len(df):,} matches")
            
            # Show dataframe preview
            with st.expander("📊 Data Preview (First 10 rows)"):
                st.dataframe(df.head(10))
            
            # Auto-detect columns
            mapping = detect_columns(df)
            
            st.sidebar.subheader("🔧 Column Mapping")
            st.sidebar.write("Map your CSV columns to the required fields:")
            
            col_map = {}
            required_fields = [
                ("HomeTeam", "Home Team"),
                ("AwayTeam", "Away Team"), 
                ("FTHG", "Full Time Home Goals"),
                ("FTAG", "Full Time Away Goals")
            ]
            
            optional_fields = [
                ("HC", "Home Corners"),
                ("AC", "Away Corners"),
                ("HS", "Home Shots"),
                ("AS", "Away Shots"), 
                ("HxG", "Home Expected Goals"),
                ("AxG", "Away Expected Goals")
            ]
            
            # Required fields
            for field, label in required_fields:
                detected = mapping.get(field)
                options = [""] + list(df.columns)
                default_idx = options.index(detected) if detected in options else 0
                col_map[field] = st.sidebar.selectbox(
                    f"**{label}** *", 
                    options=options, 
                    index=default_idx,
                    help=f"Select column for {label}"
                )
            
            # Optional fields  
            st.sidebar.write("**Optional Fields:**")
            for field, label in optional_fields:
                detected = mapping.get(field)
                options = [""] + list(df.columns)
                default_idx = options.index(detected) if detected in options else 0
                col_map[field] = st.sidebar.selectbox(
                    label,
                    options=options, 
                    index=default_idx,
                    help=f"Select column for {label} (optional)"
                )
            
            # Check required fields are mapped
            missing = [r for r in ["HomeTeam", "AwayTeam", "FTHG", "FTAG"] if not col_map[r]]
            if missing:
                st.error(f"❌ Please map these required fields: {', '.join(missing)}")
                st.stop()
            
            # Model training settings
            st.sidebar.subheader("⚙️ Model Settings")
            recency_weight = st.sidebar.slider(
                "Recency Weight", 
                0.5, 5.0, 2.0, 0.1,
                help="Higher values give more weight to recent matches"
            )
            min_matches = st.sidebar.number_input(
                "Minimum Matches per Team", 
                1, 20, 3,
                help="Minimum matches required for reliable team stats"
            )
            
            # Train the model
            with st.spinner("🔄 Training prediction model on your data..."):
                try:
                    team_stats = compute_team_stats(
                        _df=df,
                        home_col=col_map["HomeTeam"], 
                        away_col=col_map["AwayTeam"],
                        hg_col=col_map["FTHG"], 
                        ag_col=col_map["FTAG"],
                        hc_col=col_map.get("HC"), 
                        ac_col=col_map.get("AC"),
                        hs_col=col_map.get("HS"), 
                        as_col=col_map.get("AS"),
                        hxg_col=col_map.get("HxG"), 
                        axg_col=col_map.get("AxG"),
                        recency_weight=recency_weight,
                        min_matches=min_matches
                    )
                    
                    st.success("✅ Model trained successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
                    st.stop()
            
            # Prepare data for prediction
            rename_dict = {v: k for k, v in col_map.items() if v}
            df_clean = df.rename(columns=rename_dict).copy()
            teams = sorted(set(df_clean["HomeTeam"]).union(df_clean["AwayTeam"]))
            
            # Injury Input
            st.sidebar.subheader("🏥 Injury Information")
            st.sidebar.write("Format: `Team: Player (role:position, impact:15%)`")
            injury_input = st.sidebar.text_area(
                "Injured Players", 
                placeholder="Example:\nArsenal: Saka (role:forward, impact:15%)\nChelsea: James (role:defender, impact:20%)",
                height=100,
                help="Enter one injury per line in the format shown"
            )
            injuries = parse_injuries(injury_input)
            
            # PREDICTION SECTION
            st.markdown("---")
            st.subheader("🔮 Match Prediction")
            
            col1, col2 = st.columns(2)
            home_team = col1.selectbox("Home Team", teams, key="home_select")
            away_team = col2.selectbox("Away Team", teams, key="away_select")
            
            # Prediction buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎯 Standard Prediction", use_container_width=True):
                    with st.spinner("Calculating standard prediction..."):
                        pred = predict_match(home_team, away_team, team_stats, df,
                                           col_map["HomeTeam"], col_map["AwayTeam"],
                                           col_map["FTHG"], col_map["FTAG"], injuries)
                        # Display standard results (your existing display function)
                        display_standard_predictions(pred, home_team, away_team)
            
            with col2:
                if st.button("🤖 AI Enhanced Prediction", use_container_width=True, type="primary"):
                    with st.spinner("Running advanced AI models..."):
                        # Train ensemble if possible
                        if ML_AVAILABLE and len(df) >= 20:
                            ensemble_predictor.train_ensemble(df_clean, "HomeTeam", "AwayTeam", "FTHG")
                        
                        # Get enhanced prediction
                        pred = predict_match_enhanced(
                            home_team, away_team, team_stats, ensemble_predictor, df,
                            col_map["HomeTeam"], col_map["AwayTeam"], 
                            col_map["FTHG"], col_map["FTAG"], injuries
                        )
                        
                        # Display both standard and enhanced results
                        display_standard_predictions(pred, home_team, away_team)
                        display_enhanced_predictions(pred, home_team, away_team)
            
            # Team statistics overview
            with st.expander("📈 League Overview"):
                st.write(f"**Total Teams:** {len(teams)}")
                st.write(f"**Total Matches:** {len(df_clean)}")
                if 'FTHG' in df_clean.columns:
                    avg_home_goals = df_clean['FTHG'].mean()
                    avg_away_goals = df_clean['FTAG'].mean()
                    st.write(f"**Average Home Goals:** {avg_home_goals:.2f}")
                    st.write(f"**Average Away Goals:** {avg_away_goals:.2f}")
                
    except Exception as e:
        st.error(f"❌ Error processing CSV file: {str(e)}")
        st.info("💡 Make sure your CSV file has the correct format with columns for teams and goals.")

else:
    # Show when no file is uploaded
    st.info("📁 Please upload a CSV file to get started")
    
    with st.expander("💡 CSV Format Guide"):
        st.markdown("""
        **Required Columns:**
        - Home Team (e.g., 'HomeTeam', 'Home')
        - Away Team (e.g., 'AwayTeam', 'Away')  
        - Home Goals (e.g., 'FTHG', 'HG', 'HomeGoals')
        - Away Goals (e.g., 'FTAG', 'AG', 'AwayGoals')
        
        **Optional Columns:**
        - Home Corners, Away Corners
        - Home Shots, Away Shots  
        - Home xG, Away xG
        - Date (for form calculations)
        
        **Example CSV structure:**
        ```
        HomeTeam,AwayTeam,FTHG,FTAG,HC,AC,Date
        Arsenal,Chelsea,2,1,6,4,2024-01-15
        Man Utd,Liverpool,1,1,5,7,2024-01-14
        ```
        """)
    
    # Sample data option
    if st.button("🎲 Load Sample Data (for testing)"):
        # Create sample data
        sample_data = {
            'HomeTeam': ['Arsenal', 'Chelsea', 'Man Utd', 'Liverpool', 'Man City', 'Tottenham'],
            'AwayTeam': ['Chelsea', 'Arsenal', 'Liverpool', 'Man Utd', 'Tottenham', 'Man City'],
            'FTHG': [2, 1, 1, 2, 3, 1],
            'FTAG': [1, 2, 2, 1, 0, 2],
            'HC': [6, 4, 5, 7, 8, 3],
            'AC': [4, 6, 7, 5, 2, 8],
            'HS': [15, 12, 10, 14, 18, 9],
            'AS': [10, 14, 16, 11, 6, 17]
        }
        sample_df = pd.DataFrame(sample_data)
        st.session_state.sample_data = sample_df
        st.rerun()

# Check if sample data is loaded
if 'sample_data' in st.session_state:
    df = st.session_state.sample_data
    st.info("🎲 Sample data loaded for testing. Upload your own CSV for real predictions.")
    # Continue with the same processing logic as above...
