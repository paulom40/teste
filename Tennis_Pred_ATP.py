import pandas as pd
import math

# Function to compute ELO ratings from the CSV
def compute_elo_from_csv(csv_file, k_factor=32, initial_elo=1500):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Sort by tourney_date and match_num for chronological order
    df = df.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)
    
    # Get all unique player ids
    players = set(df['winner_id'].unique()).union(set(df['loser_id'].unique()))
    
    # Initialize ELO ratings
    elo_ratings = {player: initial_elo for player in players}
    
    # Function to calculate expected score
    def expected_score(rating_a, rating_b):
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))
    
    # Process each match
    for index, row in df.iterrows():
        winner = row['winner_id']
        loser = row['loser_id']
        
        # Current ratings
        rating_w = elo_ratings[winner]
        rating_l = elo_ratings[loser]
        
        # Expected scores
        exp_w = expected_score(rating_w, rating_l)
        exp_l = expected_score(rating_l, rating_w)
        
        # Update ratings (winner gets 1, loser gets 0)
        elo_ratings[winner] = rating_w + k_factor * (1 - exp_w)
        elo_ratings[loser] = rating_l + k_factor * (0 - exp_l)
    
    return elo_ratings

# Function to predict a match
def predict_match(player_a_id, player_b_id, elo_ratings):
    if player_a_id not in elo_ratings or player_b_id not in elo_ratings:
        return "One or both players not found."
    rating_a = elo_ratings[player_a_id]
    rating_b = elo_ratings[player_b_id]
    prob_a = 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))
    prob_b = 1 - prob_a
    return f"Probability {player_a_id} wins: {prob_a:.2f}, {player_b_id} wins: {prob_b:.2f}"

# Example usage:
# elo = compute_elo_from_csv('atp_matches_2025.csv')
# print(predict_match('D643', 'A0E2', elo))  # Djokovic vs Alcaraz
