import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PremierLeagueBettingModel:
    """
    Adapted model for E0.csv data (Premier League matches).
    
    Uses actual match statistics:
    - HS/AS: Shots (Home/Away)
    - HST/AST: Shots on Target
    - HF/AF: Fouls
    - HC/AC: Corners
    - HY/AY: Yellow cards
    - FTHG/FTAG: Full time goals
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.coefficients = {}
        
    def calculate_attacking_pressure(self, shots, shots_on_target, corners, fouls):
        """Calculate attacking pressure from match statistics."""
        shot_quality = shots_on_target / max(shots, 1)
        
        attacking_pressure = (
            (shots * 0.35) +
            (shots_on_target * 0.40) +
            (corners * 0.15) +
            (fouls * 0.10)
        )
        
        return attacking_pressure
    
    def calculate_defensive_pressure(self, shots_against, shots_on_target_against, 
                                    corners_against, yellow_cards):
        """Calculate defensive pressure from match statistics."""
        opponent_threat = (shots_against * 0.35) + (shots_on_target_against * 0.40)
        defensive_actions = (corners_against * 0.15) + (yellow_cards * 0.10)
        
        defensive_pressure = defensive_actions - (opponent_threat * 0.5)
        
        return defensive_pressure
    
    def create_match_features_from_row(self, row):
        """Create features from a single CSV row."""
        # Home team attacking pressure
        home_attack = self.calculate_attacking_pressure(
            row['HS'],
            row['HST'],
            row['HC'],
            row['AF']
        )
        
        # Home team defensive pressure
        home_defense = self.calculate_defensive_pressure(
            row['AS'],
            row['AST'],
            row['AC'],
            row['HY']
        )
        
        # Away team attacking pressure
        away_attack = self.calculate_attacking_pressure(
            row['AS'],
            row['AST'],
            row['AC'],
            row['HF']
        )
        
        # Away team defensive pressure
        away_defense = self.calculate_defensive_pressure(
            row['HS'],
            row['HST'],
            row['HC'],
            row['AY']
        )
        
        return {
            'home_attack': home_attack,
            'home_defense': home_defense,
            'away_attack': away_attack,
            'away_defense': away_defense,
            'total_attack': home_attack + away_attack,
            'total_defense': home_defense + away_defense,
            'total_goals': row['FTHG'] + row['FTAG']
        }
    
    def prepare_training_data(self, csv_path):
        """Load CSV and prepare training data."""
        df = pd.read_csv(csv_path)
        
        # Sort by date
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df = df.sort_values('Date').reset_index(drop=True)
        
        training_data = []
        
        for idx, row in df.iterrows():
            try:
                features = self.create_match_features_from_row(row)
                training_data.append(features)
            except:
                continue
        
        return pd.DataFrame(training_data), df
    
    def train_model(self, training_df):
        """Train regression model on prepared data."""
        X = training_df[['home_attack', 'home_defense', 'away_attack', 'away_defense', 
                        'total_attack', 'total_defense']]
        y = training_df['total_goals']
        
        # Remove NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(X_scaled, y)
        
        # Store coefficients
        self.coefficients = {
            'home_attack': self.model.coef_[0],
            'home_defense': self.model.coef_[1],
            'away_attack': self.model.coef_[2],
            'away_defense': self.model.coef_[3],
            'total_attack': self.model.coef_[4],
            'total_defense': self.model.coef_[5],
            'intercept': self.model.intercept_
        }
        
        return self.coefficients
    
    def predict_total_goals(self, match_features):
        """Predict total goals for a match."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        X = np.array([[
            match_features['home_attack'],
            match_features['home_defense'],
            match_features['away_attack'],
            match_features['away_defense'],
            match_features['total_attack'],
            match_features['total_defense']
        ]])
        
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        
        return max(prediction, 0)
    
    def calculate_implied_probability(self, odds):
        """Convert decimal odds to implied probability."""
        return 1 / odds
    
    def generate_betting_signal(self, predicted_goals, over_odds, under_odds, 
                              threshold=0.55, min_edge=0.02):
        """Generate betting signal based on prediction vs market odds."""
        goal_threshold = 2.5
        std_dev = 1.0
        
        z_score = (goal_threshold - predicted_goals) / std_dev
        
        from scipy.stats import norm
        prob_over = 1 - norm.cdf(z_score)
        
        # Market implied probabilities
        market_prob_over = self.calculate_implied_probability(over_odds)
        market_prob_under = self.calculate_implied_probability(under_odds)
        
        # Calculate edges
        edge_over = prob_over - market_prob_over
        edge_under = (1 - prob_over) - market_prob_under
        
        signal = 'PASS'
        edge = 0
        confidence = 0
        
        if edge_over > min_edge and prob_over > threshold:
            signal = 'OVER'
            edge = edge_over
            confidence = prob_over
        elif edge_under > min_edge and (1 - prob_over) > threshold:
            signal = 'UNDER'
            edge = edge_under
            confidence = 1 - prob_over
        
        return {
            'signal': signal,
            'edge': edge * 100,
            'confidence': confidence * 100,
            'predicted_goals': predicted_goals,
            'prob_over': prob_over * 100,
            'prob_under': (1 - prob_over) * 100
        }
    
    def backtest_csv(self, csv_path):
        """Backtest model on all matches in CSV."""
        # Load and prepare data
        training_data, df = self.prepare_training_data(csv_path)
        
        # Split: train on first 70%, test on last 30%
        split_idx = int(len(training_data) * 0.7)
        train_data = training_data.iloc[:split_idx].copy()
        test_data = training_data.iloc[split_idx:].copy()
        test_matches = df.iloc[split_idx:].reset_index(drop=True)
        
        # Train model
        self.train_model(train_data)
        
        # Test on holdout data
        results = []
        
        for idx, row in test_matches.iterrows():
            test_idx = split_idx + idx
            actual_goals = row['FTHG'] + row['FTAG']
            
            try:
                features = self.create_match_features_from_row(row)
                
                # Get odds
                over_odds = row['B365>2.5'] if not pd.isna(row['B365>2.5']) else 1.90
                under_odds = row['B365<2.5'] if not pd.isna(row['B365<2.5']) else 1.95
                
                # Predict
                prediction = self.predict_total_goals(features)
                signal = self.generate_betting_signal(prediction, over_odds, under_odds)
                
                # Check if correct
                predicted_over = prediction > 2.5
                actual_over = actual_goals > 2.5
                correct = predicted_over == actual_over
                
                results.append({
                    'match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                    'date': row['Date'],
                    'predicted_goals': prediction,
                    'actual_goals': actual_goals,
                    'signal': signal['signal'],
                    'edge': signal['edge'],
                    'correct': correct,
                    'over_odds': over_odds,
                    'under_odds': under_odds
                })
            except:
                continue
        
        results_df = pd.DataFrame(results)
        
        # Calculate statistics
        bettable = results_df[results_df['signal'] != 'PASS']
        
        stats = {
            'total_matches': len(results_df),
            'bettable_matches': len(bettable),
            'bet_rate': len(bettable) / len(results_df) * 100 if len(results_df) > 0 else 0,
            'avg_prediction': results_df['predicted_goals'].mean(),
            'avg_actual': results_df['actual_goals'].mean(),
            'avg_edge': bettable['edge'].mean() if len(bettable) > 0 else 0,
            'accuracy': (results_df['correct'].sum() / len(results_df) * 100) if len(results_df) > 0 else 0,
            'bet_accuracy': (bettable['correct'].sum() / len(bettable) * 100) if len(bettable) > 0 else 0,
            'over_signals': len(bettable[bettable['signal'] == 'OVER']),
            'under_signals': len(bettable[bettable['signal'] == 'UNDER']),
            'results_df': results_df,
            'bettable_df': bettable
        }
        
        return stats


# Main execution
if __name__ == "__main__":
    print("="*70)
    print("PREMIER LEAGUE BETTING MODEL - CSV ADAPTER")
    print("="*70 + "\n")
    
    csv_path = '/mnt/user-data/uploads/E0.csv'
    
    # Initialize model
    model = PremierLeagueBettingModel()
    
    print("1. LOADING AND PREPARING DATA...")
    training_data, df = model.prepare_training_data(csv_path)
    print(f"   ✓ Loaded {len(training_data)} matches")
    print()
    
    print("2. TRAINING MODEL...")
    coefficients = model.train_model(training_data)
    print("   Model coefficients:")
    for feature, coef in coefficients.items():
        if feature != 'intercept':
            print(f"   - {feature}: {coef:.4f}")
    print()
    
    print("3. BACKTESTING ON HOLDOUT DATA...")
    stats = model.backtest_csv(csv_path)
    
    print(f"   Total test matches: {stats['total_matches']}")
    print(f"   Bettable matches: {stats['bettable_matches']} ({stats['bet_rate']:.1f}%)")
    print(f"   - OVER signals: {stats['over_signals']}")
    print(f"   - UNDER signals: {stats['under_signals']}")
    print(f"\n   Average prediction: {stats['avg_prediction']:.2f} goals")
    print(f"   Average actual: {stats['avg_actual']:.2f} goals")
    print(f"   Average edge (bettable): {stats['avg_edge']:.2f}%")
    print(f"\n   Overall accuracy: {stats['accuracy']:.1f}%")
    print(f"   Bet accuracy: {stats['bet_accuracy']:.1f}%")
    print()
    
    print("4. EXAMPLE PREDICTIONS (Most Recent 5 Matches):")
    print()
    
    recent = stats['results_df'].tail(5)
    for idx, row in recent.iterrows():
        status = "✓ CORRECT" if row['correct'] else "✗ INCORRECT"
        print(f"   {row['match']}")
        print(f"   - Predicted: {row['predicted_goals']:.2f} goals | Actual: {row['actual_goals']:.0f}")
        print(f"   - Signal: {row['signal']:6} | Edge: {row['edge']:6.2f}%")
        print(f"   - {status}")
        print()
    
    print("="*70)
    print("MODEL TRAINING COMPLETE")
    print("="*70 + "\n")
