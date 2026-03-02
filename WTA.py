import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# PAGE CONFIGURATION
st.set_page_config(
    page_title="WTA Match Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MODEL TRAINING FUNCTION (CACHED)
@st.cache_resource
def load_and_train_model(csv_file):
    """Load CSV and train the Random Forest model"""
    df = pd.read_csv(csv_file)
    
    for col in ['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['Player_1', 'Player_2', 'Winner', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2'])
    df['Player_1_Won'] = (df['Winner'] == df['Player_1']).astype(int)
    
    features = []
    feature_names = []
    
    features.append((df['Rank_2'] - df['Rank_1']).values)
    feature_names.append('Ranking_Differential')
    
    features.append((df['Pts_1'] - df['Pts_2']).values)
    feature_names.append('Points_Differential')
    
    features.append(df['Rank_1'].values)
    feature_names.append('Player_1_Rank')
    
    if 'Surface' in df.columns:
        surfaces = pd.get_dummies(df['Surface'], prefix='Surface')
        for col in surfaces.columns:
            features.append(surfaces[col].values)
            feature_names.append(col)
    
    if 'Round' in df.columns:
        rounds = pd.get_dummies(df['Round'], prefix='Round')
        for col in rounds.columns:
            features.append(rounds[col].values)
            feature_names.append(col)
    
    if 'Court' in df.columns:
        courts = pd.get_dummies(df['Court'], prefix='Court')
        for col in courts.columns:
            features.append(courts[col].values)
            feature_names.append(col)
    
    if 'Odd_1' in df.columns and 'Odd_2' in df.columns:
        features.append((df['Odd_1'] - df['Odd_2']).values)
        feature_names.append('Odds_Differential')
    
    X = np.column_stack(features)
    y = df['Player_1_Won'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.tr
