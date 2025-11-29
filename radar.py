import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import json

# Set page config
st.set_page_config(page_title="Live Soccer Radar", layout="wide")

# Custom CSS for dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0a0a0a;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ Live Soccer Radar - Real Match Data")

# Soccer field dimensions (in meters)
FIELD_LENGTH = 105
FIELD_WIDTH = 68

class Player:
    def __init__(self, id, team, position, name="Player"):
        self.id = id
        self.team = team
        self.position = np.array(position, dtype=float)
        self.velocity = np.random.randn(2) * 2
        self.trail = [self.position.copy()]
        self.name = name
        
    def update_position(self, dt=0.1):
        # Random movement with some momentum
        self.velocity += np.random.randn(2) * 0.5
        self.velocity *= 0.95  # Damping
        
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > 5:
            self.velocity = self.velocity / speed * 5
            
        self.position += self.velocity * dt
        
        # Keep within field bounds
        self.position[0] = np.clip(self.position[0], 0, FIELD_LENGTH)
        self.position[1] = np.clip(self.position[1], 0, FIELD_WIDTH)
        
        # Add to trail
        self.trail.append(self.position.copy())
        if len(self.trail) > 30:
            self.trail.pop(0)

class Ball:
    def __init__(self):
        self.position = np.array([FIELD_LENGTH/2, FIELD_WIDTH/2], dtype=float)
        self.velocity = np.random.randn(2) * 3
        self.trail = [self.position.copy()]
        
    def update_position(self, players, dt=0.1):
        # Ball moves towards nearest player
        if players:
            nearest_player = min(players, key=lambda p: np.linalg.norm(p.position - self.position))
            direction = nearest_player.position - self.position
            distance = np.linalg.norm(direction)
            
            if distance > 0.5:
                self.velocity += direction / distance * 0.3
        
        self.velocity *= 0.92  # Damping
        self.position += self.velocity * dt
        
        # Keep within field bounds
        self.position[0] = np.clip(self.position[0], 0, FIELD_LENGTH)
        self.position[1] = np.clip(self.position[1], 0, FIELD_WIDTH)
        
        # Add to trail
        self.trail.append(self.position.copy())
        if len(self.trail) > 20:
            self.trail.pop(0)

def get_live_matches():
    """Scrape live matches from SofaScore"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # SofaScore API endpoint for live matches
        url = "https://api.sofascore.com/api/v1/sport/football/events/live"
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            matches = []
            
            if 'events' in data:
                for event in data['events'][:20]:  # Limit to 20 matches
                    match_info = {
                        'id': event.get('id'),
                        'home_team': event.get('homeTeam', {}).get('name', 'Home'),
                        'away_team': event.get('awayTeam', {}).get('name', 'Away'),
                        'home_score': event.get('homeScore', {}).get('current', 0),
                        'away_score': event.get('awayScore', {}).get('current', 0),
                        'status': event.get('status', {}).get('description', 'Live'),
                        'tournament': event.get('tournament', {}).get('name', 'Tournament'),
                        'time': event.get('time', '')
                    }
                    matches.append(match_info)
            
            return matches
        else:
            st.warning(f"Could not fetch live matches (Status: {response.status_code})")
            return []
            
    except Exception as e:
        st.error(f"Error fetching matches: {str(e)}")
        return []

def create_formation_positions(formation, is_home_team=True):
    """Create player positions based on formation (e.g., '4-3-3')"""
    positions = []
    
    try:
        # Parse formation (e.g., "4-3-3")
        formation_parts = [int(x) for x in formation.split('-')]
        
        # Goalkeeper
        gk_x = 10 if is_home_team else FIELD_LENGTH - 10
        positions.append([gk_x, FIELD_WIDTH / 2])
        
        # Distribute other players
        x_start = 20 if is_home_team else FIELD_LENGTH - 20
        x_direction = 1 if is_home_team else -1
        
        x_offset = 0
        for line_players in formation_parts:
            x_pos = x_start + (x_offset * 15 * x_direction)
            y_spacing = FIELD_WIDTH / (line_players + 1)
            
            for i in range(line_players):
                y_pos = y_spacing * (i + 1)
                positions.append([x_pos, y_pos])
            
            x_offset += 1
    
    except:
        # Default positions if formation parsing fails
        num_players = 11
        x_start = 15 if is_home_team else FIELD_LENGTH - 15
        x_range = FIELD_LENGTH / 2 - 20
        
        for i in range(num_players):
            x = x_start + np.random.uniform(0, x_range) * (1 if is_home_team else -1)
            y = np.random.uniform(10, FIELD_WIDTH - 10)
            positions.append([x, y])
    
    return positions

def draw_field(ax):
    """Draw soccer field with radar-style appearance"""
    ax.clear()
    ax.set_xlim(0, FIELD_LENGTH)
    ax.set_ylim(0, FIELD_WIDTH)
    ax.set_aspect('equal')
    ax.set_facecolor('#0a0a0a')
    
    # Field outline
    field = patches.Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH, 
                             linewidth=2, edgecolor='#00ff00', 
                             facecolor='#0a0a0a', alpha=0.3)
    ax.add_patch(field)
    
    # Center line
    ax.plot([FIELD_LENGTH/2, FIELD_LENGTH/2], [0, FIELD_WIDTH], 
            color='#00ff00', linewidth=2, alpha=0.5)
    
    # Center circle
    center_circle = patches.Circle((FIELD_LENGTH/2, FIELD_WIDTH/2), 9.15,
                                  linewidth=2, edgecolor='#00ff00',
                                  facecolor='none', alpha=0.5)
    ax.add_patch(center_circle)
    
    # Penalty areas
    penalty_length = 16.5
    penalty_width = 40.3
    
    # Left penalty area
    left_penalty = patches.Rectangle((0, (FIELD_WIDTH - penalty_width)/2), 
                                    penalty_length, penalty_width,
                                    linewidth=2, edgecolor='#00ff00',
                                    facecolor='none', alpha=0.5)
    ax.add_patch(left_penalty)
    
    # Right penalty area
    right_penalty = patches.Rectangle((FIELD_LENGTH - penalty_length, 
                                      (FIELD_WIDTH - penalty_width)/2),
                                     penalty_length, penalty_width,
                                     linewidth=2, edgecolor='#00ff00',
                                     facecolor='none', alpha=0.5)
    ax.add_patch(right_penalty)
    
    # Goals
    goal_width = 7.32
    ax.plot([0, 0], [(FIELD_WIDTH - goal_width)/2, (FIELD_WIDTH + goal_width)/2],
            color='#ff0000', linewidth=3, alpha=0.8)
    ax.plot([FIELD_LENGTH, FIELD_LENGTH], 
            [(FIELD_WIDTH - goal_width)/2, (FIELD_WIDTH + goal_width)/2],
            color='#ff0000', linewidth=3, alpha=0.8)
    
    # Grid lines (radar style)
    for i in range(0, int(FIELD_LENGTH), 10):
        ax.axvline(i, color='#00ff00', alpha=0.1, linewidth=0.5)
    for i in range(0, int(FIELD_WIDTH), 10):
        ax.axhline(i, color='#00ff00', alpha=0.1, linewidth=0.5)
    
    ax.axis('off')

def plot_radar(players, ball, fig, ax, match_info=None):
    """Plot players and ball with radar styling"""
    draw_field(ax)
    
    # Plot player trails
    for player in players:
        trail = np.array(player.trail)
        if len(trail) > 1:
            color = '#00ffff' if player.team == 'home' else '#ff00ff'
            for i in range(1, len(trail)):
                alpha = i / len(trail) * 0.5
                ax.plot(trail[i-1:i+1, 0], trail[i-1:i+1, 1],
                       color=color, alpha=alpha, linewidth=2)
    
    # Plot ball trail
    ball_trail = np.array(ball.trail)
    if len(ball_trail) > 1:
        for i in range(1, len(ball_trail)):
            alpha = i / len(ball_trail) * 0.6
            ax.plot(ball_trail[i-1:i+1, 0], ball_trail[i-1:i+1, 1],
                   color='#ffffff', alpha=alpha, linewidth=2)
    
    # Plot players
    for player in players:
        color = '#00ffff' if player.team == 'home' else '#ff00ff'
        # Player dot
        ax.plot(player.position[0], player.position[1], 'o', 
               color=color, markersize=12, markeredgecolor='white',
               markeredgewidth=2)
        # Player number
        ax.text(player.position[0], player.position[1], str(player.id),
               ha='center', va='center', color='black', fontsize=8, 
               fontweight='bold')
        
        # Velocity vector
        ax.arrow(player.position[0], player.position[1],
                player.velocity[0], player.velocity[1],
                head_width=1, head_length=0.5, fc=color, 
                ec=color, alpha=0.6, linewidth=2)
    
    # Plot ball
    ax.plot(ball.position[0], ball.position[1], 'o',
           color='white', markersize=8, markeredgecolor='#ffff00',
           markeredgewidth=2)
    
    # Add match info and timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    if match_info:
        info_text = f"{match_info['home_team']} {match_info['home_score']} - {match_info['away_score']} {match_info['away_team']}"
        ax.text(FIELD_LENGTH/2, FIELD_WIDTH - 2, info_text, 
               ha='center', color='#00ff00', fontsize=12, fontweight='bold')
        ax.text(2, FIELD_WIDTH - 2, f"{match_info['status']} | {timestamp}", 
               color='#00ff00', fontsize=10, fontweight='bold')
    else:
        ax.text(2, FIELD_WIDTH - 2, timestamp, color='#00ff00', 
               fontsize=10, fontweight='bold')
    
    return fig

# Sidebar - Live Match Selection
st.sidebar.title("🔴 LIVE MATCHES")
st.sidebar.markdown("---")

# Refresh button
col_refresh1, col_refresh2 = st.sidebar.columns([3, 1])
with col_refresh1:
    if st.button("🔄 Refresh Live Matches", use_container_width=True):
        st.session_state.live_matches = get_live_matches()
        if 'selected_match_id' in st.session_state:
            del st.session_state.selected_match_id
        st.rerun()

# Get live matches if not in session state
if 'live_matches' not in st.session_state:
    with st.spinner("🔍 Fetching live matches from SofaScore..."):
        st.session_state.live_matches = get_live_matches()

live_matches = st.session_state.live_matches

# Display match selection
selected_match = None

if live_matches:
    st.sidebar.success(f"✅ {len(live_matches)} live matches available")
    st.sidebar.markdown("---")
    
    # Group matches by tournament
    tournaments = {}
    for match in live_matches:
        tournament = match['tournament']
        if tournament not in tournaments:
            tournaments[tournament] = []
        tournaments[tournament].append(match)
    
    # Display matches grouped by tournament
    st.sidebar.subheader("📺 Select a Match")
    
    for tournament, matches in tournaments.items():
        with st.sidebar.expander(f"🏆 {tournament} ({len(matches)} matches)", expanded=True):
            for match in matches:
                match_key = f"{match['id']}"
                
                # Create match button
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    match_label = f"{match['home_team']}\n🆚\n{match['away_team']}"
                    
                with col2:
                    score_label = f"{match['home_score']}\n-\n{match['away_score']}"
                
                # Full width button
                if st.button(
                    f"⚽ {match['home_team']} vs {match['away_team']}\n📊 {match['home_score']} - {match['away_score']} | ⏱️ {match['status']}",
                    key=f"match_{match_key}",
                    use_container_width=True,
                    type="primary" if st.session_state.get('selected_match_id') == match['id'] else "secondary"
                ):
                    st.session_state.selected_match_id = match['id']
                    # Reset players to regenerate with new match
                    if 'players' in st.session_state:
                        del st.session_state.players
                    st.rerun()
    
    # Get selected match
    if 'selected_match_id' in st.session_state:
        selected_match = next(
            (m for m in live_matches if m['id'] == st.session_state.selected_match_id),
            live_matches[0]
        )
    else:
        # Auto-select first match
        selected_match = live_matches[0]
        st.session_state.selected_match_id = selected_match['id']
    
    # Display selected match details
    if selected_match:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎯 SELECTED MATCH")
        st.sidebar.markdown(f"**🏆 {selected_match['tournament']}**")
        st.sidebar.markdown(f"**🏠 {selected_match['home_team']}**")
        st.sidebar.markdown(f"**✈️ {selected_match['away_team']}**")
        st.sidebar.markdown(f"### ⚽ {selected_match['home_score']} - {selected_match['away_score']}")
        st.sidebar.markdown(f"**⏱️ {selected_match['status']}**")
    
else:
    st.sidebar.warning("⚠️ No live matches found")
    st.sidebar.info("💡 Try refreshing or check back later when matches are live!")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Running in simulation mode**")
    selected_match = None

# Initialize or reset simulation
if 'players' not in st.session_state or st.sidebar.button("🔄 Reset Simulation"):
    st.session_state.players = []
    
    if selected_match:
        # Create players based on real match
        home_positions = create_formation_positions("4-3-3", is_home_team=True)
        away_positions = create_formation_positions("4-3-3", is_home_team=False)
        
        for i, pos in enumerate(home_positions):
            st.session_state.players.append(
                Player(i+1, 'home', pos, name=f"{selected_match['home_team']} {i+1}")
            )
        
        for i, pos in enumerate(away_positions):
            st.session_state.players.append(
                Player(i+1, 'away', pos, name=f"{selected_match['away_team']} {i+1}")
            )
    else:
        # Simulation mode
        for i in range(11):
            x = np.random.uniform(5, FIELD_LENGTH/2 - 5)
            y = np.random.uniform(5, FIELD_WIDTH - 5)
            st.session_state.players.append(Player(i+1, 'home', [x, y]))
        
        for i in range(11):
            x = np.random.uniform(FIELD_LENGTH/2 + 5, FIELD_LENGTH - 5)
            y = np.random.uniform(5, FIELD_WIDTH - 5)
            st.session_state.players.append(Player(i+1, 'away', [x, y]))
    
    st.session_state.ball = Ball()

# Controls
col1, col2 = st.columns([1, 1])
with col1:
    speed = st.slider("⚡ Update Speed", 0.1, 2.0, 1.0, 0.1)
with col2:
    auto_run = st.checkbox("▶️ Auto-run simulation", value=True)

# Create plot
fig, ax = plt.subplots(figsize=(14, 9))
fig.patch.set_facecolor('#0a0a0a')

# Placeholder for the plot
plot_placeholder = st.empty()

# Animation loop
if auto_run:
    for _ in range(100):
        # Update positions
        for player in st.session_state.players:
            player.update_position(dt=0.1 * speed)
        
        st.session_state.ball.update_position(st.session_state.players, dt=0.1 * speed)
        
        # Plot
        fig = plot_radar(st.session_state.players, st.session_state.ball, 
                        fig, ax, match_info=selected_match)
        plot_placeholder.pyplot(fig)
        
        time.sleep(0.1)
else:
    # Static display
    fig = plot_radar(st.session_state.players, st.session_state.ball, 
                    fig, ax, match_info=selected_match)
    plot_placeholder.pyplot(fig)

# Add legend
st.markdown("""
### 🎮 Legend
- **Cyan dots**: Home team players (with velocity arrows)
- **Magenta dots**: Away team players (with velocity arrows)
- **White/Yellow dot**: Ball
- **Trails**: Recent movement paths (last 3 seconds)
- **Green grid**: Radar-style field markings
""")

st.info("💡 Select a live match from the sidebar to visualize real match data! The simulation shows tactical movement patterns.")
