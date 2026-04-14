import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import io
import numpy as np
import pandas as pd
import streamlit as st
import requests
from lightgbm import LGBMClassifier
import re

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP Predictor v5.0 - Seu Histórico Completo", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIG
# ==============================================================================
WINNER_SMOOTH = 0.55
MIN_CONFIDENCE_STRONG = 0.68
MIN_CONFIDENCE_GOOD = 0.60

# ==============================================================================
# NOME DO SEU HISTÓRICO (todos os jogadores)
# ==============================================================================
HISTORICAL_PLAYERS = [
    "Mitchell Krueger", "Trevor Svajda", "Yuta Shimizu", "Antoine Escoffier", "Andres Martin",
    "Rio Noguchi", "Nicolas Mejia", "Paul Jubb", "Stefan Dostanic", "Ilya Ivashka",
    "Rafael Jodar", "Patrick Kypson", "Alex Rybakov", "Karue Sell", "Yibing Wu",
    "Yi Zhou", "Daniel Rincon", "Denis Yevseyev", "Guy Den Ouden", "Valentin Royer",
    "Emilio Nava", "Francesco Passaro", "Sumit Nagal", "Marko Topo", "Ignacio Buse",
    "Diego Dedura", "Viktor Durasovic", "Facundo Mena", "Gauthier Onclin", "Rodrigo Pacheco Mendez",
    "Dominic Stricker", "Marco Trungelliti", "Martin Landaluce", "Adrian Mannarino", "Rinky Hijikata",
    "Otto Virtanen", "Nicolas Moreno De Alboran", "Brandon Holt", "Zachary Svajda", "Alex Bolt",
    "Murphy Cassone", "Lloyd Harris", "Marc-Andrea Huesler", "Nicolas Jarry", "Colton Smith",
    "Li Tu", "Yosuke Watanuki", "Coleman Wong", "Vitaliy Sachko", "Alejandro Tabilo",
    "Lukas Neumayer", "Hugo Dellien", "Chun-Hsin Tseng", "Alexander Shevchenko", "Norbert Gombos",
    "Zsombor Piros", "Filip Cristian Jianu", "Milos Karol", "Lukas Klein", "Dimitar Kuzmanov",
    "Alejandro Moro Canas", "Maxim Mrva", "Jurij Rodionov", "Luca Van Assche", "Mariano Kestelboim",
    "Santiago Rodriguez Taverna", "Andrea Collarini", "Matheus Pucinelli De Almeida", "Gonzalo Bueno",
    "Hernan Casanova", "Lautaro Midon", "Dali Blanch", "Luciano Emanuel Ambrogi", "Facundo Bagnis",
    "Pedro Boscardin Dias", "Maximus Jones", "Bruno Kuzuhara", "Joao Lucas Reis Da Silva", "John Sperle",
    "Gonzalo Villanueva", "Pablo Carreno Busta", "Yu Hsiou Hsu", "Arthur Cazaux", "Daniel Merida",
    "Geoffrey Blancaneaux", "Clement Tabur", "Javier Barranco Cosano", "Ivan Gakhov", "Arthur Gea",
    "Lucas Poullain", "Henri Squire", "Adolfo Daniel Vallejo", "Luca Nardi", "Dusan Lajovic",
    "Dalibor Svrcina", "Radu Albot", "Taro Daniel", "Andrea Pellegrino", "Federico Arnaboldi",
    "Pierluigi Basile", "Federico Bondioli", "Nerman Fatic", "Luka Pavlovic", "Stefano Travaglia",
    "Elias Ymer", "Tomas Barrios Vera", "Matej Dodig", "Cristian Garin", "Jan Choinski",
    "Federico Cina", "Toby Kodat", "Dino Prizmic", "Henrique Rocha", "Valentin Vacherot",
    "Sho Shimabukuro", "Oliver Crawford", "Billy Harris", "Marin Cilic", "Johannus Monday",
    "Tristan Schoolkate", "Jenson Brooksby", "Shintaro Mochizuki", "Jack Pinnington Jones", "Leandro Riedi",
    "Henry Searle", "James Trotter", "Calvin Hemery", "Titouan Droguet", "Kilian Feldbausch",
    "Mika Brunold", "Mathys Erhard", "Florent Bax", "Enzo Couacaud", "Buvaysar Gadamauri",
    "Kalin Ivanovski", "Luka Mikrut", "Tom Paris", "Andres Santamarta Roig", "Jaime Faria",
    "Ryan Peniston", "Tristan Boyer", "Terence Atmane", "Arthur Fery", "James McCabe",
    "Thiago Agustin Tirante", "Filip Misolic", "Timofey Skatov", "Hynek Barton", "Lorenzo Giustino",
    "Zdenek Kolar", "Martin Krumich", "Matteo Martineau", "Daniel Michalski", "Juan Bautista Torres",
    "Genaro Alberto Olivieri", "Luis Carlos Alvarez", "Juan Carlos Prado Angelo", "Juan Carlos Aguilar",
    "Alex Barrena", "Lucas Gerch", "Alex Hernandez", "Juan Manuel La Serna", "Franco Roncadelli",
    "Bautista Vilicich", "Carlos Taberner", "Rei Sakamoto", "Raul Brancaccio", "Francesco Maestrelli",
    "Albert Ramos-Vinolas", "Oriol Roca Batalla", "Giulio Zeppieri", "Miguel Tobon", "Blaise Bicknell",
    "Pedro Rodrigues", "Max Houkes", "Tiago Pereira", "Frederico Ferreira Silva", "Mili Poljicak",
    "Jacopo Berrettini", "Marco Cecchinato", "Thomas Faurel", "Christoph Negritu", "Oleg Prihodko",
    "Jacopo Vasami", "Harold Mayot", "Cosme Rolland De Ravel", "Tristan Lamasine", "Branko Djuric",
    "Jelle Sels", "Nikolas Sanchez Izquierdo", "Daniel Elahi Galan", "Juan Manuel Cerundolo",
    "Thiago Seyboth Wild", "Moez Echargui", "Facundo Diaz Acosta", "Jerome Kym", "Stefanos Sakellaridis",
    "Juan Pablo Ficovich", "Alvaro Guillen Meza", "Sascha Gueymard Wayenburg", "Murkel Dellien",
    "Edas Butvilas", "Gabi Adrian Boitan", "Manas Dhamne", "Alexander Donski", "Radu Mihai Papoe",
    "Liam Draxl", "Michael Zheng", "Alexis Galarneau", "Hiroki Moriya", "Kaylan Bigun",
    "Daniil Glinka", "Andre Ilagan", "Patrick Maloney", "Aidan Mayo", "Joshua Sheehy",
    "Lukas Pokorny", "Dennis Novak", "Sandro Kopp", "Chris Rodesch", "Anton Matusevich",
    "Liam Broady", "Edward Winter", "Kenny De Schepper", "George Loffhagen", "Alastair Gray",
    "Kyle Edmund", "Lui Maxted", "Hamish Stewart", "Connor Thomson", "Harry Wendelken",
    "Patrick Zahraj", "Eliot Spizzirri", "Christopher Eubanks", "Antoine Ghibaudo", "Bernard Tomic",
    "James Watt", "Oscar Weightman", "Tyler Zink", "Roberto Carballes Baena", "Raphael Collignon",
    "Mariano Navone", "Botic van de Zandschulp", "Tom Gentzsch", "Yannick Hanfmann", "David Jorda Sanchis",
    "Alex Molcan", "Alexander Blockx", "Andres Andrade", "Nicolas Arseneault", "Moerani Bouzige",
    "Masamichi Imamura", "Garrett Johns", "Kenta Miyoshi", "Elmer Moller", "Felipe Meligeni Alves",
    "Stan Wawrinka", "Duje Ajdukovic", "Luca Preda", "Matias Soto", "Michael Vrbensky",
    "Ugo Blanchet", "Hugo Grenier", "Fajing Sun", "Khumoyun Sultanov", "Darwin Blanch",
    "Gastao Elias", "Aryan Shah", "August Holmgren", "Omar Jasika", "Marek Gengel",
    "Kris Van Wyk", "Marvin Moeller", "Benjamin Hassan", "Kimmer Coppejans", "Nicolas Kicker",
    "Matteo Gigante", "Kyrian Jacquet", "Federico Agustin Gomez", "Carlo Alberto Caniato", "Justin Engel",
    "Gilles Arnaud Bailly", "Nicolai Budkov Kjaer", "Henry Bernet", "Roman Andres Burruchaga", "Jakub Paul",
    "Mark Lajal", "Tung-Lin Wu", "Aziz Dougaz", "Adria Soriano Barrera", "Nicolas Alvarez Varona",
    "Robin Bertrand", "Inaki Montes-De La Torre", "Saba Purtseladze", "Nishesh Basavareddy", "Daniel Evans",
    "Dhakshineswar Suresh", "Hady Habib", "Giles Hussey", "Alibek Kachmazov", "Ryuki Matsuda",
    "Federico Coria", "Andrej Martin", "Jakub Nicod", "Cedrik-Marcel Stebe", "Clement Chidekh",
    "Mukund Sasikumar", "Alexandr Binda", "Abdullah Shelbayh", "Ricardas Berankis", "Aleksandre Bakshi",
    "Luca Castelnuovo", "Shintaro Imai", "Mitsuki Wei Kang Leong", "Olaf Pieczkowski", "Woobin Shin",
    "Karan Singh", "Vilius Gaubas", "Niels McDonald", "Daniel Masur", "Dmitry Popko",
    "Joel Schwaerzler", "Olle Wallin", "Jay Clarke", "Max Alcala Gurri", "Gabriele Piraino",
    "Carlos Sanchez Jover", "Kamil Majchrzak", "Petr Brunclik", "Maks Kasnikowski", "Tomasz Berkieta",
    "Jie Cui", "Aslan Karatsev", "Beibit Zhukayev", "Samir Banerjee", "Naoki Nakagawa",
    "Stefan Adrian Andreescu", "Marat Sharipov", "Sebastian Gima", "Ioannis Xilas", "Pietro Fellin",
    "Dan Added", "Laurent Lokoli", "Filippo Moroni", "Theo Papamalamis", "Oliver Tarvet",
    "Max Wiskandt", "Cannon Kingsley", "Peter Bertran", "Juan Sebastian Gomez", "Alfredo Perez",
    "Johan Alexander Rodriguez", "Facundo Juarez", "Gianmarco Ferrari", "Francesco Forti", "Filippo Romano",
    "Alexey Vatutin", "Nikoloz Basilashvili", "Mattia Bellucci", "Alexander Vasilev", "Franco Agamenone",
    "Andrej Nedic", "Stuart Parker", "Christian Langmo", "Pavlos Tsitsipas", "Tadeas Paroulek",
    "Alexander Ritschard", "Matyas Cerny", "Stefan Popovic", "Benito Sanchez Martinez", "Max Schoenhaus",
    "Pedro Araujo", "Francisco Rocha", "Tiago Torres", "Thiago Monteiro", "Lorenzo Carboni",
    "Giovanni Fonio", "Philip Sekulic", "Yasutaka Uchiyama", "Ilia Simakin", "Joshua Charlton",
    "Tsung-Hao Huang", "Kokoro Isomura", "Filip Peliwo", "Sanhui Shin", "Renta Tokuda",
    "Kaichi Uchida", "Alberto Barroso Campos", "Eliakim Coulibaly", "Jay Friend", "Andrea Picchione",
    "Gianluca Cadenasso", "Enrico Dalla Valle", "Luciano Darderi", "Luca Potenza", "Mert Naci Turker",
    "Pablo Llamas Ruiz", "Jonas Forejtek", "Carlos Lopez Montagud", "Etienne Donnet", "Mae Malige",
    "Michael Mmoh", "Rudolf Molleker", "Andrew Paulson", "Kai Wehnelt", "Christopher O'Connell",
    "Petr Bar Biryukov", "Luca Pow", "Aidan Kim", "Quinn Vandecasteele", "Hugo Gaston",
    "Arthur Bouquier", "Gregoire Barrere", "Maxime Janvier", "Vit Kopriva", "Jozef Kovalik",
    "Pol Martin Tiffon", "Stefano Napolitano", "Robert Strombachs", "Guido Ivan Justo", "Juan Pablo Varillas",
    "Sebastian Sorger", "Corentin Denolly", "Miguel Damas", "Jack Loge", "Amaury Raynel",
    "Justin Boulais", "Mikhail Kukushkin", "Cezar Cretu", "Alex Marti Pujolras", "Maximilian Neuchrist",
    "Emilien Demanet", "Stefan Palosi", "Remy Bertola", "Pierre-Hugues Herbert", "Andrew Fenty",
    "Cooper Williams", "Lorenzo Joaquin Rodriguez", "Aristotelis Thanos", "Carlos Maria Zarate",
    "Mees Rottgering", "Loann Massard", "Emil Ruusuvuori", "Neil Oberleitner", "Johannes Ingildsen",
    "Charles Broom", "Toby Samuel", "Petros Tsitsipas", "Niels Visker", "Micah Braswell",
    "Daniel Milavsky", "Roger Pascual Ferra", "Dominique Rolland", "Adam Walton", "Mackenzie McDonald",
    "Pavel Kotov", "Jordan Thompson", "Jacob Fearnley", "Borna Gojo", "Daniel Dutra da Silva",
    "Mateus Alves", "Joao Eduardo Schiessl", "Aoran Wang", "Hyeon Chung", "Sergey Fomin",
    "Jason Jung", "Ye Cong Mo", "Fabrizio Andaloro", "Max Basing", "Erik Arutiunian",
    "Gustavo Heide", "Naoya Honda", "Takuya Kumasaka", "Yusuke Takahashi", "Kei Nishikori",
    "Igor Marcondes", "Yusuke Kusuhara", "Yuki Mochizuki", "Hikaru Shiraishi", "Ryotaro Taguchi",
    "Maxim Zhukov", "Joaquin Aguilar Cardozo", "Juan Estevez", "Luis David Martinez", "Paulo Andre Saraiva Dos Santos",
    "Andrea Guerrieri", "Lorenzo Sciahbasi", "Pedro Sakamoto", "Bruno Fernandez", "Orlando Luz",
    "Eduardo Ribeiro", "Maximo Zeitune", "Lucio Ratti", "Alafia Ayeni", "Strong Kirchheimer",
    "Matt Kuhar", "Daniel Antonio Nunez", "Nicolas Barrientos", "Maxwell Mckennon", "Menelaos Efstathiou",
    "Aleksandr Braynin", "Philip Henning", "Michele Ribecai", "James Story", "Nikolay Vylegzhanin",
    "Fares Zakaria", "Vladislav Melnic", "Mirza Basic", "Matisse Bobichon", "Maximilian Homberg",
    "Dominik Kellovsky", "Ivan Marrero Curbelo", "Semen Pankin", "Eero Vasa", "Nikita Bilozertsev",
    "Vadym Ursu", "Mert Alkaya", "Leo Borg", "Dominik Palan", "Leonardo Borrelli",
    "Dragos Nicolae Cazacu", "Timofei Derepasko", "Dinko Dinev", "Muzammil Murtaza", "Johan Nikles",
    "Zach Stephens", "Gian Luca Tanner", "Pavle Marinkov", "Marton Fucsovics", "Jesper de Jong",
    "Andrea Vavassori", "Arthur Rinderknech", "Alex Michelsen", "Quentin Halys", "Learner Tien",
    "Brandon Nakashima", "Jiri Lehecka", "Giovanni Mpetshi Perricard", "Jan-Lennard Struff", "Alexander Zverev",
    "Felix Auger-Aliassime", "Taylor Fritz", "Ben Shelton", "Hubert Hurkacz", "Zizou Bergs",
    "Nuno Borges", "Gabriel Diallo", "Daniil Medvedev", "Ugo Humbert", "Karen Khachanov",
    "Carlos Alcaraz", "Jack Draper", "Corentin Moutet", "Jaume Munar", "Alexei Popyrin",
    "Holger Rune", "Jakub Mensik", "Roberto Bautista Agut", "Jannik Sinner", "Andrey Rublev",
    "Tomas Machac", "Flavio Cobolli", "Tomas Martin Etcheverry", "Lorenzo Sonego", "Alexander Bublik",
    "Fabian Marozsan", "Stefanos Tsitsipas", "Roman Safiullin", "Hamad Medjedovic", "Ethan Quinn",
    "Daniel Altmaier", "Laslo Djere", "Tallon Griekspoor", "Marcos Giron", "Alejandro Davidovich Fokina",
    "Joao Fonseca", "Novak Djokovic", "Alex de Minaur", "Tommy Paul", "Gael Monfils",
    "Grigor Dimitrov", "Miomir Kecmanovic", "Pedro Martinez", "Sebastian Ofner", "Aleksandar Vukic",
    "Benjamin Bonzi", "Frances Tiafoe", "Cameron Norrie", "Matteo Arnaldi", "Alexandre Muller",
    "Yoshihito Nishioka", "Lorenzo Musetti", "Casper Ruud", "Francisco Cerundolo", "Sebastian Baez",
    "Camilo Ugo Carabelli", "Damir Dzumhur", "Juan Pablo Varillas", "Thiago Seyboth Wild", "Facundo Diaz Acosta",
    "Juan Pablo Ficovich", "Alvaro Guillen Meza", "Sascha Gueymard Wayenburg", "Murkel Dellien", "Edas Butvilas"
]

# ==============================================================================
# NAME MATCHER
# ==============================================================================
class HistoricalNameMatcher:
    def __init__(self):
        self.historical_set = set(HISTORICAL_PLAYERS)
        self.lower_map = {p.lower(): p for p in HISTORICAL_PLAYERS}
        self.last_name_map = defaultdict(list)
        for p in HISTORICAL_PLAYERS:
            parts = p.split()
            if parts:
                last = parts[-1].lower()
                self.last_name_map[last].append(p)
    
    def find_match(self, name):
        if not name:
            return None
        
        name_str = str(name).strip()
        
        # Exact match
        if name_str in self.historical_set:
            return name_str
        
        # Case insensitive
        name_lower = name_str.lower()
        if name_lower in self.lower_map:
            return self.lower_map[name_lower]
        
        # Last name match (if unique)
        parts = name_str.split()
        if parts:
            last = parts[-1].lower()
            if last in self.last_name_map:
                if len(self.last_name_map[last]) == 1:
                    return self.last_name_map[last][0]
        
        # Partial match (contains)
        for p in HISTORICAL_PLAYERS:
            if name_lower in p.lower() or p.lower() in name_lower:
                return p
        
        return None

# ==============================================================================
# SURFACE DETECTION
# ==============================================================================
def detect_surface(tournament_name):
    if pd.isna(tournament_name):
        return 'Hard'
    t = str(tournament_name).lower()
    clay = ['clay', 'monte carlo', 'madrid', 'rome', 'barcelona', 'munich', 'roland garros']
    grass = ['grass', 'wimbledon', 'queens', 'halle']
    if any(k in t for k in clay):
        return 'Clay'
    if any(k in t for k in grass):
        return 'Grass'
    return 'Hard'

# ==============================================================================
# PROCESS DATA
# ==============================================================================
def process_historical_data(df):
    df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
    
    # Find columns
    winner_col = None
    loser_col = None
    for col in df.columns:
        if 'winner' in col or 'vencedor' in col:
            winner_col = col
        elif 'loser' in col or 'perdedor' in col:
            loser_col = col
    
    if not winner_col or not loser_col:
        raise ValueError(f"Colunas não encontradas. Colunas: {list(df.columns)}")
    
    df = df.rename(columns={winner_col: 'winner', loser_col: 'loser'})
    
    if 'date' not in df.columns:
        df['date'] = pd.Timestamp.now()
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    if 'total_games' not in df.columns:
        df['total_games'] = 22
    
    df['surface'] = 'Hard'
    
    df['winner'] = df['winner'].astype(str).str.strip()
    df['loser'] = df['loser'].astype(str).str.strip()
    
    return df

# ==============================================================================
# PLAYER STATS
# ==============================================================================
def calculate_player_stats(df):
    stats = {}
    for player in HISTORICAL_PLAYERS:
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        if len(matches) == 0:
            stats[player] = {
                'matches': 0, 'wins': 0, 'win_rate': 0.5,
                'recent_form': 0.5, 'very_recent_form': 0.5, 'avg_games': 22
            }
            continue
        
        wins = len(matches[matches['winner'] == player])
        total = len(matches)
        win_rate = wins / total if total > 0 else 0.5
        
        recent = matches.sort_values('date', ascending=False).head(10)
        recent_wins = len(recent[recent['winner'] == player])
        recent_form = recent_wins / len(recent) if len(recent) > 0 else 0.5
        
        very_recent = matches.sort_values('date', ascending=False).head(3)
        very_recent_wins = len(very_recent[very_recent['winner'] == player])
        very_recent_form = very_recent_wins / len(very_recent) if len(very_recent) > 0 else 0.5
        
        avg_games = matches['total_games'].mean() if 'total_games' in matches.columns else 22
        
        stats[player] = {
            'matches': total, 'wins': wins, 'win_rate': win_rate,
            'recent_form': recent_form, 'very_recent_form': very_recent_form, 'avg_games': avg_games
        }
    
    return stats

# ==============================================================================
# H2H
# ==============================================================================
def calculate_h2h(df):
    h2h = defaultdict(lambda: {'wins': 0, 'total': 0})
    for _, row in df.iterrows():
        w, l = row['winner'], row['loser']
        if w and l and w in HISTORICAL_PLAYERS and l in HISTORICAL_PLAYERS:
            h2h[(w, l)]['wins'] += 1
            h2h[(w, l)]['total'] += 1
    return h2h

# ==============================================================================
# ELO
# ==============================================================================
def calculate_elo(df, k=32):
    elo = {p: 1500 for p in HISTORICAL_PLAYERS}
    for _, row in df.sort_values('date').iterrows():
        w, l = row['winner'], row['loser']
        if w in elo and l in elo:
            exp_w = 1 / (1 + 10 ** ((elo[l] - elo[w]) / 400))
            elo[w] += k * (1 - exp_w)
            elo[l] += k * (0 - (1 - exp_w))
    return elo

# ==============================================================================
# FEATURES
# ==============================================================================
def build_features(p1, p2, surface, player_stats, h2h, elo):
    s1 = player_stats.get(p1, {})
    s2 = player_stats.get(p2, {})
    
    if s1.get('matches', 0) == 0 or s2.get('matches', 0) == 0:
        return None
    
    elo_diff = (elo.get(p1, 1500) - elo.get(p2, 1500)) / 400
    form_diff = s1.get('recent_form', 0.5) - s2.get('recent_form', 0.5)
    very_recent_diff = s1.get('very_recent_form', 0.5) - s2.get('very_recent_form', 0.5)
    win_rate_diff = s1.get('win_rate', 0.5) - s2.get('win_rate', 0.5)
    
    h2h_adv = 0.5
    if (p1, p2) in h2h:
        h2h_adv = h2h[(p1, p2)]['wins'] / h2h[(p1, p2)]['total']
    
    games_avg = (s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 2
    games_norm = (games_avg - 21.5) / 8
    exp_diff = (s1.get('matches', 0) - s2.get('matches', 0)) / 200
    momentum = very_recent_diff * 0.6 + form_diff * 0.4
    
    return [elo_diff, form_diff, very_recent_diff, win_rate_diff, h2h_adv, games_norm, exp_diff, momentum]

# ==============================================================================
# TRAIN
# ==============================================================================
def train_model(df, player_stats, h2h, elo):
    X, y = [], []
    for _, row in df.iterrows():
        w, l = row['winner'], row['loser']
        surface = row.get('surface', 'Hard')
        
        feat = build_features(w, l, surface, player_stats, h2h, elo)
        if feat:
            X.append(feat)
            y.append(1)
        
        feat_rev = build_features(l, w, surface, player_stats, h2h, elo)
        if feat_rev:
            X.append(feat_rev)
            y.append(0)
    
    if len(X) == 0:
        raise ValueError("No training data")
    
    X = np.array(X)
    model = LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.035,
                           num_leaves=16, reg_alpha=0.8, reg_lambda=0.8,
                           random_state=42, verbose=-1)
    model.fit(X, y)
    return model

# ==============================================================================
# PREDICT
# ==============================================================================
def predict_match(model, p1, p2, surface, player_stats, h2h, elo, name_matcher):
    p1_match = name_matcher.find_match(p1)
    p2_match = name_matcher.find_match(p2)
    
    if not p1_match:
        return None, f"❌ '{p1}' não encontrado"
    if not p2_match:
        return None, f"❌ '{p2}' não encontrado"
    
    features = build_features(p1_match, p2_match, surface, player_stats, h2h, elo)
    if not features:
        return None, f"Estatísticas insuficientes"
    
    prob = model.predict_proba(np.array([features]))[0][1]
    prob_p1 = np.clip(0.5 + (prob - 0.5) * 0.55, 0.15, 0.85)
    confidence = abs(prob_p1 - 0.5) * 2
    winner = p1_match if prob_p1 > 0.5 else p2_match
    
    if confidence >= 0.68:
        rec = f"🔥 STRONG {winner}"
    elif confidence >= 0.60:
        rec = f"✅ GOOD {winner}"
    else:
        rec = f"⚪ AVOID {winner}"
    
    s1 = player_stats.get(p1_match, {})
    s2 = player_stats.get(p2_match, {})
    momentum_edge = (s1.get('very_recent_form', 0.5) - s2.get('very_recent_form', 0.5)) * 100
    exp_games = (s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 2
    
    return {
        'Jogador1_Original': p1,
        'Jogador2_Original': p2,
        'Match_Historico': f"{p1_match} vs {p2_match}",
        'Superficie': surface,
        'Prob_P1': f"{prob_p1:.1%}",
        'Prob_P2': f"{1-prob_p1:.1%}",
        'Vencedor': winner,
        'Confianca': f"{confidence:.1%}",
        'Recomendacao': rec,
        'Momentum': f"{momentum_edge:+.0f}",
        'Games_Esperados': round(exp_games, 1)
    }, None

# ==============================================================================
# SCRAPER
# ==============================================================================
def scrape_matches():
    try:
        target_date = datetime.now().strftime("%Y-%m-%d")
        url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{target_date}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return []
        
        data = r.json()
        matches = []
        for ev in data.get("events", []):
            category = ev.get("tournament", {}).get("category", {}).get("name", "")
            if "WTA" in str(category).upper():
                continue
            
            matches.append({
                "tournament": ev["tournament"]["name"],
                "player1": ev["homeTeam"]["name"],
                "player2": ev["awayTeam"]["name"],
                "surface": detect_surface(ev["tournament"]["name"])
            })
        return matches
    except:
        return []

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v5.0 - Seu Histórico Completo")
    st.caption(f"✅ {len(HISTORICAL_PLAYERS)} jogadores no histórico")
    
    name_matcher = HistoricalNameMatcher()
    
    uploaded_file = st.file_uploader("📁 Upload do ficheiro histórico", type=['xlsx', 'csv'])
    
    if uploaded_file and 'model' not in st.session_state:
        with st.spinner("🔄 Processando..."):
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                df = process_historical_data(df)
                st.success(f"✅ {len(df)} jogos carregados")
                
                player_stats = calculate_player_stats(df)
                h2h = calculate_h2h(df)
                elo = calculate_elo(df)
                model = train_model(df, player_stats, h2h, elo)
                
                st.session_state.model = model
                st.session_state.player_stats = player_stats
                st.session_state.h2h = h2h
                st.session_state.elo = elo
                st.session_state.name_matcher = name_matcher
                st.session_state.models_ready = True
                
                st.success("✅ Modelo treinado!")
                
            except Exception as e:
                st.error(f"Erro: {e}")
    
    if st.session_state.get('models_ready'):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 HOJE", use_container_width=True):
                st.session_state.matches = scrape_matches()
        with col2:
            if st.button("🔄 TESTAR", use_container_width=True):
                test = st.text_input("Digite um nome para testar:")
                if test:
                    result = st.session_state.name_matcher.find_match(test)
                    st.write(f"Resultado: {result}")
        
        # Manual
        with st.expander("✏️ Previsão Manual", expanded=True):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                manual_p1 = st.selectbox("Jogador 1", [""] + sorted(HISTORICAL_PLAYERS)[:200])
            with col_b:
                manual_p2 = st.selectbox("Jogador 2", [""] + sorted(HISTORICAL_PLAYERS)[:200])
            with col_c:
                manual_surface = st.selectbox("Superfície", ["Clay", "Hard", "Grass"])
            
            if st.button("🔮 PREVER") and manual_p1 and manual_p2:
                result, error = predict_match(
                    st.session_state.model, manual_p1, manual_p2, manual_surface,
                    st.session_state.player_stats, st.session_state.h2h, st.session_state.elo,
                    st.session_state.name_matcher
                )
                if result:
                    st.dataframe(pd.DataFrame([result]), use_container_width=True)
                else:
                    st.error(error)
        
        # Predictions
        if st.session_state.get('matches'):
            st.subheader("🎯 Previsões")
            results = []
            errors = []
            for match in st.session_state.matches:
                result, error = predict_match(
                    st.session_state.model, match['player1'], match['player2'], match['surface'],
                    st.session_state.player_stats, st.session_state.h2h, st.session_state.elo,
                    st.session_state.name_matcher
                )
                if result:
                    result['Torneio'] = match['tournament']
                    results.append(result)
                elif error:
                    errors.append(error)
            
            if errors:
                with st.expander(f"⚠️ {len(errors)} não encontrados"):
                    for e in set(errors):
                        st.write(e)
            
            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button("📥 Download", buffer.getvalue(),
                                 f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")

if __name__ == "__main__":
    main()
