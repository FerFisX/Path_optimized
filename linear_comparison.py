"""
NASA HERC 2024 - Strategy Comparison Module (Baseline)
Model: Linear Regression (OLS)
Description: Compares historical human performance against a Linear Learning-to-Rank agent.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Configuration
TIME_LIMIT = 480  # seconds
BYPASS_COST = 15.0 # seconds

class SimulationEnvironment:
    """Generates synthetic training data for the linear model."""
    
    def generate_training_data(self, n_samples=5000):
        np.random.seed(42)
        data = {
            'difficulty': np.random.randint(1, 11, n_samples),
            'points': np.random.choice([10, 20, 30, 40, 50], n_samples),
            'risk_prob': np.random.beta(2, 5, n_samples)
        }
        df = pd.DataFrame(data)
        
        # Physical constraints simulation
        mu = np.log(10 + df['difficulty'] * 5)
        df['avg_time'] = np.random.lognormal(mean=mu, sigma=0.4)
        
        # Linear Efficiency Target (Points / Time - Linear Risk Penalty)
        penalty = df['risk_prob'] * 10
        df['efficiency'] = (df['points'] / df['avg_time']) - penalty
        
        return df

class LinearAgent:
    """Agent that uses Linear Regression to make routing decisions."""
    
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self):
        print("[System] Generating synthetic data and training Linear Model...")
        sim = SimulationEnvironment()
        df = sim.generate_training_data()
        X = df[['difficulty', 'avg_time', 'risk_prob', 'points']]
        y = df['efficiency']
        self.model.fit(X, y)
        print("[System] Training complete.")

    def predict_value(self, obstacle_features):
        return self.model.predict(obstacle_features)[0]

    def solve_route(self, team_row):
        """
        reconstructs the race scenario for a specific team and finds the optimal path
        according to the linear model.
        """
        # Beam Search State: (accumulated_model_score, time_spent, path_history, actual_points)
        beam = [(0.0, 0.0, [], 0)]
        
        for i in range(1, 11):
            # Extract obstacle data from CSV row
            obs_key = f"O{i}"
            
            # Handle missing data or zeros in CSV
            try:
                real_time = float(team_row.get(f'{obs_key}_time', 0))
                real_points = float(team_row.get(f'{obs_key}_points', 0))
            except ValueError:
                real_time, real_points = 0.0, 0.0

            # Estimation logic: 
            # If human attempted it (time > 0), use real time.
            # If bypassed, use a statistical estimate for planning.
            est_time = real_time if real_time > 1.0 else 30.0 + (i * 2)
            est_points = real_points if real_points > 0 else (i * 5) + 10

            # Create features for the model
            features = pd.DataFrame([{
                'difficulty': i,
                'avg_time': est_time,
                'risk_prob': 0.1 + (i * 0.05),
                'points': est_points
            }])
            
            # Get Linear Model Score
            predicted_value = self.predict_value(features)
            
            candidates = []
            for mod_score, time, path, pts in beam:
                # Option 1: ATTEMPT
                new_time = time + est_time
                if new_time <= TIME_LIMIT:
                    candidates.append((
                        mod_score + predicted_value, 
                        new_time, 
                        path + ['Attempt'], 
                        pts + est_points
                    ))
                
                # Option 2: BYPASS
                new_time_byp = time + BYPASS_COST
                if new_time_byp <= TIME_LIMIT:
                    candidates.append((
                        mod_score, 
                        new_time_byp, 
                        path + ['Bypass'], 
                        pts
                    ))
            
            # Pruning (Beam Width = 20)
            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[:20]

        # Return the path with the highest accumulated POINTS (not model score)
        best_path = sorted(beam, key=lambda x: x[3], reverse=True)[0]
        return best_path

def analyze_dataset(file_path):
    agent = LinearAgent()
    agent.train()
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"[Error] File {file_path} not found.")
        return

    print(f"\n[Analysis] Processing {len(df)} teams from {file_path}...")
    print("-" * 100)
    print(f"{'Team':<10} | {'Human Strategy':<35} | {'Linear Agent Strategy':<35} | {'Diff'}")
    print("-" * 100)

    total_diff = 0
    
    for index, row in df.iterrows():
        team_id = row.get('team_number', index)
        
        # 1. Analyze Human Performance
        h_points = 0
        h_time = 0
        h_path_str = ""
        
        for i in range(1, 11):
            t = float(row.get(f'O{i}_time', 0))
            p = float(row.get(f'O{i}_points', 0))
            
            if t > 10: # Assuming attempts take > 10s
                h_points += p
                h_time += t
                h_path_str += "A"
            else:
                h_time += BYPASS_COST
                h_path_str += "."
        
        h_status = "OK" if h_time <= TIME_LIMIT else "DNF"
        
        # 2. Analyze AI Performance
        _, ai_time, _, ai_points = agent.solve_route(row)
        ai_status = "OK" if ai_time <= TIME_LIMIT else "DNF"
        
        # 3. Output
        diff = ai_points - h_points
        total_diff += diff
        
        h_display = f"{int(h_points)}pts ({int(h_time)}s) [{h_path_str}]"
        ai_display = f"{int(ai_points)}pts ({int(ai_time)}s)"
        
        print(f"{str(team_id):<10} | {h_display:<35} | {ai_display:<35} | {int(diff):+d}")

    print("-" * 100)
    print(f"Average Improvement per Team: {total_diff / len(df):.2f} points")

if __name__ == "__main__":
    analyze_dataset('herc_data.csv')
