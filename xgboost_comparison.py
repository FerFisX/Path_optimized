"""
NASA HERC 2024 - Strategy Comparison Module (Proposed)
Model: XGBoost (Gradient Boosting Trees)
Description: Compares historical human performance against the proposed Non-Linear L2R agent.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# Configuration
TIME_LIMIT = 480
BYPASS_COST = 15.0

class ComplexSimulation:
    """Generates synthetic data with non-linear risk penalties."""
    
    def generate_training_data(self, n_samples=10000):
        np.random.seed(42)
        data = {
            'difficulty': np.random.randint(1, 11, n_samples),
            'points': np.random.choice([10, 20, 30, 40, 50], n_samples),
            'risk_prob': np.random.beta(2, 5, n_samples)
        }
        df = pd.DataFrame(data)
        
        # Log-Normal time distribution
        mu = np.log(10 + df['difficulty'] * 5)
        df['avg_time'] = np.random.lognormal(mean=mu, sigma=0.4)
        
        # Non-Linear Efficiency Target
        # Cubic penalty for risk: high risk drastically reduces efficiency
        penalty = df['risk_prob'] ** 3
        df['efficiency'] = (df['points'] / df['avg_time']) * (1 - penalty)
        
        return df

class XGBoostAgent:
    """Agent that uses Gradient Boosting to rank obstacles and optimize paths."""
    
    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        
    def train(self):
        print("[System] Generating 10,000 simulations and training XGBoost...")
        sim = ComplexSimulation()
        df = sim.generate_training_data()
        X = df[['difficulty', 'avg_time', 'risk_prob', 'points']]
        y = df['efficiency']
        self.model.fit(X, y)
        print("[System] XGBoost model ready.")

    def optimize_path(self, team_row):
        """
        Uses Beam Search guided by XGBoost predictions.
        """
        # Beam State: (heuristic_score, time_spent, path_history, actual_points)
        beam = [(0.0, 0.0, [], 0)]
        
        for i in range(1, 11):
            obs_key = f"O{i}"
            
            # Parse CSV data
            try:
                real_time = float(team_row.get(f'{obs_key}_time', 0))
                real_points = float(team_row.get(f'{obs_key}_points', 0))
            except ValueError:
                real_time, real_points = 0.0, 0.0
            
            # Estimation for decision making
            # If the human bypassed (time ~ 0), the AI must estimate difficulty
            est_time = real_time if real_time > 1.0 else 30.0 + (i * 3)
            est_points = real_points if real_points > 0 else (i * 5)
            
            # Feature vector for inference
            # We approximate risk based on time duration and difficulty
            est_risk = 0.1 + (est_time / 300.0) 
            
            features = pd.DataFrame([{
                'difficulty': i,
                'avg_time': est_time,
                'risk_prob': est_risk,
                'points': est_points
            }])
            
            # Predict Efficiency Score
            ai_score = self.model.predict(features)[0]
            
            candidates = []
            for score, time, path, pts in beam:
                # 1. ATTEMPT
                new_time = time + est_time
                if new_time <= TIME_LIMIT:
                    candidates.append((
                        score + ai_score, 
                        new_time, 
                        path + ['A'], 
                        pts + est_points
                    ))
                
                # 2. BYPASS
                new_time_byp = time + BYPASS_COST
                if new_time_byp <= TIME_LIMIT:
                    candidates.append((
                        score, 
                        new_time_byp, 
                        path + ['.'], 
                        pts
                    ))
            
            # Beam Pruning
            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[:20]
            
        # Return best path
        best_solution = sorted(beam, key=lambda x: x[2], reverse=True)[0]
        return best_solution

def compare_strategies(file_path):
    agent = XGBoostAgent()
    agent.train()
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"[Error] Could not find {file_path}")
        return

    print("\n" + "="*95)
    print(f" FINAL COMPARATIVE REPORT: HUMAN TEAMS vs. XGBOOST AGENT")
    print("="*95)
    print(f"{'Team':<8} | {'Human Results (Pts/Time/Path)':<35} | {'XGBoost Results (Pts/Time/Path)':<35} | {'Gain'}")
    print("-" * 95)
    
    improvements = []

    for index, row in df.iterrows():
        team_id = row.get('team_number', index)
        
        # --- Human Metrics ---
        h_pts = 0
        h_time = 0
        h_path = []
        for i in range(1, 11):
            t = float(row.get(f'O{i}_time', 0))
            p = float(row.get(f'O{i}_points', 0))
            
            if t > 5: # Threshold to consider it an attempt
                h_pts += p
                h_time += t
                h_path.append('A')
            else:
                h_time += BYPASS_COST
                h_path.append('.')
        
        h_valid = "OK" if h_time <= 480 else "DNF"
        
        # --- AI Metrics ---
        _, ai_time, ai_decisions, ai_pts = agent.optimize_path(row)
        ai_valid = "OK" if ai_time <= 480 else "DNF"
        ai_path_str = "".join(ai_decisions)
        
        # --- Comparison ---
        diff = ai_pts - h_pts
        improvements.append(diff)
        
        h_str = f"{int(h_pts)} ({int(h_time)}s) [{''.join(h_path)}]"
        ai_str = f"{int(ai_pts)} ({int(ai_time)}s) [{ai_path_str}]"
        
        print(f"{str(team_id):<8} | {h_str:<35} | {ai_str:<35} | {int(diff):+d}")

    print("-" * 95)
    avg_gain = sum(improvements) / len(improvements)
    print(f"[Conclusion] The XGBoost model improved the average score by {avg_gain:.2f} points per run.")

if __name__ == "__main__":
    # Ensure your CSV is named 'herc_data.csv' and has headers:
    # team_number, O1_time, O1_points, ... O10_time, O10_points
    compare_strategies('herc_data.csv')
