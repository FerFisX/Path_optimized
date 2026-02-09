"""
NASA HERC 2024 Route Optimization - Proposed Model
Algorithm: XGBoost Gradient Boosting + DAG Beam Search
Author: Research Team
Description: Implements a non-linear L2R approach to learn complex risk/reward 
             functions and optimizes the trajectory.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- Configuration ---
N_TRAIN_SAMPLES = 10000
N_TEST_SAMPLES = 500
TIME_LIMIT = 480
BEAM_WIDTH = 20
RANDOM_SEED = 42

class HERCSimulator:
    """
    Generates synthetic data with complex non-linear interactions.
    Replicates the physics of the HERC environment.
    """
    def __init__(self, seed=RANDOM_SEED):
        np.random.seed(seed)

    def generate_batch(self, n_samples):
        # Same generation logic to ensure fair comparison
        data = {
            'difficulty': np.random.randint(1, 11, n_samples),
            'points': np.random.choice([10, 20, 30, 40, 50], n_samples),
            'risk_prob': np.random.beta(a=2, b=5, size=n_samples)
        }
        df = pd.DataFrame(data)
        mu = np.log(10 + df['difficulty'] * 5)
        df['avg_time'] = np.random.lognormal(mean=mu, sigma=0.4)
        df['energy'] = df['avg_time'] * 0.5 + df['difficulty'] * 2 + np.random.normal(0, 2, n_samples)
        
        # Target: Efficiency = (Points / Time) * (1 - Risk^3)
        # The cubic penalty is the non-linearity XGBoost must learn
        penalty = df['risk_prob'] ** 3
        df['true_efficiency'] = (df['points'] / df['avg_time']) * (1 - penalty)
        
        return df

class XGBoostRouteOptimizer:
    """Proposed solver using Gradient Boosted Trees."""
    
    def __init__(self):
        # Hyperparameters derived from Grid Search (as per paper Table II)
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        
    def train(self, X, y):
        """Trains the XGBoost model."""
        print("[INFO] Training XGBoost Model...")
        self.model.fit(X, y)
        print("[INFO] Training Complete.")
        
    def plot_importance(self):
        """Generates Feature Importance plot for the paper."""
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(self.model, importance_type='gain', max_num_features=10)
        plt.title('XGBoost Feature Importance (Information Gain)')
        plt.tight_layout()
        plt.savefig('xgboost_feature_importance.png', dpi=300)
        print("[INFO] Feature importance plot saved as 'xgboost_feature_importance.png'")

    def solve_path_beam_search(self, race_obstacles):
        """
        Executes Beam Search using XGBoost predictions as heuristic weights.
        """
        features = race_obstacles[['difficulty', 'avg_time', 'risk_prob', 'energy', 'points']]
        # Predict latent efficiency score
        pred_scores = self.model.predict(features)
        
        beam = [(0.0, 0.0, [])] # (heuristic_score, time, path)
        
        for i in range(len(race_obstacles)):
            obs_data = race_obstacles.iloc[i]
            predicted_efficiency = pred_scores[i]
            
            candidates = []
            for score, time, path in beam:
                # 1. ATTEMPT Node
                new_time = time + obs_data['avg_time']
                if new_time <= TIME_LIMIT:
                    # Accumulate the predicted efficiency (L2R score)
                    new_score = score + predicted_efficiency
                    candidates.append((new_score, new_time, path + ['Attempt']))
                
                # 2. BYPASS Node
                new_time_byp = time + 15.0
                if new_time_byp <= TIME_LIMIT:
                    candidates.append((score, new_time_byp, path + ['Bypass']))
            
            # Sort by highest predicted efficiency
            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[:BEAM_WIDTH]
            
        return beam[0]

# --- Execution Pipeline ---

if __name__ == "__main__":
    sim = HERCSimulator()
    
    # 1. Data Generation
    train_df = sim.generate_batch(N_TRAIN_SAMPLES)
    X = train_df[['difficulty', 'avg_time', 'risk_prob', 'energy', 'points']]
    y = train_df['true_efficiency']
    
    # 2. Training
    optimizer = XGBoostRouteOptimizer()
    optimizer.train(X, y)
    optimizer.plot_importance()
    
    # 3. Simulation / Testing
    print(f"[INFO] Running Evaluation on {N_TEST_SAMPLES} unseen races...")
    results = []
    
    for _ in range(N_TEST_SAMPLES):
        race_df = sim.generate_batch(10) # 10 obstacles per race
        
        # Get strategic plan
        best_solution = optimizer.solve_path_beam_search(race_df)
        decisions = best_solution[2]
        
        # Simulate Execution (Real World)
        real_points = 0
        real_time = 0
        
        for i, action in enumerate(decisions):
            if action == 'Attempt':
                # Probabilistic Failure Check
                if np.random.rand() > race_df.iloc[i]['risk_prob']:
                    real_points += race_df.iloc[i]['points']
                    real_time += race_df.iloc[i]['avg_time']
                else:
                    # Failure penalty
                    real_time += race_df.iloc[i]['avg_time'] * 1.5 
            else:
                real_time += 15.0
        
        # Constraint Check
        success = 1 if real_time <= TIME_LIMIT else 0
        if not success: real_points = 0 # DNF
        
        results.append({'points': real_points, 'time': real_time, 'success': success})

    # 4. Metrics Reporting
    res_df = pd.DataFrame(results)
    print("-" * 40)
    print("PROPOSED MODEL (XGBOOST) PERFORMANCE")
    print("-" * 40)
    print(f"Average Points:      {res_df['points'].mean():.2f}")
    print(f"Average Time:        {res_df['time'].mean():.2f} s")
    print(f"Completion Rate:     {res_df['success'].mean() * 100:.1f}%")
    print("-" * 40)
