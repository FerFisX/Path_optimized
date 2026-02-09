"""
NASA HERC 2024 Route Optimization - Baseline Model
Algorithm: Linear Learning-to-Rank (Pointwise) + Beam Search
Author: Research Team
Description: Implements a linear regression approach to estimate obstacle efficiency
             and optimizes the path using a DAG-based Beam Search.
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Configuration ---
N_TRAIN_SAMPLES = 10000
N_TEST_SAMPLES = 500
TIME_LIMIT = 480  # seconds
BEAM_WIDTH = 20
RANDOM_SEED = 42

class HERCSimulator:
    """Generates synthetic data based on NASA HERC physical constraints."""
    
    def __init__(self, seed=RANDOM_SEED):
        np.random.seed(seed)

    def generate_batch(self, n_samples):
        """
        Generates a batch of obstacle data.
        
        Features:
        - difficulty (1-10): Intrinsic complexity.
        - avg_time: Log-Normal distribution (long tail).
        - risk_prob: Beta distribution (bounded [0,1]).
        - points: Discrete values [10, 20, 30, 40, 50].
        
        Target:
        - efficiency: Synthetic ground truth for training.
        """
        data = {
            'difficulty': np.random.randint(1, 11, n_samples),
            'points': np.random.choice([10, 20, 30, 40, 50], n_samples),
            'risk_prob': np.random.beta(a=2, b=5, size=n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Simulate time with Log-Normal distribution dependent on difficulty
        # Mean scales with difficulty, sigma is fixed noise
        mu = np.log(10 + df['difficulty'] * 5)
        sigma = 0.4
        df['avg_time'] = np.random.lognormal(mean=mu, sigma=sigma)
        
        # Simulate Energy consumption (correlated with time and difficulty)
        df['energy'] = df['avg_time'] * 0.5 + df['difficulty'] * 2 + np.random.normal(0, 2, n_samples)
        
        # Ground Truth Calculation (Target Variable)
        # Non-linear penalty: if risk is high, efficiency drops drastically
        penalty = df['risk_prob'] ** 3
        df['true_efficiency'] = (df['points'] / df['avg_time']) * (1 - penalty)
        
        return df

class LinearRouteOptimizer:
    """Baseline solver using OLS Linear Regression."""
    
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, X, y):
        """Trains the linear model."""
        print("[INFO] Training Linear Regression Model...")
        self.model.fit(X, y)
        
        # Log coefficients
        coeffs = pd.Series(self.model.coef_, index=X.columns)
        print("\n[RESULTS] Learned Linear Coefficients:")
        print(coeffs)
        print(f"Intercept: {self.model.intercept_:.4f}\n")

    def predict_efficiency(self, obstacles_df):
        """Predicts strategic value (S_i) for a set of obstacles."""
        features = obstacles_df[['difficulty', 'avg_time', 'risk_prob', 'energy', 'points']]
        return self.model.predict(features)

    def solve_path_beam_search(self, race_obstacles):
        """
        Executes Beam Search on the DAG of decisions.
        State: (current_score, current_time, path_history)
        """
        # Predictions
        pred_eff = self.predict_efficiency(race_obstacles)
        
        # Beam Initialization
        # Format: (score, time, path_list)
        beam = [(0.0, 0.0, [])]
        
        for i in range(len(race_obstacles)):
            obs_data = race_obstacles.iloc[i]
            predicted_score = pred_eff[i]
            
            candidates = []
            
            for score, time, path in beam:
                # Option A: ATTEMPT
                # Linear model might underestimate time, we use the estimated avg_time for planning
                new_time_att = time + obs_data['avg_time']
                
                if new_time_att <= TIME_LIMIT:
                    # Heuristic for sorting: accumulate predicted efficiency
                    new_score_att = score + predicted_score 
                    candidates.append((new_score_att, new_time_att, path + ['Attempt']))
                
                # Option B: BYPASS
                new_time_byp = time + 15.0 # Fixed bypass cost
                if new_time_byp <= TIME_LIMIT:
                    # Bypass adds 0 efficiency
                    candidates.append((score, new_time_byp, path + ['Bypass']))
            
            # Pruning: Keep top K paths based on accumulated predicted efficiency
            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[:BEAM_WIDTH]
            
        return beam[0] # Return best path

# --- Execution Pipeline ---

if __name__ == "__main__":
    sim = HERCSimulator()
    
    # 1. Data Generation
    train_df = sim.generate_batch(N_TRAIN_SAMPLES)
    X = train_df[['difficulty', 'avg_time', 'risk_prob', 'energy', 'points']]
    y = train_df['true_efficiency']
    
    # 2. Training
    optimizer = LinearRouteOptimizer()
    optimizer.train(X, y)
    
    # 3. Evaluation on Test Set
    print(f"[INFO] Running Evaluation on {N_TEST_SAMPLES} unseen races...")
    results = []
    
    for _ in range(N_TEST_SAMPLES):
        # Generate a single race (10 obstacles)
        race_df = sim.generate_batch(10) 
        
        # Solve
        best_solution = optimizer.solve_path_beam_search(race_df)
        decisions = best_solution[2]
        
        # Calculate REAL outcome (Ground Truth)
        real_points = 0
        real_time = 0
        
        for i, action in enumerate(decisions):
            if action == 'Attempt':
                # Check for failure based on risk probability
                if np.random.rand() > race_df.iloc[i]['risk_prob']:
                    real_points += race_df.iloc[i]['points']
                    real_time += race_df.iloc[i]['avg_time']
                else:
                    # Failure case: Time spent, 0 points
                    real_time += race_df.iloc[i]['avg_time'] * 1.5 # Recovery penalty
            else:
                real_time += 15.0
        
        # Check constraints
        success = 1 if real_time <= TIME_LIMIT else 0
        if not success: real_points = 0 # Disqualified
        
        results.append({'points': real_points, 'time': real_time, 'success': success})

    # 4. Metrics Reporting
    res_df = pd.DataFrame(results)
    print("-" * 40)
    print("BASELINE MODEL PERFORMANCE METRICS")
    print("-" * 40)
    print(f"Average Points:      {res_df['points'].mean():.2f}")
    print(f"Average Time:        {res_df['time'].mean():.2f} s")
    print(f"Completion Rate:     {res_df['success'].mean() * 100:.1f}%")
    print("-" * 40)
