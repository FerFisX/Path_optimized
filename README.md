# Strategic Route Optimization for NASA HERC 2024 🚀

**A Hybrid Approach using XGBoost Learning-to-Rank and Directed Acyclic
Graphs.**

**Authors:** Adrian Acarapi & Daniel Callata

This repository contains the official implementation of the research
paper:

> *"Strategic Route Optimization in Human Exploration Rover Challenges
> (HERC) via Hybrid XGBoost and Directed Graph Models."*

The system provides a framework to compare historical human
decision-making against two Artificial Intelligence agents:

1.  **Baseline Agent (Linear):** Uses classical statistical regression.\
2.  **Proposed Agent (XGBoost):** Uses Gradient Boosting to detect
    non-linear risk thresholds and optimize strategic decision-making
    under strict time constraints ($T_{max} = 480s$).

------------------------------------------------------------------------

## 📂 Project Structure

  --------------------------------------------------------------------------
  File                      Description
  ------------------------- ------------------------------------------------
  `linear_comparison.py`    **Baseline Model.** Trains a linear regression
                            model on synthetic data and benchmarks it
                            against human teams to demonstrate the
                            limitations of linear assumptions.

  `xgboost_comparison.py`   **Proposed Model.** Trains a Gradient Boosting
                            model to learn risk aversion and executes a Beam
                            Search algorithm on a DAG to find the optimal
                            route.

  `herc_data.csv`           **Input Dataset.** Contains the real-world
                            performance metrics of the competing teams (user
                            provided).

  `requirements.txt`        List of Python dependencies required to run the
                            simulations.
  --------------------------------------------------------------------------

------------------------------------------------------------------------

## 🛠️ Installation & Requirements

### 1. Prerequisites

-   Python 3.8 or higher\
-   A standard C compiler (required for some XGBoost distributions,
    though usually pre-compiled wheels are available)

### 2. Install Dependencies

Navigate to the project directory and run:

``` bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install the core libraries manually:

``` bash
pip install numpy pandas scikit-learn xgboost networkx matplotlib
```

------------------------------------------------------------------------

## 📊 Input Data Specification (`herc_data.csv`)

To perform the comparative analysis, the scripts require a CSV file
named `herc_data.csv` located in the root directory.

The CSV must follow a specific column structure representing the
chronological performance of each team across the 10 obstacles.

### Required Column Schema

  Column Name     Data Type   Description
  --------------- ----------- ---------------------------------------------
  `team_number`   Integer     Unique identifier for the team
  `O1_time`       Float       Time taken to complete Obstacle 1 (seconds)
  `O1_points`     Integer     Points awarded for Obstacle 1
  `O2_time`       Float       Time taken to complete Obstacle 2
  `O2_points`     Integer     Points awarded for Obstacle 2
  ...             ...         ...
  `O10_time`      Float       Time taken to complete Obstacle 10
  `O10_points`    Integer     Points awarded for Obstacle 10

### Important Data Entry Rules

-   **Bypasses:** If a team bypassed an obstacle, set both time and
    points to `0`. The algorithm interprets `0` as a skipped obstacle
    and estimates the potential difficulty based on the simulation
    model.\
-   **Format:** Ensure the file uses commas (`,`) as delimiters.

------------------------------------------------------------------------

## 🚀 Execution Guide

### Experiment A: Baseline Evaluation (Linear Model)

Run this script to observe how a traditional linear cost-benefit
analysis compares to human decisions:

``` bash
python linear_comparison.py
```

**Output:**

-   Generates 5,000 synthetic samples to train the linear regressor\
-   Iterates through `herc_data.csv`\
-   Displays a side-by-side comparison: **Human Strategy vs. Linear
    Agent Strategy**

------------------------------------------------------------------------

### Experiment B: Proposed Solution (XGBoost Model)

Run this script to generate the optimized routes using the non-linear
risk-aware model:

``` bash
python xgboost_comparison.py
```

**Output:**

-   Runs a Monte Carlo simulation (10,000 scenarios) to train the
    XGBoost model\
-   Reconstructs the race context for each team in the CSV\
-   Executes the Beam Search algorithm to find the optimal path\
-   Displays the final comparative report showing the **AI Gain (Score
    Improvement)**

------------------------------------------------------------------------

## 📈 Interpreting the Results

The console output provides a detailed breakdown of the performance gap.
Example format:

    Team   | Human Results (Pts/Time/Path)       | XGBoost Results (Pts/Time/Path)     | Gain
    -------------------------------------------------------------------------------------------
    24     | 180 (478s) [AAAA.A..A.]             | 210 (460s) [AA..AAA.A.]             | +30
    25     | 200 (495s) [AAAAA.....] -> DNF      | 190 (475s) [AAAA......] -> OK       | -10 (Saved)

### Path Notation

-   `A`: Attempt --- the agent decided to traverse the obstacle\
-   `.`: Bypass --- the agent decided to skip the obstacle to save
    time/energy

### Status Labels

-   **OK:** The run was completed within the 480-second limit\
-   **DNF (Did Not Finish):** The time exceeded 480s. In the
    competition, this often results in a score of 0 for the time
    category.

### Gain Interpretation

-   **Positive (+):** The AI found a path that yields more points within
    the time limit\
-   **Saved:** The AI sacrificed raw points to ensure the team finished
    within the time limit (preventing a DNF)

------------------------------------------------------------------------

## ⚠️ Troubleshooting

### Issue: FileNotFoundError: `[Errno 2] No such file or directory: 'herc_data.csv'`

**Solution:** Ensure the CSV file is named exactly `herc_data.csv` and
is in the same folder as the Python scripts.

------------------------------------------------------------------------

### Issue: KeyError: `O1_time`

**Solution:** Check your CSV headers. They must match the naming
convention `Ox_time` and `Ox_points` (where `x` is the obstacle number
1--10).

------------------------------------------------------------------------

### Issue: Low Completion Rate in Linear Model

**Explanation:** This is expected behavior. The linear model fails to
account for non-linear risk accumulation, often leading to overtime
routes. This validates the need for the XGBoost model.

------------------------------------------------------------------------

## 📞 Contact & Citation

**Authors:** Adrian Acarapi & Daniel Callata\
**Context:** NASA HERC 2024 Research Project\
**Date:** February 2026

For academic citations, please refer to the associated paper included in
this repository.
