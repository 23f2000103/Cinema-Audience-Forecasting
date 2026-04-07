# Cinema Audience Forecasting

    Kaggle Competition — Time-series forecasting of daily theatre audience counts across multiple locations using dual booking platform data.


# Project Overview

    This project tackles a real-world time-series forecasting problem: predicting the daily audience count for hundreds of cinemas, using data from two distinct sources — an online booking platform (BookNow) and an on-site point-of-sale system (CinePOS). The challenge lies in fusing these two heterogeneous data streams and engineering features that capture temporal patterns, theatre-level behaviour, and calendar effects.
    The solution is built entirely in Python on Kaggle, using gradient boosting and ensemble methods to generate robust predictions.
  

# Methodology

  ## 1. Feature Engineering
    A rich set of temporal and behavioral features was constructed to capture audience patterns:

  ### Calendar Features
  
      Day of week, weekend flag, day, month, day of year, week of year
      Peak week indicator
  
  ### Theater-Level Statistics
  
      Historical average audience (theater_avg) and standard deviation (theater_std) per theater
      Day-of-week × theater average interaction (dow_x_avg) — captures scale-adjusted weekly patterns
      Theater + day-of-week average (theater_dow_avg)
  
  ### Lag Features (leak-safe, using .shift())
  
      lag_7 — audience from exactly 7 days ago (same weekday last week)
      lag_14 — audience from 14 days ago
  
  ### Rolling Window Features (shifted by 1 to prevent data leakage)
  
      Rolling means over windows of 3, 5, 7, 14, and 30 days
  
  ### Theater ID Encoding
  
      Integer encoding of book_theater_id for use as a model feature
  
  ## 2. Train / Validation Split
      A time-based split was used to simulate real forecasting conditions — the last 30 days of training data were held out as a validation set, preventing any look-ahead bias.
    
    Training set: 192,874 samples
    Validation set: 21,172 samples
  
  ## 3. Model Training
      Three models were trained and evaluated on the validation split:
      Model                                   Validation R²
      LightGBM                                  0.6447
      XGBoost                                   0.5953
      Random Forest—Model notes:                0.6398

      LightGBM uses leaf-wise tree growth — excels at capturing complex patterns
      XGBoost uses level-wise growth — more conservative, good regularization
      Random Forest (bagging) — parallel tree construction, strong variance reduction

  ## 4. Ensemble & Post-Processing
      The final prediction is a weighted ensemble:
      final_pred = 0.8 × LightGBM + 0.2 × Random Forest
      Post-processing steps:
      
      Smoothing: 3-day rolling average per theater to reduce prediction jitter
      Clipping: np.maximum(0, ...) to eliminate negative predictions
      Rounding: np.rint(...).astype(int) for integer audience counts

  ## 6. Test Feature Construction
    For the test period, lag and rolling features were reconstructed from the last 60 days of training history per theater. A global mean fallback was applied for theaters with insufficient history.

# Repository Structure
    cinema-audience-forecasting/
    │
    ├── notebook.ipynb          # Main Kaggle notebook (EDA → Features → Modeling → Submission)
    ├── README.md               # Project documentation

    

# How to Run

    Clone this repository
    Open notebook.ipynb in Kaggle or a local Jupyter environment
    Download the competition dataset and update the file paths accordingly
    Run all cells in order

# Dependencies:
    pandas, numpy, lightgbm, xgboost, scikit-learn

# Key Results

    Best validation R²: 0.6447 (LightGBM)
    Final predictions use a smoothed ensemble of LightGBM (80%) and Random Forest (20%)
    Submission : Achieved a kaggle score of 0.44


# Key Learnings

      Fusing two heterogeneous booking data sources requires careful ID mapping and aggregation
      Leak-safe feature construction using .shift() is critical for time-series problems
      Theater-level statistics (avg, std, day-of-week avg) are among the most predictive features
      Prediction smoothing improves real-world plausibility without hurting leaderboard score significantly
      LightGBM's leaf-wise growth consistently outperformed level-wise XGBoost on this dataset
