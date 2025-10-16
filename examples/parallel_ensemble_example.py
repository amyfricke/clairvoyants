"""
Example demonstrating parallel ensemble fitting in Clairvoyants.

This example shows how to use the new parallel parameter to speed up
ensemble model fitting when you have multiple models.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from clairvoyants import Clairvoyant
from clairvoyants import ensemble


def main():
    """Demonstrate parallel ensemble fitting."""
    print("Clairvoyants Parallel Ensemble Fitting Example")
    print("=" * 50)
    
    # Create synthetic time series data
    print("Creating synthetic time series data...")
    n_days = 200
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # Create a more realistic time series with trend and seasonality
    trend = np.linspace(100, 150, n_days)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    noise = np.random.normal(0, 5, n_days)
    actual = trend + seasonal + noise
    
    df = pd.DataFrame({'dt': dates, 'actual': actual})
    print(f"Created time series with {len(df)} observations")
    
    # Define ensemble models
    models = [
        ensemble.sarimax_013_011,
        ensemble.sarimax_003_001,
        ensemble.auto_sarif_nnet,
        ensemble.auto_scarf_nnet
    ]
    
    print(f"Using {len(models)} models in ensemble")
    
    # Configure Clairvoyant
    clair = Clairvoyant(
        models=models,
        periods_agg=[7],  # Weekly aggregation
        periods=[7],       # Weekly seasonality
        periods_trig=[365.25/7],  # Annual seasonality
        transform='none',  # No transformation
        pred_level=0.8    # 80% prediction intervals
    )
    
    # Test parallel fitting
    print("\nFitting ensemble with parallel=True...")
    import time
    start_time = time.time()
    
    clair.fit_ensemble(df, parallel=True)
    
    parallel_time = time.time() - start_time
    print(f"Parallel fitting completed in {parallel_time:.2f} seconds")
    
    # Test sequential fitting for comparison
    print("\nFitting ensemble with parallel=False...")
    start_time = time.time()
    
    clair.fit_ensemble(df, parallel=False)
    
    sequential_time = time.time() - start_time
    print(f"Sequential fitting completed in {sequential_time:.2f} seconds")
    
    # Display results
    print(f"\nPerformance comparison:")
    print(f"Parallel time:   {parallel_time:.2f}s")
    print(f"Sequential time: {sequential_time:.2f}s")
    print(f"Speedup:         {sequential_time/parallel_time:.2f}x")
    
    # Show forecast results
    consensus = clair.forecast['consensus']
    print(f"\nForecast results:")
    print(f"Forecast period: {len(consensus)} days")
    print(f"Forecast columns: {list(consensus.columns)}")
    print(f"Sample forecast values:")
    print(consensus.head())
    
    # Show individual model results
    print(f"\nIndividual model forecasts:")
    for model_name, forecast in clair.forecast['ensemble'].items():
        print(f"  {model_name}: {len(forecast)} forecast points")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
