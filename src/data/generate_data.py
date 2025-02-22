import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import multiprocessing as mp
from joblib import Parallel, delayed
import psutil
import dask.dataframe as dd
import warnings
warnings.filterwarnings('ignore')

# Global variables
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
TEMP_DIR = os.path.join(DATA_DIR, 'temp')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Add these constants at the top
BATCH_SIZE = 10000  # Number of rows to process at once in generate_approval
MAX_MEMORY_PERCENT = 75  # Maximum memory usage percentage

# Modify get_optimal_chunk_size
def get_optimal_chunk_size():
    available_memory = psutil.virtual_memory().available
    total_memory = psutil.virtual_memory().total
    max_memory = total_memory * (MAX_MEMORY_PERCENT / 100)
    estimated_row_size = 100  # bytes per row (approximate)
    return min(1_000_000, int(min(available_memory, max_memory) * 0.3 / estimated_row_size))

# Initialize parameters
N_SAMPLES = 500_000_000
CHUNK_SIZE = get_optimal_chunk_size()
N_CHUNKS = N_SAMPLES // CHUNK_SIZE
N_CORES = max(1, mp.cpu_count() - 1)  # Leave one core free

# Set random seed for reproducibility
np.random.seed(42)

def generate_batch(size):
    # Generate more realistic age distribution (bell curve centered around 35)
    age = np.random.normal(35, 10, size).astype(int)
    age = np.clip(age, 21, 75)
    
    # Income distribution with long tail (log-normal)
    annual_income = np.random.lognormal(mean=13, sigma=1, size=size)  # Centers around 600k INR
    annual_income = np.clip(annual_income, 180000, 10000000)  # Min 180k, Max 1Cr
    
    # More realistic employment years (correlated with age)
    max_work_years = age - 21  # Assuming work starts at 21
    years_employed = np.array([np.random.uniform(0, max_yr) for max_yr in max_work_years])
    
    # Credit score with realistic distribution (slightly skewed normal)
    credit_score = np.random.normal(650, 100, size).astype(int)
    credit_score = np.clip(credit_score, 300, 900)
    
    return {
        'age': age,
        'annual_income': annual_income,
        'years_employed': years_employed,
        'credit_score': credit_score,
        'num_credit_cards': np.random.binomial(4, 0.3, size),  # More realistic distribution
        'debt_ratio': np.random.beta(2, 5, size),  # Beta distribution for debt ratio
        'education_level': np.random.choice(
            ['SSC', 'HSC', 'Graduate', 'Post Graduate'], 
            size=size,
            p=[0.15, 0.25, 0.40, 0.20]  # Realistic education distribution
        ),
        'employment_type': np.random.choice(
            ['Salaried', 'Self Employed', 'Business', 'Government'], 
            size=size,
            p=[0.45, 0.25, 0.20, 0.10]  # Realistic employment distribution
        ),
        'marital_status': np.random.choice(
            ['Single', 'Married', 'Divorced'], 
            size=size,
            p=[0.35, 0.55, 0.10]  # Realistic marital status distribution
        ),
        'property_ownership': np.random.choice(
            ['Own', 'Rent', 'Living with Parents'], 
            size=size,
            p=[0.30, 0.45, 0.25]  # Realistic property ownership distribution
        ),
        'credit_history_length': np.random.poisson(5, size),  # Years of credit history
        'payment_behavior_score': np.random.normal(75, 15, size).clip(0, 100),  # Payment behavior score
        'existing_loans': np.random.binomial(3, 0.3, size),  # Number of existing loans
        'monthly_expenses': np.random.lognormal(10, 0.5, size)  # Monthly expenses
    }

def generate_approval(row):
    score = 0
    
    # Primary factors (50 points)
    score += (row['credit_score'] > 650) * 15
    score += (row['annual_income'] > 500000) * 15
    score += (row['debt_ratio'] < 0.4) * 10
    score += (row['payment_behavior_score'] > 70) * 10
    
    # Secondary factors (30 points)
    score += (row['years_employed'] > 2) * 8
    score += (row['age'] > 25) * 7
    score += (row['credit_history_length'] > 3) * 8
    score += (row['existing_loans'] < 2) * 7
    
    # Tertiary factors (20 points)
    score += (row['education_level'] in ['Graduate', 'Post Graduate']) * 5
    score += (row['employment_type'] in ['Salaried', 'Government']) * 5
    score += (row['property_ownership'] == 'Own') * 5
    score += (row['marital_status'] == 'Married') * 5
    
    # Add controlled randomness
    score += np.random.normal(0, 5)
    
    # Risk flags
    high_risk = any([
        row['debt_ratio'] > 0.7,
        row['credit_score'] < 500,
        row['existing_loans'] > 2,
        row['payment_behavior_score'] < 50
    ])
    
    return 1 if (score > 60 and not high_risk) else 0

# Generate data in parallel
def process_chunk(chunk_id):
    np.random.seed(42 + chunk_id)  # Ensure reproducibility with different seeds
    
    chunk_data = generate_batch(CHUNK_SIZE)
    df = pd.DataFrame(chunk_data)
    df['approved'] = df.apply(generate_approval, axis=1)
    
    # Save chunk immediately to disk using TEMP_DIR
    chunk_path = os.path.join(TEMP_DIR, f'temp_chunk_{chunk_id}.parquet')
    df.to_parquet(chunk_path, engine='fastparquet', compression='snappy')
    
    return chunk_path

def main():
    print(f"Starting data generation using {N_CORES} cores...")
    print(f"Chunk size: {CHUNK_SIZE:,} rows")
    print(f"Total chunks: {N_CHUNKS:,}")
    
    # Generate chunks in parallel
    chunk_paths = Parallel(n_jobs=N_CORES, verbose=10)(
        delayed(process_chunk)(i) for i in range(N_CHUNKS)
    )
    
    # Use Dask to combine chunks and split into train/test
    print("\nCombining chunks and splitting data...")
    ddf = dd.read_parquet(chunk_paths)
    
    # Shuffle and split data
    train_ddf, test_ddf = ddf.random_split([0.8, 0.2], random_state=42)
    
    # Save final datasets using Dask
    print("Saving final datasets...")
    train_ddf.to_parquet(
        os.path.join(DATA_DIR, 'train.parquet'),
        engine='fastparquet',
        compression='snappy'
    )
    test_ddf.to_parquet(
        os.path.join(DATA_DIR, 'test.parquet'),
        engine='fastparquet',
        compression='snappy'
    )
    
    # Clean up temporary files
    for path in chunk_paths:
        try:
            os.remove(path)
        except OSError as e:
            print(f"Error removing {path}: {e}")
    
    try:
        os.rmdir(TEMP_DIR)
    except OSError as e:
        print(f"Error removing temp directory: {e}")
    
    # Calculate and print statistics
    print("\nData Generation Summary:")
    print(f"Total samples: {N_SAMPLES:,}")
    print(f"Training samples: {int(N_SAMPLES * 0.8):,}")
    print(f"Testing samples: {int(N_SAMPLES * 0.2):,}")
    print(f"Approval rate: {float(ddf.approved.mean().compute()):.2%}")

if __name__ == '__main__':
    # Cloud deployment options
    print("""
    Cloud Deployment Options:
    1. AWS EMR: Use bootstrap action to install requirements
    2. Google Dataproc: Use initialization action
    3. Azure HDInsight: Use script action
    
    Example AWS EMR command:
    aws emr create-cluster 
        --name "Credit Card Data Generation" 
        --instance-type m5.xlarge 
        --instance-count 10 
        --applications Name=Spark 
        --bootstrap-actions Path=s3://bucket/bootstrap.sh
    """)
    
    # Ask user for execution preference
    choice = input("Run locally? (y/n): ")
    if choice.lower() == 'y':
        main()
    else:
        print("Please deploy to cloud using the instructions above.")