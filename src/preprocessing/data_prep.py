def load_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    return data

def feature_engineering(data):
    # Create new features from existing data
    data['income_to_debt_ratio'] = data['income'] / data['debt']
    return data