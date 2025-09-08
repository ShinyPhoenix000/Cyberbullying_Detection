import pandas as pd

def load_csv(file_path):
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def load_stream(api_type='twitter', **kwargs):
    """Placeholder for real-time API streaming (to be implemented)."""
    pass
