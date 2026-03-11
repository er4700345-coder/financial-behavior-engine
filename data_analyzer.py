import pandas as pd

class DataAnalyzer:
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)

    def preprocess(self):
        # Ensure numeric fields
        self.df['income'] = pd.to_numeric(self.df['income'], errors='coerce').fillna(0)
        self.df['expenses'] = pd.to_numeric(self.df['expenses'], errors='coerce').fillna(0)
        self.df['debt'] = pd.to_numeric(self.df['debt'], errors='coerce').fillna(0)
        self.df['loan_requested'] = pd.to_numeric(self.df['loan_requested'], errors='coerce').fillna(0)
        return self.df
