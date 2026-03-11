import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

class AILieDetector:
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)

    def preprocess(self):
        # Numeric fields only
        self.df['income'] = pd.to_numeric(self.df['income'], errors='coerce').fillna(0)
        self.df['expenses'] = pd.to_numeric(self.df['expenses'], errors='coerce').fillna(0)
        self.df['debt'] = pd.to_numeric(self.df['debt'], errors='coerce').fillna(0)
        self.df['loan_requested'] = pd.to_numeric(self.df['loan_requested'], errors='coerce').fillna(0)
        return self.df

    def detect_anomalies(self):
        # Features: income, expenses, debt, loan requested ratio
        self.df['loan_ratio'] = self.df['loan_requested'] / (self.df['income'] + 1)
        features = self.df[['income', 'expenses', 'debt', 'loan_ratio']].values

        # Isolation Forest for anomaly detection
        model = IsolationForest(contamination=0.05, random_state=42)
        preds = model.fit_predict(features)

        # -1 = anomaly / potential lie, 1 = normal
        self.df['anomaly_flag'] = np.where(preds == -1, '⚠️ Potential Lie', '✅ Normal')
        return self.df[['customer_id', 'income', 'expenses', 'debt', 'loan_requested', 'anomaly_flag']]
