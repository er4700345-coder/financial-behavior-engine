# app.py - all-in-one version
from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

# ----------------- Modules Inside One File -----------------

# Data Analyzer
class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def preprocess(self):
        self.df['income'] = pd.to_numeric(self.df['income'], errors='coerce').fillna(0)
        self.df['expenses'] = pd.to_numeric(self.df['expenses'], errors='coerce').fillna(0)
        self.df['debt'] = pd.to_numeric(self.df['debt'], errors='coerce').fillna(0)
        self.df['loan_requested'] = pd.to_numeric(self.df['loan_requested'], errors='coerce').fillna(0)
        return self.df

# Risk Model
class RiskModel:
    def __init__(self, df):
        self.df = df

    def calculate_scores(self):
        results = []
        for _, row in self.df.iterrows():
            income, expenses, debt, loan_req = row['income'], row['expenses'], row['debt'], row['loan_requested']
            stress = max(0, (expenses + debt + loan_req - income)/income) if income else 1
            loan_risk = min(max(int(stress * 100), 0), 100)
            trust_score = 100 - loan_risk
            results.append({
                'customer_id': row['customer_id'],
                'loan_risk': loan_risk,
                'trust_score': trust_score,
                'financial_stress': stress
            })
        return results

# Liquidity Monitor
class LiquidityMonitor:
    def __init__(self, results):
        self.results = results

    def detect_shock_clusters(self, threshold=70):
        return [r for r in self.results if r['loan_risk'] >= threshold]

# AI Lie Detector
class AILieDetector:
    def __init__(self, df):
        self.df = df

    def detect_anomalies(self):
        self.df['loan_ratio'] = self.df['loan_requested'] / (self.df['income'] + 1)
        features = self.df[['income', 'expenses', 'debt', 'loan_ratio']].values
        model = IsolationForest(contamination=0.05, random_state=42)
        preds = model.fit_predict(features)
        self.df['anomaly_flag'] = np.where(preds == -1, '⚠️ Potential Lie', '✅ Normal')
        return self.df[['customer_id', 'income', 'expenses', 'debt', 'loan_requested', 'anomaly_flag']]

# ----------------- Flask Routes -----------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Financial Behavior Intelligence Engine</title>
<style>
body{background:#0b0c10;color:#c5c6c7;font-family:Roboto,sans-serif;margin:0;padding:0;}
.container{max-width:1200px;margin:50px auto;padding:30px;background:#1f2833;border-radius:10px;box-shadow:0 0 20px rgba(0,0,0,0.7);}
h1{text-align:center;color:#66fcf1;margin-bottom:30px;}
form input,form button{padding:15px 20px;margin:10px;border-radius:6px;border:none;font-size:1rem;}
form input{width:400px;}
form button{background:#45a29e;color:#0b0c10;font-weight:bold;cursor:pointer;transition:.3s;}
form button:hover{background:#66fcf1;}
pre{background:#0b0c10;border-left:5px solid #45a29e;padding:20px;overflow-x:auto;border-radius:5px;}
@media screen and (max-width:768px){form input{width:90%;}pre{font-size:.9rem;}}
</style>
</head>
<body>
<div class="container">
<h1>Financial Behavior Intelligence Engine 💰</h1>
<form method="POST" enctype="multipart/form-data">
<input type="file" name="dataset" accept=".csv" required>
<button type="submit">Analyze</button>
</form>

{% if results %}
<h2>Loan Risk & Trust Scores</h2>
<pre>{{ results }}</pre>

<h2>Liquidity Shock Clusters</h2>
<pre>{{ clusters }}</pre>

<h2>AI Lie Detector / Anomaly Detection</h2>
<pre>{{ anomalies.to_string(index=False) }}</pre>
{% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def index():
    results, clusters, anomalies = [], [], []
    if request.method == "POST":
        file = request.files['dataset']
        df = pd.read_csv(file)
        df = DataAnalyzer(df).preprocess()
        results = RiskModel(df).calculate_scores()
        clusters = LiquidityMonitor(results).detect_shock_clusters()
        anomalies = AILieDetector(df).detect_anomalies()
    return render_template_string(HTML_TEMPLATE, results=results, clusters=clusters, anomalies=anomalies)

if __name__ == "__main__":
    app.run(debug=True)
