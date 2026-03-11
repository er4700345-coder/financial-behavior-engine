from flask import Flask, render_template, request
import pandas as pd
from core.data_analyzer import DataAnalyzer
from core.risk_model import RiskModel
from core.liquidity_monitor import LiquidityMonitor

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    clusters = []
    if request.method == "POST":
        file = request.files['dataset']
        df = DataAnalyzer(file).preprocess()
        risk = RiskModel(df)
        results = risk.calculate_scores()
        clusters = LiquidityMonitor(results).detect_shock_clusters()
    return render_template("index.html", results=results, clusters=clusters)

if __name__ == "__main__":
    app.run(debug=True)
