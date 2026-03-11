from core.ai_lie_detector import AILieDetector

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    clusters = []
    anomalies = []
    if request.method == "POST":
        file = request.files['dataset']

        # Preprocess
        df = DataAnalyzer(file).preprocess()

        # Risk & Trust Scores
        risk = RiskModel(df)
        results = risk.calculate_scores()

        # Liquidity Clusters
        clusters = LiquidityMonitor(results).detect_shock_clusters()

        # AI Lie Detection
        anomalies = AILieDetector(file).preprocess().pipe(
            lambda d: AILieDetector(file).detect_anomalies()
        )

    return render_template("index.html", results=results, clusters=clusters, anomalies=anomalies)
