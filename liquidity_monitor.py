class LiquidityMonitor:
    def __init__(self, results):
        self.results = results

    def detect_shock_clusters(self, threshold=70):
        clusters = [r for r in self.results if r['loan_risk'] >= threshold]
        return clusters
