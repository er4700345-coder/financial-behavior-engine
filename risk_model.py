class RiskModel:
    def __init__(self, df):
        self.df = df

    def calculate_scores(self):
        results = []
        for _, row in self.df.iterrows():
            income = row['income']
            expenses = row['expenses']
            debt = row['debt']
            loan_req = row['loan_requested']

            # Financial stress factor
            stress = max(0, (expenses + debt + loan_req - income)/income) if income else 1

            # Loan risk score (0 low - 100 high)
            loan_risk = min(max(int(stress * 100), 0), 100)

            # Trust score (inverse of risk)
            trust_score = 100 - loan_risk

            results.append({
                'customer_id': row['customer_id'],
                'loan_risk': loan_risk,
                'trust_score': trust_score,
                'financial_stress': stress
            })
        return results
