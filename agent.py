import joblib
import pandas as pd

model = joblib.load("churn_model.pkl")

def analyze_customer(
    login_freq,
    purchases,
    amount_spent,
    tickets,
    days_inactive,
    membership_months
):

    customer = pd.DataFrame({
        "LoginFreq": [login_freq],
        "Purchases": [purchases],
        "AmountSpent": [amount_spent],
        "Tickets": [tickets],
        "DaysInactive": [days_inactive],
        "MembershipMonths": [membership_months]
    })

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0]

    churn_probability = probability[1] * 100

    print("\n========== CUSTOMER REPORT ==========")

    if prediction == 1:
        print("\nPrediction: HIGH CHURN RISK")
    else:
        print("\nPrediction: LOW CHURN RISK")

    print(f"\nChurn Probability: {churn_probability:.2f}%")

    reasons = []

    if login_freq < 5:
        reasons.append("Very low login frequency")

    if purchases < 5:
        reasons.append("Low purchase activity")

    if amount_spent < 500:
        reasons.append("Low spending")

    if days_inactive > 90:
        reasons.append("Customer inactive for a long time")

    if tickets > 5:
        reasons.append("Many customer service issues")

    print("\nReasons:")

    if reasons:
        for reason in reasons:
            print("-", reason)
    else:
        print("- No major risk factors detected")

    print("\nRecommendations:")

    if prediction == 1:
        print("- Send personalized discount coupon")
        print("- Offer loyalty rewards")
        print("- Send re-engagement email")
        print("- Contact customer support proactively")
    else:
        print("- Continue loyalty program")
        print("- Recommend relevant products")
        print("- Maintain engagement campaigns")


analyze_customer(
    login_freq=8,
    purchases=10,
    amount_spent=400,
    tickets=7,
    days_inactive=120,
    membership_months=6
)