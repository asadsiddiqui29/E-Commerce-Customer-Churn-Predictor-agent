import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="E-Commerce Customer Churn Predictor",
    page_icon="🛒",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("ecommerce_churn.csv")
model = joblib.load("churn_model.pkl")

# -----------------------------
# HEADER
# -----------------------------
st.title("🛒 E-Commerce Customer Churn Predictor")
st.markdown("### AI Agent for Customer Retention and Business Analytics")

# -----------------------------
# KPI SECTION
# -----------------------------
total_customers = len(df)
churned = df["Churn"].sum()
active = total_customers - churned
churn_rate = (churned / total_customers) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", total_customers)
col2.metric("Active Customers", active)
col3.metric("Churned Customers", churned)
col4.metric("Churn Rate", f"{churn_rate:.2f}%")

st.divider()

# -----------------------------
# DATA ANALYSIS SECTION
# -----------------------------
st.subheader("📊 Dataset Analytics")

left, right = st.columns(2)

with left:

    churn_counts = df["Churn"].value_counts()

    pie_data = pd.DataFrame({
        "Status": ["Active", "Churned"],
        "Count": [
            churn_counts.get(0, 0),
            churn_counts.get(1, 0)
        ]
    })

    fig = px.pie(
        pie_data,
        values="Count",
        names="Status",
        title="Customer Churn Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:

    importance = model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": df.drop("Churn", axis=1).columns,
        "Importance": importance
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=True
    )

    fig2 = px.bar(
        feature_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("📄 Dataset Preview")

st.dataframe(df.head(10), use_container_width=True)

st.divider()

# -----------------------------
# PREDICTION SECTION
# -----------------------------
st.subheader("🤖 Customer Churn Analysis")

c1, c2 = st.columns(2)

with c1:

    login_freq = st.number_input(
        "Login Frequency",
        min_value=0,
        value=5
    )

    purchases = st.number_input(
        "Purchases",
        min_value=0,
        value=5
    )

    amount_spent = st.number_input(
        "Amount Spent ($)",
        min_value=0,
        value=500
    )

with c2:

    tickets = st.number_input(
        "Support Tickets",
        min_value=0,
        value=0
    )

    days_inactive = st.number_input(
        "Days Inactive",
        min_value=0,
        value=30
    )

    membership_months = st.number_input(
        "Membership Duration (Months)",
        min_value=1,
        value=12
    )

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("🚀 Analyze Customer"):

    customer = pd.DataFrame({
        "LoginFreq": [login_freq],
        "Purchases": [purchases],
        "AmountSpent": [amount_spent],
        "Tickets": [tickets],
        "DaysInactive": [days_inactive],
        "MembershipMonths": [membership_months]
    })

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.subheader("Prediction Result")

    # -------------------------
    # RISK LEVEL
    # -------------------------

    probability_percent = probability * 100

    st.subheader("🏷️ Customer Risk Level")

    if probability_percent < 30:

        st.success(
            f"🟢 LOW RISK ({probability_percent:.2f}%)"
        )

    elif probability_percent < 70:

        st.warning(
            f"🟡 MEDIUM RISK ({probability_percent:.2f}%)"
        )

    else:

        st.error(
            f"🔴 HIGH RISK ({probability_percent:.2f}%)"
        )

    st.metric(
        "Churn Probability",
        f"{probability_percent:.2f}%"
    )

    st.progress(float(probability))

    # -------------------------
    # REASONS
    # -------------------------

    reasons = []

    if login_freq < 5:
        reasons.append("Low Login Frequency")

    if purchases < 5:
        reasons.append("Low Purchase Activity")

    if amount_spent < 500:
        reasons.append("Low Spending")

    if tickets > 5:
        reasons.append("High Number of Support Tickets")

    if days_inactive > 90:
        reasons.append("Customer Inactive for Long Time")

    st.subheader("📌 Risk Factors")

    if reasons:

        for reason in reasons:
            st.write("•", reason)

    else:

        st.write("No major risk factors detected.")

    # -------------------------
    # RECOMMENDATIONS
    # -------------------------

    st.subheader("💡 AI Recommendations")

    if prediction == 1:

        st.warning("""
• Send personalized discount coupon

• Offer loyalty rewards

• Launch re-engagement campaign

• Contact customer proactively

• Recommend popular products
        """)

    else:

        st.success("""
• Continue loyalty program

• Maintain customer engagement

• Recommend premium products

• Offer referral rewards
        """)

        st.divider()
