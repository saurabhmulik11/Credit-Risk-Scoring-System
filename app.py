import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="CreditWise AI", layout="wide")

model = joblib.load("models/credit_risk_model.pkl")
threshold = joblib.load("models/best_threshold.pkl")


def calculate_risk_score(default_probability):
    score = int(850 - default_probability * 550)
    return max(300, min(score, 850))


def get_risk_category(score):
    if score >= 750:
        return "Low Risk"
    elif score >= 600:
        return "Medium Risk"
    else:
        return "High Risk"


def generate_reason_codes(row):
    reasons = []

    if row["loan_percent_income"] > 0.30:
        reasons.append("High loan burden compared to income")

    if row["loan_int_rate"] > 15:
        reasons.append("High interest rate")

    if row["cb_person_default_on_file"] == "Y":
        reasons.append("Historical default found")

    if row["person_income"] < 30000:
        reasons.append("Low income band")

    if row["loan_grade"] in ["D", "E", "F", "G"]:
        reasons.append("Low loan grade")

    if len(reasons) == 0:
        reasons.append("Stable borrower profile")

    return reasons

def recommend_approval_amount(loan_amount, default_probability, income):
    if default_probability < 0.15:
        approved_amount = loan_amount

    elif default_probability < 0.35:
        approved_amount = min(loan_amount * 0.70, income * 0.25)

    elif default_probability < 0.60:
        approved_amount = min(loan_amount * 0.40, income * 0.15)

    else:
        approved_amount = 0

    return int(approved_amount)

def show_shap_explanation(model_pipeline, input_df):
    try:
        preprocessor = model_pipeline.named_steps["preprocessor"]
        trained_model = model_pipeline.named_steps["model"]

        X_transformed = preprocessor.transform(input_df)

        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        feature_names = preprocessor.get_feature_names_out()

        X_transformed_df = pd.DataFrame(
            X_transformed,
            columns=feature_names
        )

        explainer = shap.TreeExplainer(trained_model)
        shap_values = explainer.shap_values(X_transformed_df)

        st.subheader("SHAP Explainability")
        st.write("This shows which features increased or decreased the default risk.")

        fig, ax = plt.subplots(figsize=(10, 5))

        if isinstance(shap_values, list):
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[1][0],
                    base_values=explainer.expected_value[1],
                    data=X_transformed_df.iloc[0],
                    feature_names=feature_names
                ),
                show=False
            )
        else:
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=X_transformed_df.iloc[0],
                    feature_names=feature_names
                ),
                show=False
            )

        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP explanation is currently unavailable for this model.")
        st.write(e)




st.title("CreditWise AI — Credit Risk Scoring System")
st.write("Predict borrower default risk and generate an explainable credit score.")

st.sidebar.header("Applicant Details")

person_age = st.sidebar.number_input("Age", 18, 100, 25)
person_income = st.sidebar.number_input("Annual Income", 1000, 10000000, 50000)
person_home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_emp_length = st.sidebar.number_input("Employment Length", 0.0, 50.0, 3.0)

loan_intent = st.sidebar.selectbox(
    "Loan Intent",
    ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
)

loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.sidebar.number_input("Loan Amount", 500, 100000, 10000)
loan_int_rate = st.sidebar.number_input("Interest Rate", 0.0, 40.0, 12.0)
cb_person_default_on_file = st.sidebar.selectbox("Historical Default?", ["N", "Y"])
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length", 0, 50, 5)

loan_percent_income = loan_amnt / person_income

input_df = pd.DataFrame({
    "person_age": [person_age],
    "person_income": [person_income],
    "person_home_ownership": [person_home_ownership],
    "person_emp_length": [person_emp_length],
    "loan_intent": [loan_intent],
    "loan_grade": [loan_grade],
    "loan_amnt": [loan_amnt],
    "loan_int_rate": [loan_int_rate],
    "loan_percent_income": [loan_percent_income],
    "cb_person_default_on_file": [cb_person_default_on_file],
    # "cb_preson_cred_hist_length": [cb_person_cred_hist_length],
    "cb_person_cred_hist_length": [cb_person_cred_hist_length],
    "loan_to_income_ratio": [loan_amnt / person_income],
    "income_band": [pd.cut(
        [person_income],
        bins=[0, 30000, 70000, 150000, float("inf")],
        labels=["Low", "Medium", "High", "Very High"]
    )[0]],
    "age_group": [pd.cut(
        [person_age],
        bins=[18, 25, 35, 50, 100],
        labels=["Young", "Adult", "Middle Age", "Senior"]
    )[0]],
    "high_interest_flag": [1 if loan_int_rate > 15 else 0],
    "high_loan_burden": [1 if loan_percent_income > 0.30 else 0]
})

if st.button("Predict Credit Risk"):
    default_probability = model.predict_proba(input_df)[0][1]
    prediction = 1 if default_probability >= threshold else 0

    risk_score = calculate_risk_score(default_probability)
    risk_category = get_risk_category(risk_score)

    approved_amount = recommend_approval_amount(
        loan_amnt,
        default_probability,
        person_income
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Default Probability", f"{default_probability:.2%}")
    col2.metric("Credit Risk Score", risk_score)
    col3.metric("Risk Category", risk_category)
    col4.metric("Recommended Approval Amount", f"₹{approved_amount:,}")

    st.subheader("Decision Recommendation")

    if risk_category == "Low Risk":
        st.success("Recommendation: Approve Loan")
    elif risk_category == "Medium Risk":
        st.warning("Recommendation: Manual Review Required / Approve Reduced Amount")
    else:
        st.error("Recommendation: Reject / Require Strong Collateral")

    st.subheader("Reason Codes")

    reasons = generate_reason_codes(input_df.iloc[0])

    for reason in reasons:
        st.write("•", reason)

 
    # show_shap_explanation(model, input_df)

    st.subheader("Model Decision")

    if prediction == 1:
        st.error("Model Prediction: Likely Default")
    else:
        st.success("Model Prediction: Likely Non-Default")

    