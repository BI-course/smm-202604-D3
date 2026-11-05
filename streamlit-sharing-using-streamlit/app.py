import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# -----------------------------------------------------------------------
# Load all four persisted artifacts ONCE, at module import time. Streamlit
# re-runs this whole script top-to-bottom on every widget interaction, so
# @st.cache_resource keeps joblib.load() from re-reading the files from
# disk on every single click -- without it, every tab switch and every
# form submission would reload all four artifacts again.
# -----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # Dynamically locate the model folder relative to app.py
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

    # 1. Regression (Farm Income)
    regression_pipeline = joblib.load(
        os.path.join(MODEL_DIR, "farm_income_linear_regression_pipeline.joblib")
    )

    # 2. Classification (Invoice Payment Risk)
    try:
        classification_pipeline = joblib.load(
            os.path.join(MODEL_DIR, "invoice_payment_risk_final_pipeline.joblib")
        )
        payment_risk_label_encoder = joblib.load(
            os.path.join(MODEL_DIR, "payment_risk_label_encoder.joblib")
        )
    except Exception:
        classification_pipeline = None
        payment_risk_label_encoder = None

    # 3. Clustering (Farmer Segmentation)
    try:
        clustering_bundle = joblib.load(
            os.path.join(MODEL_DIR, "farmer_segmentation_bundle.joblib")
        )
    except Exception:
        clustering_bundle = None

    # 4. Association Rule Mining (Food Delivery)
    try:
        arm_artifact = joblib.load(
            os.path.join(MODEL_DIR, "food_delivery_orders_apriori_recommendation_rules.joblib")
        )
    except Exception:
        arm_artifact = None

    return (regression_pipeline, classification_pipeline, payment_risk_label_encoder,
            clustering_bundle, arm_artifact)
(regression_pipeline, classification_pipeline, payment_risk_label_encoder,
 clustering_bundle, arm_artifact) = load_artifacts()
clustering_pipeline = clustering_bundle["pipeline"]
clustering_cluster_names = clustering_bundle["metadata"]["cluster_names"]
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]

st.set_page_config(page_title="BBT 4206 Model Validation Demos", page_icon="\U0001F4CA", layout="centered")
st.title("BBT 4206 -- Model Validation Demos")

tab_regression, tab_classification, tab_clustering, tab_arm = st.tabs([
    "Farm Annual Income Predictor", "Invoice Payment Risk Classifier",
    "Farmer Segmenter", "Food Delivery Order Recommender",
])

# -----------------------------------------------------------------------
# 1. Regression
# -----------------------------------------------------------------------
with tab_regression:
    with st.form("regression_form"):
        soil_quality_score = st.number_input("Soil Quality Score", value=7.8)
        farm_size_acres = st.number_input("Farm Size (Acres)", value=5.5)
        years_farming_experience = st.number_input("Years Farming Experience", value=14.0)
        num_workers_employed = st.number_input("Number of Workers Employed", value=4)
        rainfall_mm = st.number_input("Rainfall (mm)", value=1100.0)
        market_distance_km = st.number_input("Market Distance (km)", value=12.5)
        machinery_value_kes = st.number_input("Machinery Value (KES)", value=450000.0)
        fertilizer_spend_kes = st.number_input("Fertilizer Spend (KES)", value=65000.0)
        pesticide_spend_kes = st.number_input("Pesticide Spend (KES)", value=35000.0)
        irrigation_maintenance_cost_kes = st.number_input("Irrigation Maintenance Cost (KES)", value=25000.0)
        crop_type = st.selectbox("Crop Type", ["Horticulture", "Tea", "Maize", "Coffee", "Wheat"])
        county = st.text_input("County", "Nakuru")
        irrigation_type = st.selectbox("Irrigation Type", ["Drip", "Manual", "Rain-fed", "Sprinkler"])
        cooperative_member = st.selectbox("Cooperative Member", ["Yes", "No"])
        submitted = st.form_submit_button("Predict Annual Income")

    if submitted:
        payload = {
            "soil_quality_score": soil_quality_score,
            "farm_size_acres": farm_size_acres,
            "years_farming_experience": years_farming_experience,
            "num_workers_employed": num_workers_employed,
            "rainfall_mm": rainfall_mm,
            "market_distance_km": market_distance_km,
            "machinery_value_kes": machinery_value_kes,
            "fertilizer_spend_kes": fertilizer_spend_kes,
            "pesticide_spend_kes": pesticide_spend_kes,
            "irrigation_maintenance_cost_kes": irrigation_maintenance_cost_kes,
            "crop_type": crop_type,
            "county": county,
            "irrigation_type": irrigation_type,
            "cooperative_member": cooperative_member,
        }
        input_row = pd.DataFrame([payload])
        # Your pipeline handles preprocessing & returns prediction directly in KES
        predicted_revenue_kes = float(regression_pipeline.predict(input_row)[0])
        st.success(f"Predicted annual farm income: KES {predicted_revenue_kes:,.2f}")

# -----------------------------------------------------------------------
# 2. Classification
# -----------------------------------------------------------------------
with tab_classification:
    with st.form("classification_form"):
        invoice_amount_kes = st.number_input("Invoice Amount (KES)", value=120000.0, key="iak2")
        guarantee_value_kes = st.number_input("Guarantee Value (KES)", value=50000.0, key="gvk2")
        days_outstanding_avg_historical = st.number_input(
            "Avg. Days Outstanding (Historical)", value=15.0, key="doah2")
        client_credit_rating = st.selectbox(
            "Client Credit Rating", ["Poor", "Fair", "Good", "Excellent"], index=2, key="ccr2")
        has_dispute_history = st.selectbox(
            "Has Dispute History", ["No", "Yes"], index=0, key="hdh2")
        submitted2 = st.form_submit_button("Predict Payment Risk")

    if submitted2:
        payload = {
            "invoice_amount_kes": invoice_amount_kes,
            "guarantee_value_kes": guarantee_value_kes,
            "days_outstanding_avg_historical": days_outstanding_avg_historical,
            "client_credit_rating": client_credit_rating,
            "has_dispute_history": has_dispute_history,
        }
        input_row = pd.DataFrame([payload])
        # The model was trained on a label-encoded target (0/1/2), so decode
        # the prediction back to its readable class name via the saved encoder.
        predicted_class_encoded = classification_pipeline.predict(input_row)[0]
        predicted_class = payment_risk_label_encoder.inverse_transform(
            [predicted_class_encoded])[0]
        predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
        class_labels = payment_risk_label_encoder.inverse_transform(
            classification_pipeline.named_steps["model"].classes_)
        st.success(f"Predicted invoice payment risk: {predicted_class}")
        st.write({str(label): round(float(prob), 4)
                  for label, prob in zip(class_labels, predicted_probabilities)})

# -----------------------------------------------------------------------
# 3. Clustering
# -----------------------------------------------------------------------
with tab_clustering:
    with st.form("clustering_form"):
        farm_size_acres3 = st.number_input("Farm Size (Acres)", value=3.6, key="fsa3")
        annual_yield_tonnes3 = st.number_input("Annual Yield (Tonnes)", value=9.8, key="ayt3")
        income_per_acre_kes3 = st.number_input("Income per Acre (KES)", value=39000.0, key="ipak3")
        years_farming_experience3 = st.number_input(
            "Years Farming Experience", value=9.0, key="yfe3")
        market_access_score3 = st.number_input(
            "Market Access Score (0-100)", min_value=0.0, max_value=100.0, value=53.0, key="mas3")
        submitted3 = st.form_submit_button("Predict Segment")

    if submitted3:
        input_row = pd.DataFrame([{
            "farm_size_acres": farm_size_acres3,
            "annual_yield_tonnes": annual_yield_tonnes3,
            "income_per_acre_kes": income_per_acre_kes3,
            "years_farming_experience": years_farming_experience3,
            "market_access_score": market_access_score3,
        }])
        cluster = int(clustering_pipeline.predict(input_row)[0])
        cluster_name = clustering_cluster_names.get(cluster, f"Cluster {cluster}")
        st.success(f"Assigned segment: {cluster_name}")

# -----------------------------------------------------------------------
# 4. Association Rule Mining
# -----------------------------------------------------------------------
with tab_arm:
    with st.form("arm_form"):
        cart_text = st.text_input("Cart (comma-separated SKUs)", "F10, F11")
        top_n = st.number_input("Top N Recommendations", min_value=1, value=3)
        submitted4 = st.form_submit_button("Get Recommendations")

    if submitted4:
        cart = {sku.strip() for sku in cart_text.split(",") if sku.strip()}
        candidates = {}
        for antecedent, consequent_list in recommendation_lookup.items():
            if antecedent.issubset(cart):
                for sku, confidence, lift in consequent_list:
                    if sku not in cart:
                        if sku not in candidates or confidence > candidates[sku][0]:
                            candidates[sku] = (confidence, lift)

        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:int(top_n)]
        if not ranked:
            st.warning("No recommendations for this cart -- try a SKU combination seen in the training orders.")
        else:
            for sku, (confidence, lift) in ranked:
                st.write(f"**{sku_lookup.loc[sku, 'product_name']}** ({sku}) "
                         f"-- confidence {confidence:.1%}, lift {lift:.2f}")