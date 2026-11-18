import gradio as gr
import joblib
import numpy as np
import pandas as pd
import spaces


# -----------------------------------------------------------------------
# Hugging Face's ZeroGPU initialization check
# -----------------------------------------------------------------------
@spaces.GPU
def _warm_zerogpu():
    return None


_warm_zerogpu()

# -----------------------------------------------------------------------
# Load all persisted artifacts ONCE at startup
# -----------------------------------------------------------------------
regression_pipeline = joblib.load("farm_income_linear_regression_pipeline.joblib")
classification_pipeline = joblib.load("invoice_payment_risk_final_pipeline.joblib")
label_encoder = joblib.load("payment_risk_label_encoder.joblib")

clustering_bundle = joblib.load("farmer_segmentation_bundle.joblib")
clustering_pipeline = clustering_bundle["pipeline"]
clustering_meta = clustering_bundle["metadata"]

arm_artifact = joblib.load("food_delivery_orders_apriori_recommendation_rules.joblib")
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]


# -----------------------------------------------------------------------
# 1. Regression -- predict Farm Income
# -----------------------------------------------------------------------
def predict_revenue(crop_type, county, irrigation_type, cooperative_member,
                    farm_size_acres, years_farming_experience, num_workers_employed,
                    rainfall_mm, soil_quality_score, market_distance_km,
                    machinery_value_kes, fertilizer_spend_kes, pesticide_spend_kes,
                    irrigation_maintenance_cost_kes):
    payload = {
        "crop_type": crop_type, "county": county, "irrigation_type": irrigation_type,
        "cooperative_member": cooperative_member, "farm_size_acres": farm_size_acres,
        "years_farming_experience": years_farming_experience, "num_workers_employed": num_workers_employed,
        "rainfall_mm": rainfall_mm, "soil_quality_score": soil_quality_score,
        "market_distance_km": market_distance_km, "machinery_value_kes": machinery_value_kes,
        "fertilizer_spend_kes": fertilizer_spend_kes, "pesticide_spend_kes": pesticide_spend_kes,
        "irrigation_maintenance_cost_kes": irrigation_maintenance_cost_kes
    }
    input_row = pd.DataFrame([payload])
    predicted_revenue_kes = float(regression_pipeline.predict(input_row)[0])
    return f"KES {predicted_revenue_kes:,.2f}"


# -----------------------------------------------------------------------
# 2. Classification -- predict Invoice payment risk
# -----------------------------------------------------------------------
def predict_credit_risk(client_industry, client_credit_rating, client_company_size,
                        payment_method, has_dispute_history, collateral_or_guarantee,
                        payment_terms, invoice_amount_kes, client_relationship_years,
                        prior_late_payments_count, discount_offered_pct,
                        days_outstanding_avg_historical, guarantee_value_kes,
                        economic_sector_growth_rate_pct):
    payload = {
        "client_industry": client_industry, "client_credit_rating": client_credit_rating,
        "client_company_size": client_company_size, "payment_method": payment_method,
        "has_dispute_history": has_dispute_history, "collateral_or_guarantee": collateral_or_guarantee,
        "payment_terms": payment_terms, "invoice_amount_kes": invoice_amount_kes,
        "client_relationship_years": client_relationship_years,
        "prior_late_payments_count": prior_late_payments_count,
        "discount_offered_pct": discount_offered_pct,
        "days_outstanding_avg_historical": days_outstanding_avg_historical,
        "guarantee_value_kes": guarantee_value_kes,
        "economic_sector_growth_rate_pct": economic_sector_growth_rate_pct
    }
    input_row = pd.DataFrame([payload])

    predicted_class_encoded = classification_pipeline.predict(input_row)[0]
    predicted_class = label_encoder.inverse_transform([predicted_class_encoded])[0]

    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_labels = label_encoder.inverse_transform(
        classification_pipeline.named_steps["model"].classes_
    )

    probability_lines = "\n".join(
        f"  {label}: {prob:.1%}" for label, prob in zip(class_labels, predicted_probabilities)
    )
    return f"Predicted category: {predicted_class}\n\n{probability_lines}"


# -----------------------------------------------------------------------
# 3. Clustering -- assign a new farmer to an existing segment
# -----------------------------------------------------------------------
def predict_segment(farm_size_acres, annual_yield_tonnes, income_per_acre_kes,
                    years_farming_experience, market_access_score):
    payload = {
        "farm_size_acres": farm_size_acres,
        "annual_yield_tonnes": annual_yield_tonnes,
        "income_per_acre_kes": income_per_acre_kes,
        "years_farming_experience": years_farming_experience,
        "market_access_score": market_access_score or None,
    }
    features = clustering_meta["numeric_features"]
    input_row = pd.DataFrame([payload]).reindex(columns=features).astype(float)

    assigned_cluster = int(clustering_pipeline.predict(input_row)[0])
    segment_name = clustering_meta["cluster_names"][assigned_cluster]
    return f"Assigned segment: {segment_name} (Cluster {assigned_cluster})"


# -----------------------------------------------------------------------
# 4. Association Rule Mining -- recommend food items for an order cart
# -----------------------------------------------------------------------
def recommend_products(cart_text, top_n):
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
        return "No recommendations for this cart."

    lines = [
        f"{sku_lookup.loc[sku, 'product_name']} ({sku})  --  "
        f"confidence {confidence:.1%}, lift {lift:.2f}"
        for sku, (confidence, lift) in ranked
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------
# Gradio Application
# -----------------------------------------------------------------------
with gr.Blocks(title="BBT 4206 Model Validation Demos") as demo:
    gr.Markdown("# BBT 4206 -- Model Validation Demos\nSwitch tabs to try a different model.")

    with gr.Tab("Farm Income Regressor"):
        gr.Interface(
            fn=predict_revenue,
            inputs=[
                gr.Textbox(label="Crop Type", value="Maize"),
                gr.Textbox(label="County", value="Nakuru"),
                gr.Textbox(label="Irrigation Type", value="Rain-fed"),
                gr.Textbox(label="Cooperative Member", value="Yes"),
                gr.Number(label="Farm Size (Acres)", value=2.5),
                gr.Number(label="Years Farming Experience", value=10),
                gr.Number(label="Number of Workers Employed", value=3),
                gr.Number(label="Rainfall (mm)", value=850.0),
                gr.Number(label="Soil Quality Score", value=7.5),
                gr.Number(label="Market Distance (km)", value=12.4),
                gr.Number(label="Machinery Value (KES)", value=50000),
                gr.Number(label="Fertilizer Spend (KES)", value=15000),
                gr.Number(label="Pesticide Spend (KES)", value=5000),
                gr.Number(label="Irrigation Maintenance Cost (KES)", value=0),
            ],
            outputs=gr.Textbox(label="Predicted Monthly Farm Income"),
        )

    with gr.Tab("Invoice Payment Risk Classifier"):
        gr.Interface(
            fn=predict_credit_risk,
            inputs=[
                gr.Textbox(label="Client Industry", value="Manufacturing"),
                gr.Dropdown(label="Client Credit Rating", choices=["Poor", "Fair", "Good", "Excellent"],
                            value="Excellent"),
                gr.Textbox(label="Client Company Size", value="Medium"),
                gr.Textbox(label="Payment Method", value="Cheque"),
                gr.Textbox(label="Has Dispute History", value="Yes"),
                gr.Textbox(label="Collateral or Guarantee", value="Yes"),
                gr.Number(label="Payment Terms (Days)", value=90),
                gr.Number(label="Invoice Amount (KES)", value=114907),
                gr.Number(label="Client Relationship Years", value=6.2),
                gr.Number(label="Prior Late Payments Count", value=3),
                gr.Number(label="Discount Offered (%)", value=1.65),
                gr.Number(label="Days Outstanding (Avg Historical)", value=19.7),
                gr.Number(label="Guarantee Value (KES)", value=69619.19),
                gr.Number(label="Economic Sector Growth Rate (%)", value=1.42),
            ],
            outputs=gr.Textbox(label="Predicted Invoice Payment Risk Category"),
        )

    with gr.Tab("Farmer Segmenter"):
        gr.Interface(
            fn=predict_segment,
            inputs=[
                gr.Number(label="Farm Size (acres)", value=2.5),
                gr.Number(label="Annual Yield (tonnes)", value=8),
                gr.Number(label="Income per Acre (KES)", value=28000),
                gr.Number(label="Years Farming Experience", value=6),
                gr.Number(label="Market Access Score (Leave 0 if NA)", value=50),
            ],
            outputs=gr.Textbox(label="Prediction"),
        )

    with gr.Tab("Food Delivery Recommender"):
        gr.Interface(
            fn=recommend_products,
            inputs=[
                gr.Textbox(label="Cart (comma-separated SKUs)", value="F11, F10"),
                gr.Number(label="Top N Recommendations", value=3),
            ],
            outputs=gr.Textbox(label="Recommendations"),
        )

if __name__ == "__main__":
    demo.launch()