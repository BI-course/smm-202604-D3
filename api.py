"""
api.py

A REST-style API built with Flask to serve four machine learning
applications developed as part of the Business Intelligence II lab.

The API provides the following prediction services:

    1. Regression:
       Farm annual income prediction using a trained Linear Regression
       pipeline.
    2. Classification:
       Invoice payment risk classification using a trained machine
       learning pipeline and label encoder.
    3. Clustering:
       Farmer segmentation using a trained K-Means clustering pipeline.
    4. Association Rule Mining:
       Food item recommendations based on shopping cart contents using
       Apriori association rules.

The regression, classification and clustering models use saved
preprocessing pipelines to handle the transformations required during
inference. This avoids manually reproducing training-time preprocessing
when making predictions.

The association rule mining endpoint uses a saved recommendation lookup
containing the discovered association rules and food item information.

The API validates incoming JSON requests, checks required features and
returns predictions or recommendations in JSON format.

Run this file directly to start the local development server:

    python api.py
"""

import os
import numpy as np
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# -----------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
# -----------------------------------------------------------------------
# By default, a browser blocks JavaScript on one origin (e.g., a page served
# from http://127.0.0.1:5500) from calling an API on a different origin
# (e.g., http://127.0.0.1:5000). `flask_cors` relaxes that restriction for the
# specific origins we trust during local development.
#
# A few alternative configurations are shown below (commented out) so you can
# see how the restriction can be loosened or tightened depending on your
# deployment target. Only ONE of these should be active at a time.

# Option A (the most permissive. This is fine for a quick classroom demo or
# local development, *NEVER* for production):
# CORS(app)

# Option B (allow only a specific frontend origin):
# CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5500"}})

# Option C (the one actually in use here -- allow localhost/127.0.0.1 on the
# ports our lab's frontend files are typically served from):
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://localhost:443",
    "https://127.0.0.1:443",
    "http://localhost:63342",
    "null",
]}})

# -----------------------------------------------------------------------
# The best practice is to load the four persisted artifacts ONCE, at startup,
# not per-request.
# Loading inside a route function would re-read the file from disk on every
# single API call, which is slow and pointless -- the fitted model never
# changes while the server is running.
# -----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

regression_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "farm_income_linear_regression_pipeline.joblib")
)

classification_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "invoice_payment_risk_final_pipeline.joblib")
)
label_encoder = joblib.load(
    os.path.join(MODEL_DIR, "payment_risk_label_encoder.joblib")
)

clustering_bundle = joblib.load(
    os.path.join(MODEL_DIR, "farmer_segmentation_bundle.joblib")
)
clustering_pipeline = clustering_bundle["pipeline"]
clustering_meta = clustering_bundle["metadata"]

arm_artifact = joblib.load(
    os.path.join(MODEL_DIR, "food_delivery_orders_apriori_recommendation_rules.joblib")
)
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]

print("All the four artifacts have been loaded. The API is ready.")


# -----------------------------------------------------------------------
# 1. REGRESSION -- predict annual farm income (KES)
# -----------------------------------------------------------------------
#
# Endpoint:
# POST /api/v1/models/farm-income-regressor/predictions
#
# Required numerical features:
# soil_quality_score, farm_size_acres, years_farming_experience,
# num_workers_employed, rainfall_mm, market_distance_km,
# machinery_value_kes, fertilizer_spend_kes, pesticide_spend_kes,
# irrigation_maintenance_cost_kes
#
# Required categorical features:
# crop_type, county, irrigation_type, cooperative_member
#
# Sample request (cURL):
# curl -X POST http://127.0.0.1:5000/api/v1/models/farm-income-regressor/predictions \
#      -H "Content-Type: application/json" \
#      -d '{"soil_quality_score":7.8,"farm_size_acres":5.5,
#           "years_farming_experience":14,"num_workers_employed":4,
#           "rainfall_mm":1100,"market_distance_km":12.5,
#           "machinery_value_kes":450000,"fertilizer_spend_kes":65000,
#           "pesticide_spend_kes":35000,
#           "irrigation_maintenance_cost_kes":25000,
#           "crop_type":"Horticulture","county":"Nakuru",
#           "irrigation_type":"Drip","cooperative_member":"Yes"}'
#
# Successful response includes:
# predicted_annual_farm_income_kes and model.
#
@app.route("/api/v1/models/farm-income-regressor/predictions", methods=["POST"])
def predict_revenue():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Request body must be a valid JSON object."
        }), 400

    numeric_features = [
        "soil_quality_score",
        "farm_size_acres",
        "years_farming_experience",
        "num_workers_employed",
        "rainfall_mm",
        "market_distance_km",
        "machinery_value_kes",
        "fertilizer_spend_kes",
        "pesticide_spend_kes",
        "irrigation_maintenance_cost_kes",
    ]

    categorical_features = [
        "crop_type",
        "county",
        "irrigation_type",
        "cooperative_member",
    ]

    required_features = numeric_features + categorical_features

    missing_features = [
        feature for feature in required_features
        if feature not in payload or payload[feature] is None
    ]

    if missing_features:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing_features,
        }), 400

    try:
        for feature in numeric_features:
            value = payload[feature]

            if isinstance(value, bool):
                raise ValueError

            numeric_value = float(value)

            if (
                pd.isna(numeric_value)
                or numeric_value == float("inf")
                or numeric_value == float("-inf")
            ):
                raise ValueError

            payload[feature] = numeric_value

    except (TypeError, ValueError):
        return jsonify({
            "error": "Numerical features must contain valid finite numbers."
        }), 400

    for feature in categorical_features:
        if not isinstance(payload[feature], str) or not payload[feature].strip():
            return jsonify({
                "error": f"{feature} must be a non-empty string."
            }), 400

        payload[feature] = payload[feature].strip()

    input_row = pd.DataFrame([
        {feature: payload[feature] for feature in required_features}
    ])

    predicted_income_kes = float(
        regression_pipeline.predict(input_row)[0]
    )

    return jsonify({
        "predicted_annual_farm_income_kes": round(predicted_income_kes, 2),
        "model": "Farm Income Linear Regression",
    })


# -----------------------------------------------------------------------
# 2. CLASSIFICATION -- classify invoice payment risk
# -----------------------------------------------------------------------
#
# Endpoint:
# POST /api/v1/models/invoice_payment_risk_final_pipeline/predictions
#
# Required numerical features:
# invoice_amount_kes, guarantee_value_kes,
# days_outstanding_avg_historical
#
# Required categorical features:
# client_credit_rating, has_dispute_history
#
# Sample request (cURL):
# curl -X POST http://127.0.0.1:5000/api/v1/models/invoice_payment_risk_final_pipeline/predictions \
#      -H "Content-Type: application/json" \
#      -d '{"invoice_amount_kes":120000,
#           "guarantee_value_kes":50000,
#           "days_outstanding_avg_historical":15,
#           "client_credit_rating":"Good",
#           "has_dispute_history":"No"}'
#
# Successful response includes:
# predicted_invoice_payment_category, class_probabilities and model.
#
# The label encoder converts the predicted encoded class back into
# its original payment category.
#
@app.route("/api/v1/models/invoice_payment_risk_final_pipeline/predictions", methods=["POST"])
def predict_invoice_payment_risk():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a valid JSON object."}), 400

    required_features = [
        "invoice_amount_kes",
        "guarantee_value_kes",
        "days_outstanding_avg_historical",
        "client_credit_rating",
        "has_dispute_history",
    ]

    missing_features = [
        feature for feature in required_features
        if feature not in payload or payload[feature] is None
    ]

    if missing_features:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing_features,
        }), 400

    numeric_features = [
        "invoice_amount_kes",
        "guarantee_value_kes",
        "days_outstanding_avg_historical",
    ]

    try:
        for feature in numeric_features:
            value = payload[feature]

            if isinstance(value, bool):
                raise ValueError

            numeric_value = float(value)

            if not np.isfinite(numeric_value):
                raise ValueError

            payload[feature] = numeric_value
    except (TypeError, ValueError):
        return jsonify({
            "error": "Numerical features must contain valid finite numbers."
        }), 400

    categorical_features = [
        "client_credit_rating",
        "has_dispute_history",
    ]

    for feature in categorical_features:
        if not isinstance(payload[feature], str) or not payload[feature].strip():
            return jsonify({
                "error": f"{feature} must be a non-empty string."
            }), 400

        payload[feature] = payload[feature].strip()

    input_row = pd.DataFrame(
        [{feature: payload[feature] for feature in required_features}]
    )

    predicted_class_encoded = classification_pipeline.predict(input_row)[0]

    predicted_class = label_encoder.inverse_transform(
        [predicted_class_encoded]
    )[0]

    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]

    class_labels = label_encoder.inverse_transform(
        classification_pipeline.named_steps["model"].classes_
    )

    return jsonify({
        "predicted_invoice_payment_category": str(predicted_class),
        "class_probabilities": {
            str(label): round(float(prob), 4)
            for label, prob in zip(class_labels, predicted_probabilities)
        },
        "model": "Invoice Payment Risk Classifier",
    })


# -----------------------------------------------------------------------
# 3. CLUSTERING -- assign a farmer to a segment
# -----------------------------------------------------------------------
#
# Endpoint:
# POST /api/v1/models/farmer-segmenter/predictions
#
# Required features:
# farm_size_acres, annual_yield_tonnes,
# income_per_acre_kes, years_farming_experience
#
# Optional feature:
# market_access_score
#
# Sample request (cURL):
# curl -X POST http://127.0.0.1:5000/api/v1/models/farmer-segmenter/predictions \
#      -H "Content-Type: application/json" \
#      -d '{"farm_size_acres":3.6,
#           "annual_yield_tonnes":9.8,
#           "income_per_acre_kes":39000,
#           "years_farming_experience":9,
#           "market_access_score":53}'
#
# market_access_score may be omitted or null. The saved pipeline
# handles missing values through its preprocessing steps.
#
# Successful response includes assigned_cluster, segment_name and model.
#
@app.route("/api/v1/models/farmer-segmenter/predictions", methods=["POST"])
def predict_farmer_segment():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a valid JSON object."}), 400

    features = clustering_meta["numeric_features"]

    # Everything except market_access_score is required
    required = [f for f in features if f != "market_access_score"]
    missing = [f for f in required if payload.get(f) is None]
    if missing:
        return jsonify({"error": "Missing required field(s)", "missing": missing}), 400

    try:
        validated_payload = {}
        for feature in features:
            value = payload.get(feature)
            # market_access_score is optional and may be null.
            if feature == "market_access_score" and value is None:
                validated_payload[feature] = np.nan
                continue
            if isinstance(value, bool):
                raise ValueError
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                raise ValueError
            validated_payload[feature] = numeric_value
        # Preserve the exact feature order expected by the trained model.
        input_row = pd.DataFrame(
            [validated_payload],
            columns=features
        )
    except (ValueError, TypeError):
        return jsonify({
            "error": "All supplied clustering features must contain valid finite numbers."
        }), 400

    assigned_cluster = int(clustering_pipeline.predict(input_row)[0])

    return jsonify({
        "assigned_cluster": assigned_cluster,
        "segment_name": clustering_meta["cluster_names"][assigned_cluster],
        "model": "K-Means (k=3, imputed and standardized features)",
    })


# -----------------------------------------------------------------------
# 4. ASSOCIATION RULE MINING -- recommend food delivery items
# -----------------------------------------------------------------------
#
# Endpoint:
# POST /api/v1/models/food-delivery-recommender/predictions
#
# Required input:
# cart: a non-empty list of food item IDs.
#
# Optional input:
# top_n: number of recommendations to return (default: 3, range: 1-20).
#
# Sample request (cURL):
# curl -X POST http://127.0.0.1:5000/api/v1/models/food-delivery-recommender/predictions \
#      -H "Content-Type: application/json" \
#      -d '{"cart":["F10","F11"],"top_n":3}'
#
# The saved Apriori association rules are used to identify recommended
# food items that are not already present in the cart.
#
# Successful response includes cart, recommendations and model.
# Each recommendation contains item_id, product_name, confidence and lift.
#
@app.route("/api/v1/models/food-delivery-recommender/predictions", methods=["POST"])
def recommend_food_items():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Request body must be a valid JSON object."
        }), 400

    if "cart" not in payload:
        return jsonify({
            "error": "Missing required field: cart."
        }), 400

    cart_input = payload["cart"]

    if not isinstance(cart_input, list) or not cart_input:
        return jsonify({
            "error": "Cart must be a non-empty list of item IDs."
        }), 400

    if any(not isinstance(item, str) or not item.strip() for item in cart_input):
        return jsonify({
            "error": "Every cart item must be a non-empty string."
        }), 400

    cart = set(item.strip() for item in cart_input)

    top_n = payload.get("top_n", 3)

    if isinstance(top_n, bool):
        return jsonify({
            "error": "top_n must be a positive integer."
        }), 400

    try:
        numeric_top_n = float(top_n)
        if not np.isfinite(numeric_top_n) or not numeric_top_n.is_integer():
            raise ValueError
        top_n = int(numeric_top_n)
    except (TypeError, ValueError):
        return jsonify({
            "error": "top_n must be a positive integer."
        }), 400

    if top_n < 1 or top_n > 20:
        return jsonify({
            "error": "top_n must be between 1 and 20."
        }), 400

    candidates = {}

    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for sku, confidence, lift in consequent_list:
                if sku not in cart:
                    if sku not in candidates or confidence > candidates[sku][0]:
                        candidates[sku] = (confidence, lift)

    ranked = sorted(
        candidates.items(),
        key=lambda item: item[1][0],
        reverse=True
    )[:top_n]

    recommendations = [
        {
            "item_id": str(sku),
            "product_name": sku_lookup.loc[sku, "product_name"],
            "confidence": round(float(confidence), 3),
            "lift": round(float(lift), 2),
        }
        for sku, (confidence, lift) in ranked
    ]

    return jsonify({
        "cart": sorted(cart),
        "recommendations": recommendations,
        "model": "Apriori association rules, food delivery menu lookup",
    })


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Development server. debug=True gives auto-reload and an interactive
    # debugger on errors -- convenient in class, but NEVER use debug=True in
    # a deployed/public environment (it can expose a remote code execution
    # console to anyone who can reach the server).
    app.run(debug=True, use_reloader=False)

    # Production-style alternative (no debugger, no auto-reload):
    # app.run(debug=False, host="0.0.0.0", port=5000)

    # HTTPS/TLS alternative for local development (requires a self-signed
    # certificate; see app_server_reverse_proxy_server_setup.md):
    # app.run(debug=False, ssl_context=("cert.pem", "key.pem"))
