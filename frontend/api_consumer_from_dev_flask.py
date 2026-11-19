
"""
api_consumer_from_dev_flask.py

A Python client demonstrating how to consume all four prediction
endpoints exposed by the local Flask API.

Start the Flask API first:
    python api.py

Then run this client:
    python frontend/api_consumer_from_dev_flask.py

The client demonstrates:
    1. Farm annual income regression
    2. Invoice payment risk classification
    3. Farmer segmentation
    4. Food delivery item recommendations
"""

import requests

BASE_URL = "http://127.0.0.1:5000"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "ServingMLModels-Client/1.0",
}


def call_endpoint(path, payload):
    """Send a POST request and return its JSON response.

    Returns None if the request fails or the response cannot be decoded.
    """
    url = f"{BASE_URL}{path}"

    try:
        response = requests.post(
            url,
            json=payload,
            headers=HEADERS,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as error:
        print("HTTP error:", error)
        if error.response is not None:
            print("Response body:", error.response.text)

    except requests.exceptions.RequestException as error:
        print("Request failed:", error)

    except ValueError as error:
        print("Invalid JSON response:", error)

    return None


# -----------------------------------------------------------------------
# 1. REGRESSION -- predict annual farm income (KES)
# -----------------------------------------------------------------------

regression_payload = {
    "soil_quality_score": 7.8,
    "farm_size_acres": 5.5,
    "years_farming_experience": 14,
    "num_workers_employed": 4,
    "rainfall_mm": 1100,
    "market_distance_km": 12.5,
    "machinery_value_kes": 450000,
    "fertilizer_spend_kes": 65000,
    "pesticide_spend_kes": 35000,
    "irrigation_maintenance_cost_kes": 25000,
    "crop_type": "Horticulture",
    "county": "Nakuru",
    "irrigation_type": "Drip",
    "cooperative_member": "Yes",
}

result = call_endpoint(
    "/api/v1/models/farm-income-regressor/predictions",
    regression_payload,
)

if result is not None:
    print(
        "Predicted annual farm income (KES) =",
        result.get("predicted_annual_farm_income_kes"),
    )
    print("Model =", result.get("model"))

print("-" * 60)


# -----------------------------------------------------------------------
# 2. CLASSIFICATION -- classify invoice payment risk
# -----------------------------------------------------------------------

classification_payload = {
    "invoice_amount_kes": 120000,
    "guarantee_value_kes": 50000,
    "days_outstanding_avg_historical": 15,
    "client_credit_rating": "Good",
    "has_dispute_history": "No",
}

result = call_endpoint(
    "/api/v1/models/invoice_payment_risk_final_pipeline/predictions",
    classification_payload,
)

if result is not None:
    print(
        "Predicted invoice payment category =",
        result.get("predicted_invoice_payment_category"),
    )
    print("Class probabilities =", result.get("class_probabilities"))
    print("Model =", result.get("model"))

print("-" * 60)


# -----------------------------------------------------------------------
# 3. CLUSTERING -- assign a farmer to a segment
# -----------------------------------------------------------------------

clustering_payload = {
    "farm_size_acres": 3.6,
    "annual_yield_tonnes": 9.8,
    "income_per_acre_kes": 39000,
    "years_farming_experience": 9,
    "market_access_score": 53,
}

result = call_endpoint(
    "/api/v1/models/farmer-segmenter/predictions",
    clustering_payload,
)

if result is not None:
    print("Assigned cluster =", result.get("assigned_cluster"))
    print("Farmer segment =", result.get("segment_name"))
    print("Model =", result.get("model"))

print("-" * 60)


# -----------------------------------------------------------------------
# 4. ASSOCIATION RULE MINING -- recommend food delivery items
# -----------------------------------------------------------------------

recommendation_payload = {
    "cart": ["F10", "F11"],
    "top_n": 3,
}

result = call_endpoint(
    "/api/v1/models/food-delivery-recommender/predictions",
    recommendation_payload,
)

if result is not None:
    print("Cart =", result.get("cart"))
    print("Recommendations =", result.get("recommendations"))
    print("Model =", result.get("model"))

print("-" * 60)
