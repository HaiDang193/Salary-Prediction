
"""Car Price Prediction Streamlit App."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# Project root
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.predict import load_model, predict_prices
from src.preprocessing import REFERENCE_YEAR


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered",
)

st.title("🚗 Car Price Prediction")

st.caption(
    "Predict used car selling prices using XGBoost. "
    "All prices are estimated in Indian Rupees (INR)."
)


# ==========================================
# LOAD MODEL
# ==========================================

try:
    load_model()
except Exception as exc:
    st.error(f"Cannot load model: {exc}")
    st.stop()


# ==========================================
# USER INPUT
# ==========================================

with st.form("prediction_form"):

    st.subheader("Vehicle Information")

    name = st.text_input(
        "Car Name",
        value="Maruti Swift Dzire VDI",
    )

    year = st.number_input(
        "Manufacturing Year",
        min_value=1980,
        max_value=REFERENCE_YEAR,
        value=2018,
        step=1,
    )

    km_driven = st.number_input(
        "Kilometres Driven",
        min_value=0,
        max_value=2000000,
        value=50000,
        step=1000,
    )

    fuel = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "CNG", "LPG", "Electric"],
    )

    seller_type = st.selectbox(
        "Seller Type",
        [
            "Individual",
            "Dealer",
            "Trustmark Dealer",
        ],
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"],
    )

    owner = st.selectbox(
        "Owner",
        [
            "First Owner",
            "Second Owner",
            "Third Owner",
            "Fourth & Above Owner",
            "Test Drive Car",
        ],
    )

    submitted = st.form_submit_button(
        "Predict Price",
        use_container_width=True,
    )


# ==========================================
# PREDICTION
# ==========================================

if submitted:

    if not name.strip():
        st.error("Please enter a car name.")

    else:
        car = pd.DataFrame([{
            "name": name.strip(),
            "year": int(year),
            "selling_price": 0,
            "km_driven": int(km_driven),
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
        }])

        try:
            predicted_price = float(
                predict_prices(car)[0]
            )

            st.success("Prediction completed!")

            st.metric(
                label="Estimated Selling Price",
                value=f"₹{predicted_price:,.0f}",
            )

            st.caption(
                "This is a model estimate, not a "
                "guaranteed market selling price."
            )

        except Exception as exc:
            st.error(
                f"Prediction failed: {exc}"
            )
