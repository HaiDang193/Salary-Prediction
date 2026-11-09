
"""Test inference using the saved model."""

import pandas as pd

from src.predict import predict_prices


def main():

    car = pd.DataFrame([{
        "name": "Maruti Swift Dzire VDI",
        "year": 2018,
        "selling_price": 0,
        "km_driven": 50000,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
    }])

    price = predict_prices(car)[0]

    print(f"Predicted price: INR {price:,.0f}")


if __name__ == "__main__":
    main()
