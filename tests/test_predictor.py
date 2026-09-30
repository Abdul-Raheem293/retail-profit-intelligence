from src.models.predictor import load_model, predict_profit


def test_predict_profit_returns_number():
    model = load_model()

    input_data = {
        "Ship Mode": "Standard Class",
        "Segment": "Consumer",
        "Region": "West",
        "Category": "Furniture",
        "Sub-Category": "Bookcases",
        "Sales": 100.0,
        "Quantity": 1,
        "Discount": 0.0,
        "Order Year": 2017,
        "Order Month": 1,
        "Order Quarter": 1,
        "Order DayOfWeek": 6
    }

    prediction = predict_profit(model, input_data)

    assert isinstance(prediction, float)
