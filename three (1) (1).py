#Low Stock Prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("slate_inventory_daily.csv")

features = [
    'opening_stock',
    'received_qty',
    'usage_qty',
    'closing_stock',
    'par_level',
    'reorder_point'
]

X = df[features]
y = df['low_stock_alert']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

#Expiry Risk Prediction
from sklearn.ensemble import RandomForestClassifier

features = [
    'opening_stock',
    'usage_qty',
    'wastage_qty',
    'closing_stock'
]

X = df[features]
y = df['expiry_risk_flag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

expiry_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

expiry_model.fit(X_train, y_train)

predictions = expiry_model.predict(X_test)

print(classification_report(y_test, predictions))

#Supplier Recommendation Logic
def recommend_order(
    current_stock,
    predicted_consumption,
    safety_stock=5
):
    return max(
        0,
        predicted_consumption + safety_stock - current_stock
    )

milk_order = recommend_order(
    current_stock=15,
    predicted_consumption=20
)

print("Recommended Order:", milk_order)