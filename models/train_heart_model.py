import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("heart.csv")

# Select required features
X = data[['age', 'trestbps', 'chol', 'thalach']]
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Random Forest trained successfully")
print("Model accuracy:", accuracy)

# Save trained model
joblib.dump(model, "heart_model.pkl")
print("Model saved as heart_model.pkl")
