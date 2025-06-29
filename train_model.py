import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Generate synthetic data
n_samples = 1000

data = {
    'age': np.random.randint(18, 90, size=n_samples),
    'num_procedures': np.random.poisson(lam=1.5, size=n_samples),
    'num_medications': np.random.poisson(lam=7, size=n_samples),
    'time_in_hospital': np.random.randint(1, 15, size=n_samples),
}

df = pd.DataFrame(data)

# Create a label: "readmitted" based on some conditions
df['readmitted'] = np.where(
    (df['time_in_hospital'] > 7) | 
    (df['num_medications'] > 10), 
    1, 0
)

# Save the generated dataset
df.to_csv('patient_readmission_data.csv', index=False)
print("✅ Dataset saved as 'patient_readmission_data.csv'")

# Features and label
X = df[['age', 'num_procedures', 'num_medications', 'time_in_hospital']]
y = df['readmitted']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the model
joblib.dump(model, 'readmission_model.pkl')
print("✅ Model saved as 'readmission_model.pkl'")
