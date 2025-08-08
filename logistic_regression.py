# Step 2: Load and Explore the Data

import pandas as pd

# Load the dataset
df = pd.read_csv("data.csv")

# ✅ Drop Unnamed: 32 column first (important!)
if 'Unnamed: 32' in df.columns:
    df.drop('Unnamed: 32', axis=1, inplace=True)


# Show the shape of the dataset
print("Dataset shape:", df.shape)

# Show first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Show info about columns
print("\nDataset info:")
print(df.info())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())


# Drop the 'id' column
df.drop('id', axis=1, inplace=True)

# Convert 'diagnosis' to numeric: M = 1, B = 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})


# Split features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Check the sizes
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)


from sklearn.preprocessing import StandardScaler

# Standardize after fixing the issue
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

# Initialize the model
log_reg = LogisticRegression()

# Train the model on the scaled training data
log_reg.fit(X_train_scaled, y_train)

print("Training Accuracy:", log_reg.score(X_train_scaled, y_train))

# Predict on test data
y_pred = log_reg.predict(X_test_scaled)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Predict probabilities
y_probs = log_reg.predict_proba(X_test_scaled)[:, 1]  # probability for class 1

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Plot
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_probs):.2f})")
plt.plot([0, 1], [0, 1], 'k--')  # random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()


# Predicted probabilities for the positive class
y_probs = log_reg.predict_proba(X_test_scaled)[:, 1]

# Display the first 10 probabilities
print("First 10 predicted probabilities:\n", y_probs[:10])

# Apply a custom threshold of 0.3
custom_threshold = 0.3
y_pred_custom = (y_probs >= custom_threshold).astype(int)

# Evaluate
print(f"\nEvaluation with Threshold = {custom_threshold}")
print(confusion_matrix(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom))
