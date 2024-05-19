import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression  # Import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, confusion_matrix

df = pd.read_csv(r"C:\Users\HP\Documents\Master's thesis\collected data\collected data.csv")

categorical_features = ["Project_name", "country"]
le = LabelEncoder()
for feature in categorical_features:
  df[feature] = le.fit_transform(df[feature])

numerical_features = ["backers_count", "goal", "pledged", "Average Contribution"]
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

X = df.drop(columns='state')
y = df['state']
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define Logistic Regression model
model = LogisticRegression(random_state=8)  # Set random state for reproducibility

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn = cm[0][0]
fp = cm[0][1]
specificity = tn / (tn + fp)

print("Specificity:", specificity)
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)
print("F1-Score:", f1)
print("AUC-ROC:", auc)
print("Recall:", recall)