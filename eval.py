import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler  
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from category_encoders.target_encoder import TargetEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, History
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


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

eval_set = [(X_val, y_val)]  # Create a list of validation sets

estimators = [
    ('encoder', TargetEncoder()),
    ("scaler", StandardScaler()),
    ('clf', XGBClassifier(eval_set=eval_set, random_state=8))  # Pass eval_set here
]
pipe = Pipeline(steps = estimators)

Search_space = {
    'clf__max_depth' : Integer(2,8),
    'clf__learning_rate':Real(0.001,1.0,prior='log-uniform'),
    'clf__subsample': Real(0.5, 1.0),
    'clf__colsample_bytree': Real(0.5, 1.0),
    'clf__colsample_bylevel': Real(0.5, 1.0),
    'clf__colsample_bynode': Real(0.5, 1.0),
    'clf__reg_alpha': Real(0.0, 10.0),
    'clf__reg_lambda': Real(0.0, 10.0),
    'clf__gamma': Real(0.0, 10.0),
}

opt = BayesSearchCV(pipe, Search_space, cv=3, n_iter=10, scoring='roc_auc', random_state=8)

opt.fit(X_train, y_train)

opt.score(X_val, y_val)

xgboost_step = opt.best_estimator_.steps[2]

y_pred = opt.best_estimator_.predict(X_test)
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

xgboost_model = opt.best_estimator_.steps[-1][1]  # Accessing the classifier step


training_history = xgboost_model.evals_result_

loss_history = training_history['validation_0']['loss']  # Assuming validation loss is tracked
accuracy_history = training_history['validation_0']['mean_accuracy']
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

plt.figure()  

plt.plot(accuracy_history, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.legend()

# plt.show()
# ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
# ax.set_title('XGBoost Confusion Matrix')
# ax.set_xlabel('Predicted Label')
# ax.set_ylabel('True Label')
# plt.show()
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# auc = roc_auc_score(y_test, y_pred)
# plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % auc)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.show()
# model = opt.best_estimator_
# joblib.dump(model, "prediction.model")


