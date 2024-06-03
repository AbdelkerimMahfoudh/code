import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler  
from sklearn.model_selection import RandomizedSearchCV
from skopt.space import Real, Categorical, Integer
from category_encoders.target_encoder import TargetEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
df = pd.read_csv(r"C:\Users\HP\Documents\Master's thesis\collected data\collected data.csv")

categorical_features = ["Project_name", "country"]
le = LabelEncoder()
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

numerical_features = ["backers_count", "goal", "pledged", "Average Contribution"]
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# X = df.drop(columns='state')
# y = df['state']  
# le = LabelEncoder()
# y = le.fit_transform(y)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
evalset = [(X_val, y_val)]

estimators = [
    ('encoder', TargetEncoder()),
    ("scaler", StandardScaler()),
    ('clf', XGBClassifier(random_state=8, eval_metric='logloss'))
]
pipe = Pipeline(steps = estimators)

search_space = {
    'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    'clf__max_depth': Integer(2, 8),
    'clf__n_estimators': Integer(100, 500),
    'clf__reg_alpha': Real(0.0, 10.0),
    'clf__reg_lambda': Real(0.0, 10.0),
}


opt = RandomizedSearchCV(pipe, search_space, cv=3, n_iter=50, scoring='roc_auc', random_state=42)

opt.fit(X_train, y_train)
results = opt.evals_result()
plt.plot(results['validation_0']['logloss'], label='train')
plt.plot(results['validation_1']['logloss'], label='test')

plt.legend()

plt.show()


# opt.fit(X_train, y_train)

# opt.score(X_test, y_test)

# xgboost_step = opt.best_estimator_.steps[2]

# y_pred = opt.best_estimator_.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# auc = roc_auc_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# cm = confusion_matrix(y_test, y_pred)
# tn = cm[0][0] 
# fp = cm[0][1]  
# specificity = tn / (tn + fp)
# print("Specificity:", specificity)
# print("Confusion Matrix:\n", cm)
# print("Accuracy:", accuracy)
# print("F1-Score:", f1)
# print("AUC-ROC:", auc)
# print("Recall:", recall)



# ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
# ax.set_title('XGBoost Confusion Matrix')
# ax.set_xlabel('Predicted Label')
# ax.set_ylabel('True Label')
# plt.show()
# model = opt.best_estimator_
# joblib.dump(model, "prediction.model")



