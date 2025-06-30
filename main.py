import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings("ignore")

# ✅ Correct path to CSV file
file_path = r'C:\Users\wahab\Desktop\iris-eda\archive\Churn_Modelling.xlsx.csv'

# ✅ Check file exists and read CSV
if os.path.exists(file_path):
    print("✅ File mil gayi! Reading as CSV...")
    df = pd.read_csv(file_path)
else:
    print("❌ File nahi mili! Path check karo:")
    print(file_path)
    exit()

# ✅ Show available columns
print("📊 Available columns:")
print(df.columns.tolist())

# ✅ Drop irrelevant columns if they exist
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True, errors='ignore')

# ✅ Check for missing values
assert df.isnull().sum().sum() == 0, "❌ Missing values found"

# ✅ Encode categorical columns
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# ✅ Outlier removal using IQR
for col in ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# ✅ Feature/target split
X = df.drop('Exited', axis=1)
y = df['Exited']

# ✅ Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# ✅ Scale features
scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)

# ✅ Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X_res_scaled, y_res, test_size=0.2, stratify=y_res, random_state=42)

# ✅ Train multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# ✅ Evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    print(f"\n🔍 {name} Results:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {auc:.4f}")

# ✅ Choose best model (XGBoost)
best_model = models['XGBoost']

# ✅ SHAP Explainability
explainer = shap.Explainer(best_model)
shap_values = explainer(X_train)

shap.summary_plot(shap_values, X, plot_type="bar")
shap.summary_plot(shap_values, X)

# ✅ Feature importance bar plot
feature_importance = pd.Series(best_model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', figsize=(10, 5), title='Feature Importance')
plt.tight_layout()
plt.show()
