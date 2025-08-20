import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_companies = 200
data = {
    'CurrentRatio': np.random.uniform(0.5, 3, num_companies),
    'DebtToEquity': np.random.uniform(0.1, 2, num_companies),
    'ProfitMargin': np.random.uniform(-0.1, 0.3, num_companies),
    'ReturnOnAssets': np.random.uniform(-0.1, 0.2, num_companies),
    'Bankrupt': np.random.choice([0, 1], size=num_companies, p=[0.8, 0.2]) # 20% bankruptcy rate
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Feature Engineering ---
# No significant cleaning needed for synthetic data, but this section would be crucial for real-world data.
# Example: Handling missing values, outlier detection, etc.
# --- 3. Model Training ---
X = df.drop('Bankrupt', axis=1)
y = df['Bankrupt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000) # Increased max_iter to ensure convergence
model.fit(X_train, y_train)
# --- 4. Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
# --- 5. Visualization ---
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Solvent', 'Bankrupt'], yticklabels=['Solvent', 'Bankrupt'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
# Save the plot to a file
output_filename = 'confusion_matrix.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.figure(figsize=(8,6))
feature_importance = pd.Series(model.coef_[0], index=X.columns)
feature_importance.plot(kind='bar')
plt.title('Feature Importance')
plt.ylabel('Coefficient')
plt.tight_layout()
output_filename = 'feature_importance.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")