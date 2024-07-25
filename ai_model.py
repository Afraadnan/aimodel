import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_percentage_error as MAPE, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Read and preprocess the data
df = pd.read_csv("sample3.csv")
df.columns = df.columns.str.strip()

# Impute missing values
from sklearn.impute import SimpleImputer
si = SimpleImputer(missing_values=np.nan, strategy="median")
df[['SaleUnits', 'TargetUnits', 'AchSale']] = si.fit_transform(df[['SaleUnits', 'TargetUnits', 'AchSale']])

# Plot histogram
df['SaleUnits'].plot(kind='hist', bins=20, title='SaleUnits')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

# Prepare data for modeling
X = df[['SaleUnits', 'TargetUnits']]
y = df['AchSale']

# Convert to numeric type
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Handle potential missing values introduced by conversion
df.dropna(subset=['SaleUnits', 'TargetUnits', 'AchSale'], inplace=True)
X = X.dropna()
y = y[X.index]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Scale features
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Train and evaluate linear regression model
reg = linear_model.LinearRegression()
reg.fit(X_train_scaled, y_train)

# Predict and evaluate
estimated = reg.predict(X_test_scaled)
print("Predictions:", estimated)

# Plot predictions vs actual values
plt.scatter(y_test, estimated)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Sales Predictor')
plt.show()

# Define MASE function
def mase(y_test, estimated, y_train):
    n = len(y_train)
    d = np.abs(np.diff(y_train, axis=0)).sum() / (n - 1)
    errors = np.abs(y_test - estimated)
    return errors.mean() / d

# Calculate MASE
mase_value = mase(y_test, estimated, y_train)
print("MASE:", mase_value)

# K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = []
mase_scores = []

for train_index, test_index in kfold.split(X):
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Scale features for each fold
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)

    reg.fit(x_train_scaled, y_train)
    estimated = reg.predict(x_test_scaled)

    mse = mean_squared_error(y_test, estimated)
    mse_scores.append(mse)

    mase_value = mase(y_test, estimated, y_train)
    mase_scores.append(mase_value)

print("Mean MSE:", np.mean(mse_scores))
print("Mean MASE:", np.mean(mase_scores))
