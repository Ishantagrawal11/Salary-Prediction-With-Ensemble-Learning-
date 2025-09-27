import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import joblib


data = pd.read_excel("Employees.xlsx")


# Exploratory Data Analysis (EDA)

print(data)
print(data.head())
print(data.info())
print("Shape:", data.shape)
print("Missing values\n", data.isna().sum())
print("Duplicated rows:", data.duplicated().sum())

# Pie chart of Gender
data["Gender"].value_counts().sort_values(ascending=False).plot(kind="pie", autopct="%1.1f%%")
plt.title("Pie chart of Gender")
plt.ylabel("")
plt.show()

# Histogram of Job Rate
plt.hist(data["Job Rate"], bins=10)
plt.title("Histogram of Job Rate")
plt.xlabel("Rate")
plt.ylabel("Count")
plt.show()
print(data["Job Rate"].describe())

# Average salary by department
data.groupby("Department")["Annual Salary"].mean().sort_values(ascending=False).head(7).plot(kind="bar")
plt.title("Average Salary by Department")
plt.xlabel("Department")
plt.ylabel("Average Salary")
plt.show()

# Average Monthly Salary by Center
print(data.groupby("Center")["Monthly Salary"].mean().sort_values(ascending=False))
data.groupby("Center")["Monthly Salary"].mean().sort_values(ascending=False).plot(kind="bar")
plt.title("Average Monthly Salary by Center")
plt.xlabel("Center")
plt.ylabel("Monthly Salary")
plt.show()  
#histogram
plt.hist(data["Overtime Hours"], bins=10)
plt.title("Histogram of Overtime Hours")
plt.xlabel("Overtime Hours")
plt.ylabel("Frequency")
plt.show()
print(data["Overtime Hours"].describe())
print(data["Annual Salary"].describe())
data = pd.get_dummies(data, columns=["Department"], drop_first=True)

X = data[["Years", "Job Rate"] + [col for col in data.columns if col.startswith("Department_")]]
y_reg = data["Annual Salary"]    
y_clf = (data["Annual Salary"] >= 50000).astype(int)

print("Final Feature Columns used for training:\n", X.columns)


X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# Linear Regression

lr = LinearRegression()
lr.fit(X_train_r, y_train_r)
pred_lr = lr.predict(X_test_r)
mae_lr = mean_absolute_error(y_test_r, pred_lr)
print("\nLinear Regression Predictions:\n", pred_lr[:10])
print("Linear Regression MAE:", mae_lr)
joblib.dump(lr, "linear_model.pkl")


# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X_train_r, y_train_r)
pred_gbr = gbr.predict(X_test_r)
mae_gbr = mean_absolute_error(y_test_r, pred_gbr)
print("\nGradient Boosting Predictions:\n", pred_gbr[:10])
print("Gradient Boosting MAE:", mae_gbr)
joblib.dump(gbr, "gradient_boosting_model.pkl")

# Logistic Regression

logr = LogisticRegression(max_iter=1000)
logr.fit(X_train_c, y_train_c)
pred_logr = logr.predict(X_test_c)
acc_logr = accuracy_score(y_test_c, pred_logr)
print("\nLogistic Regression Predictions:\n", pred_logr[:10])
print("Logistic Regression Accuracy:", acc_logr)
print("Classification Report:\n", classification_report(y_test_c, pred_logr))
joblib.dump(logr, "logistic_model.pkl")


# Model Performance Comparison
plt.bar(["Linear Regression (MAE)", "Gradient Boosting (MAE)"], [mae_lr, mae_gbr])
plt.title("Regression Model Comparison (Lower = Better)")
plt.ylabel("Mean Absolute Error")
plt.show()

plt.bar(["Logistic Regression (Accuracy)"], [acc_logr])
plt.title("Classification Model Performance (Higher = Better)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()
