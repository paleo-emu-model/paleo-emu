from sklearn import make_regression
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipelines import Pipeline
from sklearn.preprocessing import StandardScaler


# Generate synthetic data
# Here, n_samples is lat, lon and n_features is the simulations
X, y = make_regression(n_samples=100, n_features=5, noise=0.1)

# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a linear regression model

pipe = Pipeline(['scaler', StandardScaler(), 
                 'transformer', PCA(n_components=15),
                 'regressor', GaussianProcessRegressor()])

pipe.fit(X_train, y_train)

# Make predictions
y_pred = pipe.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# Compare the predicted and actual values
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()
