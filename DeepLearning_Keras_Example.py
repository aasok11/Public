#import the relevant packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Load the concrete data from the CSV file
data = pd.read_csv(r"C:\Users\aaaso\Downloads\concrete_data.csv")

# The target variable is 'concrete_strength' named 'Strength'
X = data.drop('Strength', axis=1)
y = data['Strength']

# Split the data into training and test sets and hold 30% of the data for testing
mse_list = []

for _ in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #

    # Standardize the features using StandardScaler (commented out for first part)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build the neural network model with keras with three hidden layers, each with 10 nodes 
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))  # Output layer with 1 node for regression
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model on the training data with 50 epochs
    model.fit(X_train, y_train, epochs=50, verbose=0)

    # Evaluate the model on the test data
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)

# Output the list of mean squared errors
print("List of Mean Squared Errors:")
print(mse_list)

# Export the list of mean squared errors to a CSV file
mse_df = pd.DataFrame(mse_list, columns=['Mean Squared Error'])
mse_df.to_csv(r'C:\Users\aaaso\Downloads\mean_squared_errors.csv', index=False)

# Report the mean and standard deviation of the mean squared errors
mean_mse = np.mean(mse_list)
std_mse = np.std(mse_list)

print("Mean of Mean Squared Errors:", mean_mse)
print("Standard Deviation of Mean Squared Errors:", std_mse)
