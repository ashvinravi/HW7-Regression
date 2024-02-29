from regression import LogisticRegressor

"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
# (you will probably need to import more things here)

def test_prediction():

	log_model = LogisticRegressor(num_feats=3)

	# Does prediction work when your weights are zero? 
	# Define sample input features
	X = np.array([[1, 2, 3, 4],
				  [5, 6, 7, 8],
				  [9, 10, 11, 12]])

    # Set weights for testing (e.g., all zeros for simplicity)
	log_model.W = np.zeros(4)
    # Expected output for the given weights and features
	# If your weights are zero, you should expect 0.5 as the probability - equal chance of 1 or 0. 
	expected_y = np.array([0.5, 0.5, 0.5]) 
	predicted_y= log_model.make_prediction(X)
	assert np.array_equal(expected_y, predicted_y)

def test_loss_function():
	log_model = LogisticRegressor(num_feats=1)
	y_true = np.array([1, 0])
	y_pred = np.array([0.95, 0.05])
	# Loss = -1/1(log(0.95)) + (1 - 1) * log(0.05) = 0.0513 + 0 = 0.0513
    # Expected output for the given true labels and predicted probabilities
	expected_loss = 0.0513  
	predicted_loss = round(log_model.loss_function(y_true, y_pred), 4)
	assert ( expected_loss == predicted_loss )

def test_gradient():
	import numpy as np

	# Define input features (X) and true labels (y_true)
	X = np.array([[1.5, 2.5, 4],
				[1, 2, 5.6],
				[1, 0, 7.94],
				[3, 0, 3.33]])
	
	y_true = np.array([1, 0, 0, 1])

	# iterations should be only 1 since gradient is calculated iteratively, same with batch 
	log_model = LogisticRegressor(num_feats=2, max_iter=1, batch_size=1)

	# Manually calculate the gradient by making prediction, calculating error, and then manual gradient. 
	y_pred = log_model.make_prediction(X)
	error = y_pred - y_true
	manual_gradient = np.dot(X.T, error) / len(y_true)
	# Calculate the gradient using the implemented method
	predicted_gradient = log_model.calculate_gradient(y_true, X)

	# compare if gradient calculations are similar to each other 
	assert np.allclose(manual_gradient, predicted_gradient)

def test_training():
	X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS',

        ],
        split_percent=0.8,
        split_seed=42
    )
    # Check if weights remain the same pre and post training your model. 
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.1, tol=0.01, max_iter=10000, batch_size=10)
	# store previous weights - should be random 
	previous_weights = log_model.W
	log_model.train_model(X_train, y_train, X_val, y_val)
	# store current weights 
	updated_weights = log_model.W

	assert ( list(previous_weights) != list(updated_weights))