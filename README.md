# linear_regression
linear_regression model
Given: In file train_X_lr.csv, we have the data of customers(Time spent on website, Duration of Membership, Time Spent on App and session duration along with the yearly amount spent by them)
Aim: to make a linear regression model to calculate yearly amount spent by the users based on the other factors and give it to the business team to make business strategies.

Files:

train.py: This file contains the python code to use the data in train_X_lr.csv and make a hypothesis function. This code produces WEIGHTS_FILE.csv which contains the optimized weights i.e. coefficient values which are produced using the gradient descent function.

predict.py: This file is used for prediction. train_Y_lr.csv contains tha customer's data of which we need to compute the yearly amount spent. This code generates predicted_test_Y-lr.csv containing the predicted yearly amount.

validate.py: This file contains code to verify the predicted values in the predicted_test_Y_lr.csv file.

# HELPER FUNCTIONS

To read a csv file and convert into numpy array, we can use genfromtxt of the numpy package.
For Example:
```
train_data = np.genfromtxt(train_X_file_path, dtype=np.float64, delimiter=',', skip_header=1)
```
Can use the python csv module for writing data to csv files.
Refer to https://docs.python.org/2/library/csv.html.
For Example:
```
with open('sample_data.csv', 'w') as csv_file:
	writer = csv.writer(csv_file)
    writer.writerows(data)
```
