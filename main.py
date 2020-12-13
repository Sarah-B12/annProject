import numpy as np
import pandas as pd
import tensorflow as tf  # Preferences - Python interpreter - + - tensorflow
import xlrd
import csv


def csv_from_excel():
    wb = xlrd.open_workbook('/Users/brownies/Desktop/BA/BIG_Project/ANN/VideosData.xlsx')
    sh = wb.sheet_by_name('Sheet1')
    csv_file = open('/Users/brownies/Desktop/BA/BIG_Project/ANN/VideosData.csv', 'w')
    wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

    for rownum in range(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    csv_file.close()

#call the function
#csv_from_excel()



# DATA PROCESSING
# Import the CSV file
dataset = pd.read_csv(r"/Users/brownies/Desktop/BA/BIG_Project/ANN/VideosData.csv")
X = dataset.iloc[:, 1:-1].values  # From the 2nd column to the end but not the last column
y = dataset.iloc[:, -1].values  # Only the last column


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# BUILD THE ANN
# Initializing the ANN
ann = tf.keras.models.Sequential()  # create the ANN

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=8, activation='relu'))  # We put 6 neurons in this hidden layer
                                                            # Relu: rectifier activation function

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=8, activation='relu'))

ann.add(tf.keras.layers.Dense(units=8, activation='relu'))


# Adding the output layer (contain what we want to predict)
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  # The output will be binary so we need only one neuron
                                            # Sigmoid activation function (give the probability that the output is 1)
                                            # If not binary output, we put 'softmax' instead of 'sigmoid'


# TRAIN THE ANN
# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # The adam optimizer will update the weight of the ANN through socastic
    # The loss is binary because the output is binary. If it's not we put 'categorical_crossentropy'

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 16, epochs = 100) # Epochs is the number of "iteration" for the ANN to learn

# MAKING THE PREDICTIONS AND EVALUATING THE MODEL
# Predicting the result of a single observation
# Use our ANN model to predict if the video with the following informations is violent or not:
print(ann.predict(sc.transform([[97.0880982819946, 0.137288326796803, 0.5, 375.10536085933, 102.316417905489, 0.0692592658259288, 0.446666666666667, 362.362828129655]])) > 0.5)
# The output is the PROBABILITY (because of the sigmoid function) that the video is violent or not
# So to not have a number at the output, we put the '> 0.5' (if the prob. is >0.5 then we consider the result is 1)


# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5) # Convert the probabilities to 0/1
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
    # Put together [the prediction and the real result]

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)  # See the accuracy of the ANN


'''
[ [Correct non violent    False positive]
  [False negative  Correct violent] ]
  
  [Prediction  True value]
'''