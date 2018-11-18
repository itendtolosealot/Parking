import tensorflow as tf
import numpy as np
import sklearn.model_selection as sk
import csv

feature_size = 33
labels = []
features = []
def one_hot_encoding(value, max_size):
    b = np.zeros((max_size,1))
    b[value,1] = 1
    return b

max_rows = len(open('/home/ashvin/smart_parking/filtered.csv').readlines())
print("MaxRows: " + str(max_rows))


with open('/home/ashvin/smart_parking/filtered.csv') as csvfile:
    parked_car_data = csv.DictReader(csvfile)
    features = np.zeros([max_rows, feature_size])
    labels = np.zeros([max_rows,1])
    count = 0
    for row in parked_car_data:
        labels[count] = (float(row['Total_Vehicle_Count'])*1.00)/(float(row['Parking_Spaces']) *1.0)
        time = np.array([row['Time']])
        features[count, int(row['Area_ID'])] =1
        features[count, 25+int(row['Day_of_Week'])] = 1
        features[count, 32] = float(row['Time'])
        count = count + 1

print("Features Shape: " + str(features.shape))
print("Labels Shape: " + str(labels.shape))
X_train, X_test, Y_train, Y_test = sk.train_test_split(features,labels,test_size=0.1, random_state = 5)



print(X_train)
print(Y_train)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(8192, activation=tf.nn.softmax),
  tf.keras.layers.Dense(512, activation=tf.nn.softmax),
  tf.keras.layers.Dense(1, activation=tf.nn.softmax)
])

adam = tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999)

model.compile(optimizer=adam,
        loss='cosine_proximity',
              metrics=['mse'])

model.fit(X_train, Y_train, batch_size= max_rows, epochs=100)
model.evaluate(X_test, Y_test)