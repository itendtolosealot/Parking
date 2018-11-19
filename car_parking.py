import tensorflow as tf
import numpy as np
import math
import sklearn.model_selection as sk
import csv
import sys
import time
feature_size = 35
congestion_levels = 2.0
pred_samples = 300
init_learning_rate = 1.0
epochs = 25
test_percentage = 0.19
labels = []
features = []
layer_size = 16
max_neurons = 2048

neurons_in_layer = np.zeros(layer_size)
file_to_read = "/home/ashvin/smart_parking/filtered.csv"


if(test_percentage > 0.2):
    print("Error! The available data is partitioned twice for CV and testing. Proposed value will use up " + str(int(2*test_percentage*100)) + "% of training data.Use a value less than 20%" )
    sys.exit(1)

def custom_accuracy(x,y):
    return tf.keras.metrics.top_k_categorical_accuracy(x,y,k=1)

for i in range(0,layer_size):
    neurons_in_layer[i] = max(int(max_neurons/pow(2,i)),int(congestion_levels))

max_rows = len(open(file_to_read).readlines())
print("MaxRows: " + str(max_rows))


with open(file_to_read) as csvfile:
    parked_car_data = csv.DictReader(csvfile)
    features = np.zeros([max_rows, feature_size])
    labels = np.zeros([max_rows, int(congestion_levels)])
    count = 0
    for row in parked_car_data:
        cong_level = int(congestion_levels*(float(row['Total_Vehicle_Count'])*1.00)/(float(row['Parking_Spaces']) *1.0))
        if(cong_level == int(congestion_levels)):
            print(" Error! Veh Count: " + row['Total_Vehicle_Count'] + " Parking Spaces: " + row['Parking_Spaces'] + " Row: " + str(count))
            sys.exit(1)
        labels[count,cong_level] = 1
        features[count, int(row['Area_ID'])] =1
        features[count, 25+int(row['Day_of_Week'])] = math.sin(float(row['Day_of_Week'])*2*math.pi/7.0)
        features[count, 26+int(row['Day_of_Week'])] = math.cos(float(row['Day_of_Week'])*2*math.pi/7.0)
        features[count, 33] = math.sin(float(row['Time'])*2*math.pi)
        features[count, 34] = math.cos(float(row['Time'])*2*math.pi)
        count = count + 1

print("Features Shape: " + str(features.shape))
print("Labels Shape: " + str(labels.shape))
X_train, X_test, Y_train, Y_test = sk.train_test_split(features,labels,test_size=test_percentage, random_state = 50)

model = tf.keras.models.Sequential()
constraint = tf.keras.constraints.UnitNorm(axis=0)
weight_initializer=tf.keras.initializers.RandomNormal(mean=1.00, stddev=0.05,seed=1)
adam = tf.keras.optimizers.Adadelta(lr=init_learning_rate, decay=0.05)


for i in range(0,layer_size):
    if(i==0):
        model.add(tf.keras.layers.Dense(neurons_in_layer[i], input_shape = (feature_size,),kernel_initializer=weight_initializer, kernel_constraint=constraint))
    else:
        model.add(tf.keras.layers.Dense(neurons_in_layer[i],kernel_constraint=constraint, kernel_initializer=weight_initializer))
    model.add(tf.keras.layers.Activation('softmax'))

model.add(tf.keras.layers.Dense(int(congestion_levels)))
model.add(tf.keras.layers.Activation('softmax'))




model.compile(optimizer=adam,
        loss='categorical_crossentropy',
              metrics=[custom_accuracy])

start = time.time()
model.fit(X_train, Y_train, batch_size= int(max_rows*(1-2*test_percentage)), validation_split=test_percentage, verbose=1,epochs=epochs)


end = time.time()
loss = model.evaluate(X_test, Y_test)
print("Test Loss: " + "{:2.3f}".format(loss[0]) + " Accuracy: " + "{:2.3f}".format(loss[1]))


rand_sample = np.random.randint(int(max_rows*test_percentage)-1,size=pred_samples)
X_cv =X_test[rand_sample,:]
y_cv = Y_test[rand_sample,:]

y_pred=model.predict(X_cv, verbose=1)

cong_level_actual = np.argmax(y_cv,axis=1)
cong_level_pred = np.argmax(y_pred,axis=1)

error_norm = (1.0/pred_samples)*np.linalg.norm(cong_level_actual-cong_level_pred, ord=2)



print("Error Norm: " + "{:2.3f}".format(error_norm))

gFlops = 0
for i in range(0,layer_size):
    if(i==0):
        gFlops = gFlops + 2*feature_size*max_neurons
    else:
        gFlops = gFlops + 2*max_neurons*max_neurons

gFlops = gFlops*(1-2*test_percentage)*max_rows*1.0*epochs/((end-start)*1e9)

print("GigaFlops: " + "{:2.3f}".format(gFlops))