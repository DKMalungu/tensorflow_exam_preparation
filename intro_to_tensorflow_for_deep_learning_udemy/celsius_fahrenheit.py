import pandas
import logging

"""TensorFlow Logging only error  messages when running project"""
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""Dataset of data of temperature in fahrenheit and corresponding degree celsius """

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38, 49, 53], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100, 120, 127], dtype=float)

for i, c in enumerate(celsius_q):
    print(f"{c} degrees celsius = {fahrenheit_a[i]} degree fahrenheit")

"""Creating the model"""
temp = tf.keras.layers.Dense(input_shape=[1], units=1)
# temp_2 = tf.keras.layers.Dense(units=1)
# temp_3 = tf.keras.layers.Dense(units=1)

"""Assemble Layers into the model"""
temp_model = tf.keras.Sequential([temp])

"""Compile the model with loss and optimizer function"""
temp_model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

"""Training the model"""
history = temp_model.fit(x=celsius_q, y=fahrenheit_a, epochs=3500, verbose=True)
print(type(history))
print('\n Finished training the model \n')

"""Displaying training statistics report"""
plt.xlabel("Epoch Number")
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

"""Using the  model ro predict values"""
print(f"\n100 degree celsius is equal to: {temp_model.predict([100])} fahrenheits\n")

"""Displaying Layer weghits"""
print(f"These are thr layer variables: {temp.get_weights()}")
