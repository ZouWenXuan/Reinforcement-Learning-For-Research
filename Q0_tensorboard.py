# =============================================================================
# Q0: How to use tensorboard ?
# =============================================================================

#%% An example of automatic processing

import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' 


# Load and normalize MNIST data
mnist_data = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist_data.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0


# Define the model
model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(28, 28)),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(10, activation='softmax')
])


# Compile the model
model.compile(
optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

tf_callback = tf.keras.callbacks.TensorBoard(log_dir="./tensorboard_save/log_train")
model.fit(X_train, y_train, epochs=5, callbacks=[tf_callback])


#%% An example of manual processing

import numpy as np
# Specify a directory for logging data
logdir = "./tensorboard_save/log_sin"

# Create a file writer to write data to our logdir
file_writer = tf.summary.create_file_writer(logdir)

# Loop from 0 to 199 and get the sine value of each number
for i in range(200):
    with file_writer.as_default():
        tf.summary.scalar('sine wave', np.math.sin(i), step=i)