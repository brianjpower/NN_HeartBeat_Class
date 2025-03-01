import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load data
train_data = pd.read_csv("C:/Users/brian/PycharmProjects/NN_HeartBeat_Class/data_mitbih_train.csv", header=None).values
test_data = pd.read_csv("C:/Users/brian/PycharmProjects/NN_HeartBeat_Class/data_mitbih_test.csv", header=None).values

# Separate features and labels
x_train = train_data[:, :-1]
y_train = to_categorical(train_data[:, -1])
x_test = test_data[:, :-1]
y_test = to_categorical(test_data[:, -1])

# Get input shape
V = x_train.shape[1]

# Define the model
model = Sequential([
    Dense(50, activation="relu", input_shape=(V,)),
    Dense(20, activation="relu"),
    Dense(y_train.shape[1], activation="softmax")
])

# Compile model
model.compile(
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    optimizer=RMSprop()
)

# Define batch size
N = x_train.shape[0]
batch_size = round(N * 0.01)

# Fit model
fit = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=100,
    batch_size=batch_size,
    verbose=1,
    callbacks=[EarlyStopping(monitor="val_accuracy", patience=5)]
)

# Smooth line function
def smooth_line(y):
    x = np.arange(len(y))
    from scipy.interpolate import make_interp_spline
    spline = make_interp_spline(x, y)
    return spline(x)

# Plot performance
cols = ["black", "dodgerblue"]
accuracy = np.array([fit.history['accuracy'], fit.history['val_accuracy']]).T
plt.plot(accuracy, marker='o', linestyle='None', alpha=0.3)
plt.plot(smooth_line(accuracy[:, 0]), color=cols[0], linewidth=2)
plt.plot(smooth_line(accuracy[:, 1]), color=cols[1], linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
#plt.yscale("log")
plt.legend(["Training", "Test"])
plt.show()

# Classification performance
y_raw = np.argmax(y_test, axis=1)
class_hat = np.argmax(model.predict(x_test), axis=1)
conf_matrix = tf.math.confusion_matrix(y_raw, class_hat).numpy()
print(conf_matrix)

# Compute accuracy
accuracy_score = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
print("Accuracy:", accuracy_score)

# Last validation accuracy
tail_val_accuracy = fit.history['val_accuracy'][-1]
print("Last validation accuracy:", tail_val_accuracy)

# Actual epochs
actual_epochs = len(fit.history['loss'])
print("Actual epochs:", actual_epochs)

# Model with Dropout
model_dropout = Sequential([
    Dense(50, activation="relu", input_shape=(V,)),
    Dropout(0.5),
    Dense(20, activation="relu"),
    Dropout(0.4),
    Dense(y_train.shape[1], activation="softmax")
])

model_dropout.compile(
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    optimizer=RMSprop()
)

fit_dropout = model_dropout.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=100,
    batch_size=batch_size,
    verbose=1,
    callbacks=[
        EarlyStopping(monitor="val_accuracy", patience=10),
        ReduceLROnPlateau(monitor="loss", patience=10, factor=0.1)
    ]
)

# Plot performance with Dropout
accuracy_dropout = np.array([fit_dropout.history['accuracy'], fit_dropout.history['val_accuracy']]).T
plt.plot(accuracy_dropout, marker='o', linestyle='None', alpha=0.3)
plt.plot(smooth_line(accuracy_dropout[:, 0]), color=cols[0], linewidth=2)
plt.plot(smooth_line(accuracy_dropout[:, 1]), color=cols[1], linewidth=2)
plt.ylim(0.8, 1)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
#plt.yscale("log")
plt.legend(["Training", "Test"])
plt.show()
