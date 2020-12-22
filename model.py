from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, BatchNormalization
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

df = pd.read_csv("train.csv")
X = df.iloc[:, 1:]
y = df["label"]
print(y.shape, X.shape)

X = np.array(X).reshape(42000,28,28,1)
y = np.array(y).reshape(42000,1)
print(X.shape,y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=20)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = X_train.astype("float32")/255.0
X_test = X_test.astype("float32")/255.0

aug = ImageDataGenerator(width_shift_range=0.2, fill_mode="nearest", zoom_range=0.15)
# model = Sequential([
#         Conv2D(32, (5,5), padding="same", activation="relu", input_shape=(28,28,1)),
#         MaxPooling2D(2,2),
        
#         Conv2D(32, (3,3), padding="same", activation='relu'),
#         MaxPooling2D(2,2),
        
#         Flatten(),
#         Dense(64, activation='relu'),
#         Dropout(0.5),

#         Dense(64, activation='relu'),
#         Dropout(0.5),

#         Dense(10, activation='softmax')

#   		# Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(28,28,1)),
# 		# Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'),
# 		# MaxPooling2D((2, 2)),
		
# 		# Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu'),
# 		# Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu'),
# 		# MaxPooling2D(pool_size=(2,2), strides=(2,2)),
# 		# Dropout(0.1),

# 		# Conv2D(filters = 128, kernel_size = (3,3), padding = 'Same', activation ='relu'),
# 		# Conv2D(filters = 128, kernel_size = (3,3), padding = 'Same', activation ='relu'),
# 		# MaxPooling2D(pool_size=(2,2), strides=(2,2)),
# 		# Dropout(0.2),

# 		# Flatten(),
# 		# Dense(256, activation="relu"),
# 		# Dropout(0.2),
# 		# Dense(10, activation="softmax")
# ])

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='sigmoid')
])

print(y_train.shape, y_test.shape)
y_train = to_categorical(np.array(y_train).astype("int"))
y_test = to_categorical(np.array(y_test).astype("int"))
print(y_train.shape, y_test.shape)

eph = 10
BS = 128

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

history = model.fit(
         x = aug.flow(X_train, y_train, batch_size=BS),
         validation_data=(X_test, y_test), 
         epochs=eph, 
         steps_per_epoch=len(X_train)//BS)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(eph)

plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label="Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.show()

predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=-1)
print(y_pred, np.argmax(y_test, axis=-1))

model_json = model.to_json()
with open("model6.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model6.h5")

print("Saved model to disk.")