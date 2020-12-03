# -*- coding: utf-8 -*-
"""
In this project we have a data set of numerical vectors. Each vector is the 784 element array from a flattened 28x28 greyscale image, so effectively this data has been pre-processed for us. In other situations we might be able to use the raw images to flip, rotate, colorshift, ReLU etc. but here we can just assume that was done to create this data set in the first place and move forward from there.

Our first step is to import. Since we'll be starting with a basic analysis and moving on from there we have quite a few imports. "Clean" code would condense these and unify our approach, but for the purposes of exploring our dataset and documenting what we find this will do.
"""

import os
import pandas as pd
import zipfile
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from random import randint
import tensorflow as tf

"""# Loading the Data
"""

pip install kaggle

"""First set your environment variables to your kaggle username and key in order to access the kaggle API, then we'll download the data"""

#Set the enviroment variables

os.environ['KAGGLE_USERNAME'] = "xxxxxxxxxxx"
os.environ['KAGGLE_KEY'] = "#####x##xx####xx##x"

#download the kaggle data
!kaggle competitions download -c dsti-s20-ann

"""Great, we've downloaded the test data!

Next let's unzip the training set, set them into a pandas dataframe, and have a quick look so we know what we're dealing with here
"""

train_zf = zipfile.ZipFile('/content/train_data.csv.zip') 

#this reads the csv and puts it in a pandas dataframe
train_df = pd.read_csv(train_zf.open('train_data.csv'), header=None)

train_df.head()

"""785 columns, the last of which is the correct category. First we'll take the last column as our Y and the rest as our X"""

#print(train_df.iloc[:,:-1])
train_x = train_df.iloc[:,:-1]
train_y=train_df.iloc[:,-1]

"""Next we randomly split the training set into train and validation so we can gauge how well our model is performing."""

x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=1) # 0.25 x 0.8 = 0.2

x_train.shape

y_train.shape

x_val.shape

y_val.shape

"""Here the validation sets are just one column. Later we'll one-hot encode these so they can be used by some of the more advanced models, but for now we'll leave them as-is and start with a a linear classification model.

# Building and Training Classifiers

**Linear Classifiers**

This is a linear classifier SGDClassifier:

Our data has 784 'dimensions', i.e. variables we use to classify it. If, for simplicity, we imagine these put onto just a 2D plane then we can image this classifier is just trying to draw straight lines through those points to separate each category. Then, when it makes a prediction, it just asks which side of the line the new point is on. The advantage to this approach is that it's mathematically simple, which also makes it fast to compute
"""

from sklearn.linear_model import SGDClassifier
model = SGDClassifier()
model.fit(x_train, y_train)

score = model.score(x_train, y_train)
print("Training score: ", score)

y_hat = model.predict(x_val)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_hat)
print(cm)

from sklearn.metrics import classification_report
cr = classification_report(y_val, y_hat)
print(cr)

"""On the validation data we get 70 percent accuracy - far better than a wild guess! However, we can imagine that drawing straight lines between our classifications does not

**Non-linear Classification**

What if we use a non-linear base function instead of a linear one? Support Vector Machines give us a way to use perceptrons alongside a useful learning procedure we know well; stochastic gradient descent. With this more advanced analysis we hope to outperform the simple linear classifier.
"""

# I think I'll use the SGDClassifier 
from sklearn.svm import SVC # "Support vector classifier"
model = SVC(kernel='linear', C=1E10)
model.fit(x_train, y_train)

score = model.score(x_train, y_train)
print("Training score: ", score)

y_hat = model.predict(x_val)
cm = confusion_matrix(y_val, y_hat)
print(cm)
cr = classification_report(y_val, y_hat)
print(cr)

"""This shows a slight improvement, but we'll need to do better still to win any competitions!

# Neural Networks

**A very basic NN**

Let's move on to a Neural Network approach. This will give us layers (in the first case, just one layer) of hidden non-linear units. Each unit has a weight and back propogation allows us to adjust as the weights as the NN 'learns'. 

For these kinds of methods we'll need to change our outputs y_train and y_val to one-hot encoded outputs first:
"""

categories = 10
y_train = keras.utils.to_categorical(y_train, categories)
y_val = keras.utils.to_categorical(y_val, categories)

"""Let's build a really simple network - just one hidden layer. We don't expect this to win any awards but it's a good place to start and to see if what we've done so far can work in a neural network.

This model is Sequential(), meaning every layer has just one input and output. Again, starting nice and simple.
"""

image_size = 784
model = Sequential()

model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=categories, activation='softmax'))
model.summary()



"""Now that we've build the model we can pass our train and validation data and see how it performs.

We'll start with some typical parameters: our optimizer is again Stochastic Gradient Descent, our loss function is the popular 'categorical cross entropy', and for batch size we'll take a guess at 128. Epoch is the number of forward and backward passes the data takes through the network, so we'll take another guess and start with 50.
"""

model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=50, verbose=False, validation_split=.1)
loss, accuracy  = model.evaluate(x_val, y_val, verbose=False)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')

"""Training and validation scores are very close to each other which is good. Both appear to still be climbing at the 50th epoch, which means we want to give our model more time to keep learning. We expect to see the curve flatten off when we've squeezed all the accuracy we can from the model we've built. Let's try 300 epochs instead, and leave the other parameters the same so we can compare apples to apples."""

model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=300, verbose=False, validation_split=.1)
loss, accuracy  = model.evaluate(x_val, y_val, verbose=False)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')

"""We can see the validation score flattens at about 150 to 200 epochs. 

This gives us a model that's about 8 times better than just guessing randomly, and all from a single hidden layer! But as always the question remains - can we do better?

# Fine-tuning our parameters

Neural networks are, by design, choosing weights for reasons we can't directly see or understand. We know this allows them to perform better but unfortunately it also means we can only tune them with trial and error - we can never 'prove' and optimal configuration.

Here we'll try a range for our parameters in order to explore the space and find the best parameter values we can, which is all we can hope to do.

First we build a create_dense function and an evaluate function. The create_dense function allows us to add a layer in the same way every time so that we know these models are comprable. The evaluate function gives us a summary of each model, graphs its training and validation for us and then prints a summary of its performance.
"""

def create_dense(layer_sizes):
    model = Sequential()
    model.add(Dense(layer_sizes[0], activation='sigmoid', input_shape=(image_size,)))

    for s in layer_sizes[1:]:
        model.add(Dense(units = s, activation = 'sigmoid'))

    model.add(Dense(units=categories, activation='softmax'))

    return model

def evaluate(model, batch_size=128, epochs=200):
    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=.1, verbose=False)
    loss, accuracy  = model.evaluate(x_val, y_val, verbose=False)
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()

    print()
    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')

"""Now that we have our functions built we can try a range of depths and see how they perform."""

for layers in range(1, 5):
    model = create_dense([32] * layers)
    evaluate(model)

"""Two hidden layers looked promising and may be worth running for more than 200 epochs, but after that we don't see promising results despite a more complicated model. This is almost certainly due to overfitting.

If we return to our mental picture of the data in 2D, we can picture the increase in complexity as the ability to draw a more 'curvy' line around the datapoints. If we make it *really* complex, the line gets so curvy it only works for data we already have - it hugs every edge case so closely that any new ones don't fit though perhaps they should.

So for now let's see how well we can do with two hidden layers and 500 epochs.
"""

model = create_dense([32, 32])
evaluate(model, epochs=500)

"""Two hidden layers with 32 nodes each and 500 epochs gives us an accuracy of 76%, which is not really an improvement despite the complexity and length of compute time.

There is another parameter, however, that we haven't considered yet: the number of nodes.

Each layer compresses our image to pull out the useful traits. We need to extract the import information from the noise, but too much compression and we lose the import information as well.

Our single hidden layer NN worked best so far, so let's try running it with a range of nodes to see how that effects the outcome.
"""

for nodes in [32, 64, 128, 256, 512, 1024, 2048]:
    model = create_dense([nodes])
    evaluate(model)

"""From this little experiment it looks like 1024 nodes is ideal for our dataset.

Now that we have a better idea of how many nodes to use let's try creating a slightly deeper network again and see if we can't break that 80% accuracy ceiling.

We'll rebuild it from scratch since we just did a lot of experimenting on the "model" variable.
"""

model = Sequential()

model.add(Dense(units=1024, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=1024, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=categories, activation='softmax'))

model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=200, verbose=False, validation_split=.1)
loss, accuracy  = model.evaluate(x_val, y_val, verbose=False)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')

model.predict()

"""Again we find ourselves stuck at the 80% barrier, and so we can decide that perhaps that is all we can get from this tool and we need to move on to a more advanced method, the Convolutional Neural Network.

# Convolutional Neural Networks (CNN)

Let's convert these dataframes to numpy arrays so we can use .reshape(). There's certainly a way to do this in pandas but I'm not interested in figuring that out right now.

reassign the data just in case:
"""

x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=1)

#x_train = x_train.to_numpy()
np_x_train = x_train.to_numpy()
np_y_train = y_train.to_numpy()

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets
np_x_train = np_x_train.reshape(np_x_train.shape[0], 1, img_rows, img_cols)
np_y_train = np_y_train.reshape(np_y_train.shape[0], 1, img_rows, img_cols)
np_x_train = x_train.astype('float32')
np_y_train = np_y_train.astype('float32')

from keras.layers import Dense, Conv2D, MaxPool2D , Flatten

model = Sequential()
model.add(Conv2D(input_shape=(28,28,1), filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

from keras.optimizers import Adam
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=2, activation="softmax"))

model.build()
model.summary()

"""VGG16 implementation:"""

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
history = model.fit(x_train, y_train, batch_size=128, epochs=300, verbose=False, validation_split=.1)
#hist = model.fit(x_train, y_train, steps_per_epoch=100, validation_steps=10,epochs=100,callbacks=[checkpoint,early])

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg

from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

base_model = VGG16(include_top=False, weights='imagenet', input_shape = (224,224,3))
base_model.summary()