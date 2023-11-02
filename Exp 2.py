import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense
import tensorflow as tf
from keras.optimizers import RMSprop

training_data = pd.read_csv("/content/sample_data/mnist_train_small.csv",header=None)

column_names = ["Digit"]
for i in range(1,training_data.shape[1]):
    column_names.append("Pixel_"+str(i))

training_data.columns = column_names

def display_image(single_img_pixels):
    single_img_pixels = np.array(single_img_pixels)
    single_img_pixels = single_img_pixels.reshape(28,28)
    plt.imshow(single_img_pixels,cmap="gray")

display_image(training_data.iloc[1,1:])

validation_data = pd.read_csv("/content/sample_data/mnist_test.csv",header=None)
validation_data.columns = column_names
display_image(validation_data.iloc[0,1:])

X = training_data.iloc[:,1:]
Y = training_data.iloc[:,0]
X_cv = validation_data.iloc[:,1:]
Y_cv = validation_data.iloc[:,0]
def create_functional_cls_nw():
    input_to_nw = Input(shape=(X.shape[1],))
    hidden_layer_out = Dense(units=X.shape[1],activation="relu") (input_to_nw)
    nw_output = Dense(units=10,activation="softmax") (hidden_layer_out)
    return Model(inputs=input_to_nw,outputs=nw_output)

Y_ohe = np.eye(10,10)[Y]
Y_cv_ohe = np.eye(10,10)[Y_cv]

X = X/255.0
X_cv = X_cv/255.0

multi_cls_nw = create_functional_cls_nw()
multi_cls_nw.summary()

opt = RMSprop(learning_rate=0.01)
multi_cls_nw.compile(optimizer=opt,loss="categorical_crossentropy",metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
multi_cls_nw.fit(x=X,y=Y_ohe,batch_size=200,epochs=20,validation_data=(X_cv,Y_cv_ohe))