from google.colab import files

import os
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

!pwd
files.upload()
!mkdir ~/.kaggle
!mv /content/kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json
!pip install kaggle
!kaggle datasets download -d ananddd/gujarati-ocr-typed-gujarati-characters
!unzip /content/gujarati-ocr-typed-gujarati-characters.zip

single_dir_no_images = len(os.listdir("/content/Gujarati OCR/Gujarati/Train/A"))
img_np_array = plt.imread(os.path.join("/content/Gujarati OCR/Gujarati/Train/A",os.listdir("/content/Gujarati OCR/Gujarati/Train/A")[0]))
plt.imshow(img_np_array)

def create_custom_cnn():
    input_to_cnn = Input(shape=img_np_array.shape)
    first_conv_out = Conv2D(filters=25,kernel_size=(5,5),activation="relu") (input_to_cnn)
    second_conv_out = Conv2D(filters=50,kernel_size=(5,5),activation="relu") (first_conv_out)
    third_conv_out = Conv2D(filters=100,kernel_size=(5,5),activation="relu") (second_conv_out)
    maxpool_out = MaxPooling2D(pool_size=(4,4),strides=(4,4)) (third_conv_out)
    flattened_out = Flatten() (maxpool_out)
    nn_out = Dense(units=385,activation="softmax") (flattened_out)
    return Model(inputs=input_to_cnn,outputs=nn_out)

custom_cnn = create_custom_cnn()
custom_cnn.summary()

datagen = ImageDataGenerator(rescale=1/255.0)
training_datagen = datagen.flow_from_directory(directory="/content/Gujarati OCR/Gujarati/Train",target_size=(32,32),classes=os.listdir("/content/Gujarati OCR/Gujarati/Train"),batch_size=512)
testing_datagen = datagen.flow_from_directory(directory="/content/Gujarati OCR/Gujarati/Test",target_size=(32,32),classes=os.listdir("/content/Gujarati OCR/Gujarati/Train"))

custom_cnn.compile(loss="categorical_crossentropy",metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
X_train_mb,Y_train_mb = training_datagen.__next__()
X_test_mb,Y_test_mb = testing_datagen.__next__()
custom_cnn.fit(training_datagen,epochs=10,validation_data=testing_datagen)