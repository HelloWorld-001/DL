import pandas as pd

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.optimizers import RMSprop
import tensorflow as tf
from keras.layers import BatchNormalizationV1
from keras.layers import Dropout

training_data = pd.read_csv("/content/sample_data/california_housing_train.csv")
X = training_data.iloc[:,0:8]
cols = training_data.columns
Y = training_data[cols[-1]]

num_hidden_layers = 3
def create_functional_regression_model():
    input_to_nw = Input(shape=(8,))
    x = Dense(units=8) (input_to_nw)
    x = BatchNormalizationV1() (x)
    for _ in range(num_hidden_layers-1):
        x = Dense(units=8) (x)
        x = BatchNormalizationV1() (x)
        X = Dropout(rate=0.5) (x)
        nw_out = Dense(units=1, activation="relu") (x)
    return Model(inputs=input_to_nw,outputs=nw_out)

regression_nw = create_functional_regression_model()
regression_nw.summary()

tf.keras.utils.plot_model(regression_nw)
opt = RMSprop(learning_rate=0.1)
regression_nw.compile(optimizer=opt,loss=tf.keras.losses.Huber())
cv_data = pd.read_csv("/content/sample_data/california_housing_test.csv")

prine("efnoiwsclke3frn")