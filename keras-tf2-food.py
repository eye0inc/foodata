import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from keras.callbacks.callbacks import History
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img

import pandas as pd 
import argparse
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", type=str, required=True,
# 	help="path to input dataset of house images")
ap.add_argument("-E", "--epochs", type=int, required=False, default=200,
	help="number of epoches to train for")
args = vars(ap.parse_args())

INPUT_SHAPE = (224, 224, 3)
INPUT_SIZE = INPUT_SHAPE[:2]

print("INPUT_SHAPE = " + str(INPUT_SHAPE) + "INPUT_SIZE = " + str(INPUT_SIZE))
# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

def simple_model(input_shape):
    
    ret = models.Sequential()
    ret.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    ret.add(layers.MaxPooling2D((2, 2)))
    ret.add(layers.Conv2D(64, (3, 3), activation='relu'))
    ret.add(layers.MaxPooling2D((2, 2)))
    ret.add(layers.Conv2D(64, (3, 3), activation='relu'))

    ret.add(layers.Flatten())
    ret.add(layers.Dense(64, activation='relu'))
    ret.add(layers.Dense(1))
    return ret

def mobilenetv2_model(input_shape):

    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                               include_top=False,
                                               weights='imagenet')

def vgg16_model(input_shape):

    base_model = tf.keras.applications.VGG16(input_shape=input_shape,
                                               include_top=False,
                                               weights='imagenet')
    base_model.trainable = False

    regression_model = models.Sequential()
    regression_model.add(layers.Flatten())
    regression_model.add(layers.Dense(64, activation='relu'))
    regression_model.add(layers.Dense(1))

    return tf.keras.Sequential([base_model, regression_model])

food_calorie_training_data = pd.read_csv("training_data/private/dev/nutrition_info.csv")
food_calorie_validation_data = pd.read_csv("training_data/private/dev/nutrition_info-validate.csv") 

print(food_calorie_training_data.head())
print(food_calorie_validation_data.head())

training_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

food_training_generator = training_datagen.flow_from_dataframe(
    dataframe=food_calorie_training_data,
    directory='training_data/private/data/', 
    target_size=INPUT_SIZE,
    batch_size=32,
    class_mode='raw',
    x_col='filename',
    y_col='calories')

food_validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=food_calorie_validation_data,
    directory='training_data/private/data/', 
    target_size=INPUT_SIZE,
    batch_size=32,
    class_mode='raw',
    x_col='filename',
    y_col='calories')

base_model = simple_model(INPUT_SHAPE)
base_model = mobilenetv2_model(INPUT_SHAPE)
base_model = vgg16_model(INPUT_SHAPE)

regression_model = models.Sequential()
regression_model.add(layers.Flatten())
regression_model.add(layers.Dense(64, activation='relu'))
regression_model.add(layers.Dense(1))

model = tf.keras.Sequential([base_model, regression_model])

model.summary()



# with tf.Session() as sess:
#   print(sess.list_devices())

# base_learning_rate = 0.0001

model.compile(optimizer='sgd',
              loss='mae',
              metrics=['mse', 'mae', 'mape'])


model.fit_generator(food_training_generator, steps_per_epoch=1, epochs=args['epochs'],
                    validation_data=food_validation_generator)

training_results = model.evaluate(food_validation_generator, verbose=2)

print(str(training_results))
