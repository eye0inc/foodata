import glob, os

import tensorflow as tf, pandas as pd
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.models import Model
from keras_preprocessing.image import ImageDataGenerator

import foodata_private_data as fpd

# model = VGG16()
# model = VGG16(include_top=False, input_tensor=new_input)

model = VGG16(include_top=False, input_shape=(224, 224, 3))
for layer in model.layers:
    layer.trainable = False
    
flattened = Flatten()(model.outputs)
dense1 = Dense(4096,activation="relu")(flattened)
output = Dense(1, activation='linear')(dense1)

model = Model(inputs=model.inputs, outputs=output)

print(model.summary())

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

print(fpd.datapath())

input_images_folder = fpd.datapath() + '/data'
print('input_images_folder = ' + input_images_folder)

food_images_and_nutrition_csv = fpd.datapath()+"dev/nutrition_info.csv"
print('food_images_and_nutrition_csv = ' + food_images_and_nutrition_csv)

food_images_and_nutrition_pandas = pd.read_csv(food_images_and_nutrition_csv, dtype=str)

datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)
train_generator=datagen.flow_from_dataframe(
    dataframe=food_images_and_nutrition_pandas,
    directory=input_images_folder,
    x_col="filename",
    y_col=["calories"],
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="raw",
    validate_filenames=False,
    target_size=(224,224))

valid_generator=datagen.flow_from_dataframe(
    dataframe=food_images_and_nutrition_pandas,
    directory=input_images_folder,
    x_col="filename",
    y_col=["calories"],
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="raw",
    validate_filenames=False,
    target_size=(224,224))

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)

# TODO
# * Get initial training done on any model
#   * Get input data into appropriate array of images (?)
#   * Get output data working with fitting to calorie values
# 
# * Divide data into training and validation

# dataset = tf.data.experimental.make_csv_dataset(
#     food_images_and_nutrition,
#     batch_size=5, # Artificially small to make examples easier to show.
#     label_name='calories',
#     na_value="?",
#     num_epochs=1,
#     ignore_errors=False,
#     select_columns=['filename'])

# print(dataset)

#model.fit(input_images, 

# f = []
# for (dirpath, dirnames, filenames) in walk('/'.join([fpd.datapath,'data']))]:
#     f.extend(filenames)
#     break

# print(str(f))                           
#/home/kevin/datahub/venv/src/foodata-private-data/foodata_private_data/data



#print(dir(fpd))
