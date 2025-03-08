from tkinter import filedialog

def get_jpg_image_path():
    file_path = filedialog.askopenfilename(filetypes=[("JPG Files", "*.jpg")])  
    return file_path


image_path = get_jpg_image_path()

        ##    For Ending path system   ##
        ##    For MachineLearning path system   ##
import tensorflow as tf
import numpy as np
import tensorflow as keras
import os
import cv2  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory("", target_size=(150, 150), batch_size = 32, class_mode = 'binary')
test_dataset = train.flow_from_directory("https://github.com/DONCHAN70047/FireOrNotFireUPDATE/tree/main/static/ProjectFire/Testing", target_size=(150, 150), batch_size = 32, class_mode = 'binary')


test_dataset.class_indices

#Layer
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Binary classification (Fire / No Fire)
])


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


r = model.fit(train_dataset, epochs = 5, validation_data = test_dataset)

model.save('model/model.h5')

predictions = model.predict(test_dataset)
predictions = np.round(predictions)

#predictions

#plt.plot(r.history['loss'], label = 'loss')
#plt.plot(r.history['val_loss'], label = 'val_loss')
#plt.legend()
#plt.show()


#Predict using algorithm
def predictImage(filename) :
    img1 = image.load_img(filename, target_size=(150, 150))
    plt.imshow(img1)
    y = image.img_to_array(img1)
    x = np.expand_dims(y, axis = 0)
    val = model.predict(x)
    #print(val)
    if val==1 :
        #plt.xlabel("No Fire", fontsize = 30)
        #plt.show()
        print('No fire')
    elif val == 0 :
        #plt.xlabel("Fire", fontsize = 30)
        #plt.show()
        print('fire')

#Input
predictImage(image_path)

model = tf.keras.models.load_model('model/model.h5') #For save
        ##    For end MachineLearning path system   ##
