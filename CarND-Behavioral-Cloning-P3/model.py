import csv
import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint, Callback
import random
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten,Cropping2D
from keras.layers import Conv2D
from keras.layers import Lambda
from keras.regularizers import l2

lines =[]
# Load the data
with open('./data/driving_log1.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    print('loading udacity-data(clock direction)')    
del lines[0]

with open('./data_clock_1/data4/driving_log.csv') as csvfile1:
    reader1 = csv.reader(csvfile1)
    for line in reader1:
        lines.append(line)
    print('loading data4(clock direction)')
    
with open('./data_clock_1/data5/driving_log.csv') as csvfile2:
    reader2 = csv.reader(csvfile2)
    for line in reader2:
        lines.append(line)
    print('loading data5(encouter clock direction)')

with open('./data_track1/data6/driving_log.csv') as csvfile3:
    reader3 = csv.reader(csvfile3)
    for line in reader3:
        lines.append(line)
    print('loading data6 (recovery)')

with open('./data_track2/data8/driving_log1.csv') as csvfile4:
    reader4 = csv.reader(csvfile4)
    for line in reader4:
        lines.append(line)
    print('loading data8 (track 2)')
    
#adapt the data to get a balanced distrubution
lines_new = []
for line in lines:
    if (float(line[3])<0.05) and (float(line[3])>-0.05):
        if np.random.rand()< 0.4: #Keep 40% data
            lines_new.append(line)
    elif (float(line[3])<-0.05) and (float(line[3])>-0.2):
        if np.random.rand()< 0.75: #Keep 75% data
            lines_new.append(line)
    else:    
        lines_new.append(line)

#split the data to train_samples:validation_samples:test_samples = 8:1:1
train_samples, validation_and_test_samples = train_test_split(lines_new, test_size=0.2)
validation_samples, test_samples = train_test_split(validation_and_test_samples, test_size=0.5)
print('Number of whole samples: ', len(lines_new))
print('Number of train_samples: ', len(train_samples))
print('Number of validation_samples: ', len(validation_samples))
print('Number of test_samples: ', len(test_samples))

#define the functions
def createPreProcessingLayers():
    '''
     This funciton is used to bild a initial model, where the images would preprocessed.
     '''
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model
    
def random_darken(img):
    '''
     Function to distort the dataset through adding random shadow to the iamge
     '''
    #Convert original images from RGB colorspace to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #Generate new random brightness
    rand = random.uniform(0.5,1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    #Convert images back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img 
    
def generator(samples, batch_size=32,validation=False):
    '''
    This function gereate the data for training and validation.
    The validationsdata will darken the images randomly and add change some random noise on the angle.
    '''
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            correction = 0.2
            images = []
            angles = []
            for batch_sample in batch_samples:
                name_c = batch_sample[0].strip()
                name_l = batch_sample[1].strip()
                name_r = batch_sample[2].strip()
                if validation:
                    center_image = cv2.cvtColor(cv2.imread(name_c), cv2.COLOR_BGR2RGB)
                    left_image = cv2.cvtColor(cv2.imread(name_l), cv2.COLOR_BGR2RGB)    
                    right_image = cv2.cvtColor(cv2.imread(name_r), cv2.COLOR_BGR2RGB)
                    center_angle = float(batch_sample[3])
                    left_angle = center_angle + correction
                    right_angle = center_angle - correction                                     
                else:
                    center_image = random_darken(cv2.cvtColor(cv2.imread(name_c), cv2.COLOR_BGR2RGB))
                    left_image = random_darken(cv2.cvtColor(cv2.imread(name_l), cv2.COLOR_BGR2RGB))
                    right_image = random_darken(cv2.cvtColor(cv2.imread(name_r), cv2.COLOR_BGR2RGB))  
                    center_angle = float(batch_sample[3])*(1 + np.random.uniform(-0.06,0.06))
                    left_angle = center_angle + correction
                    right_angle = center_angle - correction
                images.extend((center_image, left_image, right_image))
                angles.extend((center_angle, left_angle, right_angle))
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

#Bild the NVIDIA training model
#l2 regularizer was here used to reduce overfit
model = createPreProcessingLayers()
model.add(Conv2D(24,5,5, subsample=(2,2), activation='relu',W_regularizer=l2(0.001)))
model.add(Conv2D(36,5,5, subsample=(2,2), activation='relu', W_regularizer=l2(0.001)))
model.add(Conv2D(48,5,5, subsample=(2,2), activation='relu', W_regularizer=l2(0.001)))
model.add(Conv2D(64,3,3, activation='relu', W_regularizer=l2(0.001)))
model.add(Conv2D(64,3,3, activation='relu', W_regularizer=l2(0.001)))
model.add(Flatten())
model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(Dense(1))

# compile and train the model using the generator function
model.compile(loss = 'mse', optimizer = 'adam')     
train_generator = generator(train_samples, batch_size=32, validation=False)
validation_generator = generator(validation_samples, batch_size=32, validation=True)
test_generator = generator(test_samples, batch_size=32, validation=True)
#define the checkpoint for each epoch
checkpoint = ModelCheckpoint('model{epoch:02d}.h5',monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), epochs=5, callbacks=[checkpoint])
#Evaluate the model with test_samples
print('Test Loss:', model.evaluate_generator(test_generator, 128))
#save the model and print the summary
model.save('model.h5') 
print("Model saved")
print(model.summary())