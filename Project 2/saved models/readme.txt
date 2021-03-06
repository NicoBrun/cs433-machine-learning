##Here are some tried models:

# Model 1:

test_size = 0.2
batch_size = 64
epoch = 200
steps_per_epoch = len(x_train)*2 // batch_size

ImageDataGenerator(
        rotation_range=180,
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='nearest')


model = Sequential()
model.add(Conv2D(128, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(batch_size))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

results: 0.87
time to create model: 14h



#model 2:

test_size = 0.2
batch_size = 64
epoch = 200
steps_per_epoch = len(x_train)*2 // batch_size

ImageDataGenerator(
        rotation_range=180,
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='nearest')


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(64,(3, 3)))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(batch_size))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

results: 0.85484
time to create model: 5h


#model 3:

comme model 1
batch_size =  128

result:0.86376

#model 4:

patch_size = 8
test_size = 0.2
batch_size = 64
epoch = 200
steps_per_epoch = len(x_train)*2 // batch_size

ImageDataGenerator(
        rotation_range=180,
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='nearest')

model = Sequential()
model.add(Conv2D(128, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(batch_size))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

results:
time to create model: 5h

#model 5 unet (discard it)

#model 6
comme model 1 mais sans image augmentation
Test loss: 1.91540325628
Test accuracy: 0.826080000038

#model 8 
Test loss: 0.356196646347
Test accuracy: 0.83832

#model 11
Test loss: 0.339187149124
Test accuracy: 0.845840000038