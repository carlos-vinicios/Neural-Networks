from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
import numpy as np

#definição das imagens para treinamento
width=767
height=576 
train_dir= "../bases/split1/train"
validation_dir= "../bases/split1/valid"
test_dir = '../bases/split1/test'

#importa do modelo da vgg
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(width, height, 3))

for layer in vgg_conv.layers[:-4]:
    layer.trainable=False

model = models.Sequential()

model.add(vgg_conv)

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

#model.summary()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    rescale=1./255
)

train_batchsize=4
val_batchsize=2

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(width, height),
    batch_size=train_batchsize,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(width, height),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False,
)

test_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size=(width, height),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False,
)

model.compile(loss='categorical_crossentropy', 
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc'])

history = model.fit_generator(
    train_generator, 
    steps_per_epoch=train_generator.samples/train_generator.batch_size,
    epochs=16,
    validation_data = validation_generator,
    validation_steps=validation_generator.samples/validation_generator.batch_size,
    verbose=1)

print(history.history['acc'])
print(history.history['val_acc'])

predict = model.predict_generator(
    test_generator, 
    steps=test_generator.samples/test_generator.batch_size)

teste_img, test_labels = next(test_generator)

test_labels = test_labels[:,0]

predict = model.predict_generator(test_generator, steps=1, verbose=0)

print(metrics.accuracy_score(test_labels, np.round(predict[:,0])))

print(metrics.classification_report(test_labels, np.round(predict[:,0])))

confusion = metrics.confusion_matrix(test_labels, np.round(predict[:,0]))

print()
print(confusion)

i = int(input("Save?"))

if i == 1:
    model.save('teste_FT.h5')
