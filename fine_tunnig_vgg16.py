from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from sklearn import metrics
import numpy as np
#from keras.metrics import categorical_crossentropy
#from keras.layers.normalization import BatchNormalization
#from keras.layers.convolutional import *

train_path = "train_base/train"
test_path = "train_base/test"
valid_path = "train_base/valid"

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(768, 574), classes=["melanoma", "normais"], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(768, 574), classes=["melanoma", "normais"], batch_size=4)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(768, 574), classes=["melanoma", "normais"], batch_size=10)

model_vgg16 = VGG16()

#necessario criar um novo modelo sequencial para após aproveitar as camadas da VGG16, adicionando uma a uma
model = Sequential()
for layer in model_vgg16.layers:
    model.add(layer)

model.layers.pop() # retira a última dense layer contendo as 1000 classes

for layer in model.layers: #congela as camadas iniciais do modelo para que não aprendam novas caracteristicas
    layer.trainable = False

model.add(Dense(2, activation='softmax')) #adicionamos uma nova dense layer como softmax, para classificação entre melanoma e não melanoma

model.compile(Adam(lr=.0001), loss="categorical_crossentropy", metrics=['accuracy', 'recall'])

model.fit_generator(train_batches, steps_per_epoch=4, validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)

predict = model.predict_generator(test_batches, steps=1, verbose=0)

confusion = metrics.confusion_matrix(test_labels, np.round(predict[:,0]))

print(confusion)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# salvando os pesos
model.save_weights("model.h5")
print("Saved model to disk")