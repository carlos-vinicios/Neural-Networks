from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model
import numpy as np

def extract(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
    except FileNotFoundError:
        return []
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    return features[0]

def writeSVM(arq, values, classe):
    i = 1
    line=str(classe) + " "
    for v in values:
        line = line + str(i) + ":" + str(v) + " "
        i+=1
    arq.write(line+'\n')


model = VGG16(weights="imagenet", include_top=False, pooling="max") #carrega o modelo com base na imagenet, removendo as top layers e pedindo como resultado o pooling

normais = []
melanoma = []

for i in range(1, 438):
    path1 = "/home/carlos_vinicios/Documentos/Bases/Rois_2_classes/0_Normais/"
    path2 = "/home/carlos_vinicios/Documentos/Bases/Rois_2_classes/1_Melanomas/"
    if i < 10:
        name = "IMD00"+str(i)+".bmp"
    elif i < 100:
        name = "IMD0"+str(i)+".bmp"
    else:
        name = "IMD"+str(i)+".bmp"
    if i != 58 and i != 61 and i != 63 and i != 64 and i != 65 and i != 80 and i != 85 and i != 88 and i != 90 and i != 91 and i != 168 and i != 211 and i != 219 and i != 240 and i != 242 and i != 284 and i != 285 and i != 348 and i != 349 and i != 403 and i != 404 and i != 405 and i != 406 and i != 407 and i != 408 and i != 409 and i != 410 and i != 411 and i !=413 and i != 417 and i != 418 and i != 419 and i != 420 and i != 421 and i != 423 and i != 424 and i != 425 and i != 426 and i != 429 and i != 435 :
        normais.append(path1+name)
    else:
        melanoma.append(path2+name)

arq_svm = open('vgg16_features.libsvm', 'w')
for path in normais:
    f = extract(path, model)
    if len(f) > 0:
        writeSVM(arq_svm, f, 0)
for path in melanoma:
    f = extract(path, model)
    if len(f) > 0:
        writeSVM(arq_svm, f, 1)
arq_svm.close()
