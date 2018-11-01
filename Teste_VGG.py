#faz a classificação de uma imagem segundo a softmax contida na imagenet

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions,preprocess_input
from keras.preprocessing import image
import numpy as np

model = VGG16(weights='imagenet') #uso da rede pré treinada, baseada na imagenet

img_path = 'elephant.jpg' #imagem a ser classificada
img = image.load_img(img_path, target_size=(224, 224)) #carrega a imagem no formato reconhecido pela rede
x = image.img_to_array(img) #converte a imagem para o Array NumPy
#x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2])) #remodela a imagem
x = np.expand_dims(x, axis=0) #mesmo processo de remodelagem da imagem, porém, utilizando função NumPy
x = preprocess_input(x) #realiza o pré processamento da imagem

features = model.predict(x) #realiza a predição do modelo

label = decode_predictions(features) #decodifica a predição para realização da classificação
classification = label[0][0] #pega o elemento de maior probabilidade

print('%s (%.2f%%)' % (classification[1], classification[2]*100))