from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Model
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from keras.optimizers import SGD
from sklearn import metrics
from shutil import copy
import numpy as np
import os

vgg16 = VGG16(weights='imagenet', include_top=False) #Carrega a rede já treinada, neste caso a VGG16

x = vgg16.get_layer('block5_conv3').output  #pega a saida da camada de convolução 3 do bloco 5
x = GlobalAveragePooling2D()(x) #realiza um pooling médio 2D na camada recebida
x = Dense(256, activation='relu')(x) #
x = Dropout(0.5)(x) #
x = Dense(1, activation='sigmoid')(x) #usa-se a sigmoid para obter a classificação

pasta_raiz = "" #define a pasta raiz contendo os splits para treino da rede
false_p_dist = "" #define o destino das imagens de erro
model_save_name = "" #define o nome principal para salvar o modelo
model_save_dest = "ISIC_ARCHIVE/" + false_p_dist #define o destino para salvar o modelo

#define o tamanho padrão das imagens que serão passadas na rede, sendo que a mesma aceita imagens maiores que o padrão definido da VGG16 (255x255)
#img_width = 881
#img_height = 770

#img_width = 768
#img_height = 576

img_width = 900
img_height = 612

#define o batch_size, treino e validação, das imagens de acordo com a memória disponivél na máquina
batch_size = 15
batch_size_val = 40

#define as épocas 
epochs1 = 10
epochs2 = 50

#0 -> melanomas 1 -> normais
class_weight = {0: 2.7, 1: 1.} #total de imagens de treino normais / total de imagens de treino melanomas 

#diretórios de treino, validação e teste, para cada split 
trains_dir= [pasta_raiz + "split1/train", pasta_raiz + "split2/train", pasta_raiz + "split3/train", pasta_raiz + "split4/train", pasta_raiz + "split5/train"]
validations_dir= [pasta_raiz + "split1/valid", pasta_raiz + "split2/valid", pasta_raiz + "split3/valid", pasta_raiz + "split4/valid", pasta_raiz + "split5/valid"]
tests_dir = [pasta_raiz + 'split1/test', pasta_raiz + 'split2/test', pasta_raiz + 'split3/test', pasta_raiz + 'split4/test', pasta_raiz + 'split5/test']

#DataGenerator utilizado para fazer o augmentation on the batch
datagen = ImageDataGenerator(rescale=1., 
    featurewise_center=True,
    rotation_range=10,
    width_shift_range=.1,
    height_shift_range=.1,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode="reflect") #generator de treino

validgen = ImageDataGenerator(rescale=1., featurewise_center=True) #generator de teste e validação, evita-se realizar alterações nas imagens

#como as imagens apresentam um tamanho maior que o padrão, deve-se fazer uma normalização das mesmas para que sejam aceitas na rede
datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(1,1,3)
validgen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(1,1,3)

#for train_data_dir, test_data_dir, validation_data_dir in zip(trains_dir, tests_dir, validations_dir): 
for split in range(sp-1, 5): #laço de repetição para atingir todos os diretórios com imagens para treino, validação e teste
  train_data_dir = trains_dir[split]
  test_data_dir = tests_dir[split]
  validation_data_dir = validations_dir[split]

  #pegando as camadas da VGG e jogando no modelo que será feito o fine-tuning
  model_final = Model(inputs=vgg16.input, outputs=x) #pega toda a rede VGG16 e mais as camadas criadas e insere em um novo modelo

  for layer in vgg16.layers: #realiza o congelamento das primerias camadas
    layer.trainable = False

  model_final.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy']) #compila o modelo 
  
  #definindo os geradores para cada pasta
  train_gen = datagen.flow_from_directory( #generator para treino
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True)

  val_gen = validgen.flow_from_directory( #generator para validação
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True)

  test_gen = validgen.flow_from_directory( #generator para teste
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode="binary",
        shuffle=False)
  
  #pega a quantidade de amostras de cada generator
  train_samples = len(train_gen.filenames)
  validation_samples = len(val_gen.filenames)
  test_samples = len(test_gen.filenames)
  
  #inicio da fase de treino
  #as imagens são passadas na rede
  early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto') #define-se um ponto de pausa quando a taxa de perda parar de reduzir

  model_final.fit_generator(train_gen, #realiza o treino da rede sobre os generators
                            epochs=epochs1, 
                            steps_per_epoch=int(train_samples/batch_size), 
                            validation_data=val_gen, 
                            validation_steps=batch_size_val, 
                            class_weight = class_weight,
                            verbose=1, callbacks=[early_stopping])

  for layer in model_final.layers[:15]: #congela as camadas finais da rede
    layer.trainable = False

  for layer in model_final.layers[15:]: #descongela as camadas iniciais da rede
    layer.trainable = True

  model_final.compile(optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True),  loss='binary_crossentropy', metrics=['accuracy']) #recompila a rede

  early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto') #define um ponto de parada
  
  model_final.fit_generator(train_gen, #realiza mais uma treino na rede para melhor as métricas da rede
                            epochs=epochs2, 
                            steps_per_epoch=int(train_samples/batch_size), 
                            validation_data=val_gen, 
                            validation_steps=int(validation_samples/batch_size),
                            class_weight = class_weight,
                            verbose=1, callbacks=[early_stopping])
  #fim da fase de treino
  
  labels = [] #classes correspondente das imagens
  for i in range(0, test_samples): #separação das imagens para teste da rede
    test_img, test_label = next(test_gen)
    labels.append(int(test_label))
  
  preds = model_final.predict_generator(test_gen, test_samples) #realiza o teste de classificação das imagens na rede
  
  preds_rounded = []
  for pred in preds: #adiciona os valores arredondados no vetor
    if (pred > .5):
        preds_rounded.append(1)
    else:
        preds_rounded.append(0)
  
  print()
  print("SPLIT " + str(split+1))
  print() 
  print(metrics.accuracy_score(labels, preds_rounded)) #calcula o acurácia

  print(metrics.classification_report(labels, preds_rounded)) #pega outras métricas importantes para o teste

  confusion = metrics.confusion_matrix(labels, preds_rounded) #monta a matriz de confusão
  
  print() 
  print(confusion) #imprime a matriz de confusão
  print()
  
  #realiza a separação das imagens que deram falso positivos e falsos negativos: Erros da rede
  if os.path.exists("drive/PIBIC/Falsos_positivos/"+ false_p_dist +"/sp_"+str(split+1)) == False: #cria a pasta do split correspondente
    os.mkdir("drive/PIBIC/Falsos_positivos/"+ false_p_dist +"/sp_"+str(split+1))
  
  if os.path.exists("drive/PIBIC/Falsos_positivos/"+ false_p_dist +"/sp_"+str(split+1)+"/normais") == False: #cria a pasta para os erros das normais
    os.mkdir("drive/PIBIC/Falsos_positivos/"+ false_p_dist +"/sp_"+str(split+1)+"/normais")
   
  if os.path.exists("drive/PIBIC/Falsos_positivos/"+ false_p_dist +"/sp_"+str(split+1)+"/melanomas") == False: #cria a pasta para os erros das melanomas
    os.mkdir("drive/PIBIC/Falsos_positivos/"+ false_p_dist +"/sp_"+str(split+1)+"/melanomas")
  
  for i in range(0, test_samples): #realiza a cópia das imagens erradas para suas respectivas pastas
    if(preds_rounded[i] != labels[i]):
      if labels[i] == 1:
        dest = "/normais"
      else:
        dest = "/melanomas"
        
      copy(test_data_dir + '/' +test_gen.filenames[i], "drive/PIBIC/Falsos_positivos/"+ false_p_dist +"/sp_"+str(split+1)+dest) #realização da cópia das imagens
      
  model_final.save(model_save_dest +"/"+ model_save_name + "_SP" + str(split+1) + ".h5") #salva o modelo treinado

  del model_final