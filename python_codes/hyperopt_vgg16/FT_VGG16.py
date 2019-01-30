from keras.layers import Dropout, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Convolution2D
from hyperopt import STATUS_OK, STATUS_FAIL
from keras.models import load_model, Model
from keras.applications.vgg16 import VGG16
from hyperopt import hp, tpe, fmin, Trials
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from keras.optimizers import SGD
from sklearn import metrics
from bson import json_util
import keras.backend as K
import numpy as np
import traceback
import pickle
import json
import uuid
import os

#define o tamanho padrão das imagens que serão passadas na rede, sendo que a mesma aceita imagens maiores que o padrão definido da VGG16 (255x255)
#img_width = 881
#img_height = 770

#img_width = 768
#img_height = 576

img_width = 900
img_height = 612

batch_size = 15 #batch_size para o treino

RESULTS_DIR = "results/" #pasta para salvar os resultados dos treinamentos

train_data_dir = ""
validation_data_dir = ""
test_data_dir = ""

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

def build_model(hype_space):
    print("Hyperspace:")
    print(hype_space)

    vgg16 = VGG16(weights='imagenet', include_top=False) #Carrega a rede já treinada, neste caso a VGG16

    x = vgg16.get_layer(hype_space['output_layer']).output  #pega a saida da camada de convolução 3 do bloco 5
    
    for i in range(0, hype_space['qtd_conv']):
        x = Convolution2D(512, 3, 3, activation='relu')(x)

    if hype_space['pooling'] == 'AVG':
        x = GlobalAveragePooling2D()(x) #realiza um pooling médio 2D na camada recebida
    else:
        x = GlobalMaxPooling2D()(x)
    if hype_space['qtd_dense'] == 2:    
        x = Dense(int(hype_space['neurons']/2), activation='relu')(x)
    x = Dense(hype_space['neurons'], activation='relu')(x) 
    x = Dropout(hype_space['dropout'])(x) 
    if hype_space['classification'] == 'sigmoid':
        x = Dense(1, activation='sigmoid')(x) #usa-se a sigmoid para obter a classificação
    else:
        x = Dense(2, activation='softmax')(x)

    model_final = Model(inputs=vgg16.input, outputs=x) #pega toda a rede VGG16 e mais as camadas criadas e insere em um novo modelo

    for layer in vgg16.layers: #realiza o congelamento das primerias camadas
        layer.trainable = False
  
    model_final.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy']) #compila o modelo

    return model_final

def build_and_train(hype_space):
    #define o batch_size de validação, das imagens de acordo com a memória disponivél na máquina
    batch_size_val = 40

    #define as épocas 
    epochs1 = 10
    epochs2 = 50
    
                                    #0 -> melanomas 1 -> normais
    class_weight = {0: 2.7, 1: 1.} #total de imagens de treino normais / total de imagens de treino melanomas 

    model_final = build_model(hype_space)

    #pega a quantidade de amostras de cada generator
    train_samples = len(train_gen.filenames)
    validation_samples = len(val_gen.filenames)
    test_samples = len(test_gen.filenames)

    #inicio da fase de treino
    #as imagens são passadas na rede
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto') #define-se um ponto de pausa quando a taxa de perda parar de reduzir

    if hype_space['class_weight']: #realiza o treino da rede sobre os generators
        model_final.fit_generator(train_gen,
                                epochs=epochs1, 
                                steps_per_epoch=int(train_samples/batch_size), 
                                validation_data=val_gen, 
                                validation_steps=batch_size_val, 
                                class_weight = class_weight,
                                verbose=1, callbacks=[early_stopping])
    else:
        model_final.fit_generator(train_gen,
                                epochs=epochs1, 
                                steps_per_epoch=int(train_samples/batch_size), 
                                validation_data=val_gen, 
                                validation_steps=batch_size_val,
                                verbose=1, callbacks=[early_stopping])

    for layer in model_final.layers[:15]: #congela as camadas finais da rede
        layer.trainable = False

    for layer in model_final.layers[15:]: #descongela as camadas iniciais da rede
        layer.trainable = True

    model_final.compile(optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True),  loss='binary_crossentropy', metrics=['accuracy']) #recompila a rede

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto') #define um ponto de parada
  
    if hype_space['class_weight']: #realiza o treino da rede sobre os generators
        model_final.fit_generator(train_gen,
                                epochs=epochs2, 
                                steps_per_epoch=int(train_samples/batch_size), 
                                validation_data=val_gen, 
                                validation_steps=batch_size_val, 
                                class_weight = class_weight,
                                verbose=1, callbacks=[early_stopping])
    else:
        model_final.fit_generator(train_gen,
                                epochs=epochs2, 
                                steps_per_epoch=int(train_samples/batch_size), 
                                validation_data=val_gen, 
                                validation_steps=batch_size_val,
                                verbose=1, callbacks=[early_stopping])
    #fim da fase de treino
    
    #fase de teste
    labels = [] #classes correspondente das imagens
    for i in range(0, test_samples): #separação das imagens para teste da rede
        test_img, test_label = next(test_gen)
        labels.append(int(test_label))
    
    preds = model_final.predict_generator(test_gen, test_samples) #realiza o teste de classificação das imagens na rede

    if hype_space['classification'] == 'sigmoid':    
        preds_rounded = []
        for pred in preds: #adiciona os valores arredondados no vetor
            if (pred > .5):
                preds_rounded.append(1)
            else:
                preds_rounded.append(0)    
    elif hype_space['classification'] == 'softmax':
        preds_rounded = np.round(preds[:,0])

    acc = metrics.accuracy_score(labels, preds_rounded) #calcula o acurácia
    class_report = metrics.classification_report(labels, preds_rounded)

    model_name = "model_{}_{}".format(str(acc), str(uuid.uuid4())[:5])

    result = {
        'loss': 1-acc,
        'acurracy': acc,
        'report': class_report,
        'model_name': model_name,
        'space': hype_space,
        'status': STATUS_OK
    }

    print(result)

    return model_final, model_name, result

def print_json(result):
    """Pretty-print a jsonable structure (e.g.: result)."""
    print(json.dumps(
        result,
        default=json_util.default, sort_keys=True,
        indent=4, separators=(',', ': ')
    ))


def save_json_result(model_name, result):
    """Save json to a directory and a filename."""
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    with open(os.path.join(RESULTS_DIR, result_name), 'w') as f:
        json.dump(
            result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )


def load_json_result(best_result_name):
    """Load json from a path (directory + filename)."""
    result_path = os.path.join(RESULTS_DIR, best_result_name)
    with open(result_path, 'r') as f:
        return json.JSONDecoder().decode(
            f.read()
            # default=json_util.default,
            # separators=(',', ': ')
        )

def load_best_hyperspace():
    results = [
        f for f in list(sorted(os.listdir(RESULTS_DIR))) if 'json' in f
    ]
    if len(results) == 0:
        return None

    best_result_name = results[-1]
    return load_json_result(best_result_name)["space"]

def plot_best_model():
    """Plot the best model found yet."""
    space_best_model = load_best_hyperspace()
    if space_best_model is None:
        print("No best model to plot. Continuing...")
        return

    print("Best hyperspace yet:")
    print_json(space_best_model)

def optimize_cnn(hype_space):
    """Build a convolutional neural network and train it."""
    try:
        model, model_name, result = build_and_train(hype_space)

        # Save training results to disks with unique filenames
        save_json_result(model_name, result)

        K.clear_session()
        del model

        return result

    except Exception as err:
        try:
            K.clear_session()
        except:
            pass
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)
        return {
            'status': STATUS_FAIL,
            'err': err_str,
            'traceback': traceback_str
        }

    print("\n\n")


def run_a_trial():
    """Run one TPE meta optimisation step and save its results."""
    max_evals = nb_evals = 1

    print("Attempt to resume a past training if it exists:")

    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open("results.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    best = fmin(
        optimize_cnn,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open("results.pkl", "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")

space = {
    #A quantidade de camadas de convolução do modelo
    'qtd_conv': hp.choice('qtd_conv', [0, 1, 2]),
    #Camada que vai ser descongelada e utilizada para saída da VGG16
    'unfreeze': hp.choice('output_layer', ['block5_conv3', 'block5_conv2']),
    #Escolha do tipo de pooling na rede
    'pooling': hp.choice('pooling', ['AVG', 'MAX']),
    #Escolha da quantidade máxima de neurônios da rede
    'neurons': hp.choice('neurons', [256, 128]),
    #Quantidade de camadas Dense
    'qtd_dense': hp.choice('qtd_dense', [1, 2]),
    #Escolha da camada de classificação
    'classification': hp.choice('classification', ['sigmoid', 'softmax']),
    #Escolha do valor de dropout
    'dropout': hp.choice('dropout', [0.5, 0.4, 0.3]),
    #Habilitar o class_weight
    'class_weight': hp.choice('class_weight', [True, False])
}

if __name__ == "__main__":
    print("Now, we train many models, one after the other. "
          "Note that hyperopt has support for cloud "
          "distributed training using MongoDB.")

    print("\nYour results will be saved in the folder named 'results/'. "
          "You can sort that alphabetically and take the greatest one. "
          "As you run the optimization, results are consinuously saved into a "
          "'results.pkl' file, too. Re-running optimize.py will resume "
          "the meta-optimization.\n")

    while True:

        # Optimize a new model with the TPE Algorithm:
        print("OPTIMIZING NEW MODEL:")
        try:
            run_a_trial()
        except Exception as err:
            err_str = str(err)
            print(err_str)
            traceback_str = str(traceback.format_exc())
            print(traceback_str)

        # Replot best model since it may have changed:
        print("PLOTTING BEST MODEL:")
        plot_best_model()
