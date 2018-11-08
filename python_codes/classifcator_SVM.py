from sklearn.model_selection import train_test_split #faz o balanço dos dados entre treino e teste
from sklearn.datasets import load_svmlight_file #responsavel pela leitura do arquivo .libsvm
from sklearn import svm
from sklearn import metrics

def load_svm(path):
    data = load_svmlight_file(path)
    return data[0].toarray(), data[1]


file = "vgg16_features.libsvm"

x, y = load_svm(file)

split_test_size = 0.30

for i in range(0, 5):
    X_treino, X_teste, Y_treino, Y_teste = train_test_split(x, y, test_size=split_test_size, random_state=42)

    modelo = svm.SVC()
    modelo.fit(X_treino, Y_treino.ravel())
    svm_predict_train = modelo.predict(X_treino)
    svm_predict_test = modelo.predict(X_teste)
    confusion_matrix = metrics.confusion_matrix(Y_teste, svm_predict_test, labels = [1, 0])
    sensi = confusion_matrix[0][0] / len(X_teste)
    espec = confusion_matrix[1][1] / len(X_teste)
    accu_test = metrics.accuracy_score(Y_teste, svm_predict_test)
    accu_train = metrics.accuracy_score(Y_treino, svm_predict_train)
    preci = metrics.precision_score(Y_teste, svm_predict_test, labels = [1, 0], average=None)
    recall = metrics.recall_score(Y_teste, svm_predict_test, labels = [1, 0], average=None)
    f1_s = metrics.f1_score(Y_teste, svm_predict_test, labels = [1, 0], average=None)
    
    print("   Treino Accuracy        Teste Accuracy        precision        recall        f1-score")
    print("1       {0:.4f}                  {1:.4f}             {2:.2f}           {3:.2f}          {4:.2f}".format(accu_train, accu_test, preci[0], recall[0], f1_s[0]))
    print("0                                                  {0:.2f}           {1:.2f}          {2:.2f}".format(preci[1], recall[1], f1_s[1]))
    print("\n\n")
    print("Confusion Matrix")
    print("{0}".format(confusion_matrix))
    print("\n\n")

