from pickle import load
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score


dataset = load(open("./dataset_normalized.pkl","rb"))

def generate_classifiers():
    classifier_list = [
        RandomForestClassifier(n_estimators=100),
        MLPClassifier(hidden_layer_sizes=(208,104,208),activation="relu",max_iter=500)
        ]
    return classifier_list


def test1():
    print("+======================================================+")
    print("              Test 1 -  Identify if there is an object in the short configuration ")
    print("+======================================================+")

    x_empty = dataset.query("type=='empty' and configuration=='CURTO'").iloc[:,:104]
    x_object = dataset.query("type!='empty' and configuration=='CURTO'").iloc[:,:104]
    y_empty = [0]*len(x_empty)
    y_object = [1]*len(x_object)
    x = np.array(pd.concat([x_empty,x_object]).values)
    y = np.array(y_empty + y_object)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    for classifier in generate_classifiers():
        # K-Fold 10
        c = classifier
        scores = cross_val_score(c,x,y,cv=10)
        print(f"Cross validation score: {scores.mean()} +/- {scores.std()}")
        classifier.fit(x_train,y_train)
        y_pred = classifier.predict(x_test)
        print(classifier)
        print(accuracy_score(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        # Confusion matrix
        print(confusion_matrix(y_test,y_pred))
        print("=======================================")




def test2():
    print("+======================================================+")
    print("              Test 2 - Identify if there is an object in the long configuration ")
    print("+======================================================+")
    x_empty = dataset.query("type=='empty' and configuration=='COMPRIDO'").iloc[:,:104]
    x_object = dataset.query("type!='empty' and configuration=='COMPRIDO'").iloc[:,:104]
    y_empty = [0]*len(x_empty)
    y_object = [1]*len(x_object)
    x = np.array(pd.concat([x_empty,x_object]).values)
    y = np.array(y_empty + y_object)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    for classifier in generate_classifiers():
        # K-Fold 10
        c = classifier
        scores = cross_val_score(c,x,y,cv=10)
        print(f"Cross validation score: {scores.mean()} +/- {scores.std()}")
        classifier.fit(x_train,y_train)
        y_pred = classifier.predict(x_test)
        print(classifier)
        print(accuracy_score(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        # Confusion matrix
        print(confusion_matrix(y_test,y_pred))
        print("=======================================")


def test3():
    print("+======================================================+")
    print("              Test 3 - Identify if there is an organic object in the long configuration (independent of the day)")
    print("+======================================================+")
    x_empty = dataset.query("type=='empty' and configuration=='COMPRIDO'").iloc[:,:104]
    x_object = dataset.query("type=='organic' and configuration=='COMPRIDO'").iloc[:,:104]
    y_empty = [0]*len(x_empty)
    y_object = [1]*len(x_object)
    x = np.array(pd.concat([x_empty,x_object]).values)
    y = np.array(y_empty + y_object)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    for classifier in generate_classifiers():
        # K-Fold 10
        c = classifier
        scores = cross_val_score(c,x,y,cv=10)
        print(f"Cross validation score: {scores.mean()} +/- {scores.std()}")
        classifier.fit(x_train,y_train)
        y_pred = classifier.predict(x_test)
        print(classifier)
        print(accuracy_score(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        # Confusion matrix
        print(confusion_matrix(y_test,y_pred))
        print("=======================================")

def test3b():
    print("+======================================================+")
    print("              Test 3b - Identify if there is an organic object in the long configuration (for day 2)")
    print("+======================================================+")
    x_empty = dataset.query("type=='empty' and configuration=='COMPRIDO' and day=='2'").iloc[:,:104]
    x_object = dataset.query("type=='organic' and configuration=='COMPRIDO' and day=='2' ").iloc[:,:104]
    y_empty = [0]*len(x_empty)
    y_object = [1]*len(x_object)
    x = np.array(pd.concat([x_empty,x_object]).values)
    y = np.array(y_empty + y_object)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    for classifier in generate_classifiers():
        # K-Fold 10
        c = classifier
        scores = cross_val_score(c,x,y,cv=10)
        print(f"Cross validation score: {scores.mean()} +/- {scores.std()}")
        classifier.fit(x_train,y_train)
        y_pred = classifier.predict(x_test)
        print(classifier)
        print(accuracy_score(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        # Confusion matrix
        print(confusion_matrix(y_test,y_pred))
        print("=======================================")


def test4():
    print("+======================================================+")
    print("              Test 4 - Identify if there is a metal object in the long configuration (independent of the day)  ")
    print("+======================================================+")
    x_empty = dataset.query("type=='empty' and configuration=='COMPRIDO'").iloc[:,:104]
    x_object = dataset.query("type=='metalic' and configuration=='COMPRIDO'").iloc[:,:104]
    y_empty = [0]*len(x_empty)
    y_object = [1]*len(x_object)
    x = np.array(pd.concat([x_empty,x_object]).values)
    y = np.array(y_empty + y_object)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    for classifier in generate_classifiers():
        # K-Fold 10
        c = classifier
        scores = cross_val_score(c,x,y,cv=10)
        print(f"Cross validation score: {scores.mean()} +/- {scores.std()}")
        classifier.fit(x_train,y_train)
        y_pred = classifier.predict(x_test)
        print(classifier)
        print(accuracy_score(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        # Confusion matrix
        print(confusion_matrix(y_test,y_pred))
        print("=======================================")

def test4b():
    print("+======================================================+")
    print("              Test 4b - Identify if there is a metal object in the long configuration (for day 2)")
    print("+======================================================+")
    x_empty = dataset.query("type=='empty' and configuration=='COMPRIDO' and day=='2'").iloc[:,:104]
    x_object = dataset.query("type=='metalic' and configuration=='COMPRIDO' and day=='2' ").iloc[:,:104]
    y_empty = [0]*len(x_empty)
    y_object = [1]*len(x_object)
    x = np.array(pd.concat([x_empty,x_object]).values)
    y = np.array(y_empty + y_object)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    for classifier in generate_classifiers():
        # K-Fold 10
        c = classifier
        scores = cross_val_score(c,x,y,cv=10)
        print(f"Cross validation score: {scores.mean()} +/- {scores.std()}")
        classifier.fit(x_train,y_train)
        y_pred = classifier.predict(x_test)
        print(classifier)
        print(accuracy_score(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        # Confusion matrix
        print(confusion_matrix(y_test,y_pred))
        print("=======================================")


def test_5():
    print("+======================================================+")
    print("              Test 5 - Identify if the object is organic or metalic in the long configuration (independent of the day)  ")
    print("+======================================================+")

    x_metalico = dataset.query("type=='metalic' and configuration=='COMPRIDO'").iloc[:,:104]
    x_organico = dataset.query("type=='organic' and configuration=='COMPRIDO'").iloc[:,:104]
    y_metalico = [0]*len(x_metalico)
    y_organico = [1]*len(x_organico)
    x = np.array(pd.concat([x_metalico,x_organico]).values)
    y = np.array(y_metalico + y_organico)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    for classificador in generate_classifiers():
        print(classificador)
        # K-Fold 10
        c = classificador
        scores = cross_val_score(c,x,y,cv=10)
        print(f"Cross validation score: {scores.mean()} +/- {scores.std()}")
        classificador.fit(x_train,y_train)
        y_pred = classificador.predict(x_test)
        print(classificador)
        print(accuracy_score(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        # Confusion matrix
        print(confusion_matrix(y_test,y_pred))
        print("=======================================")


