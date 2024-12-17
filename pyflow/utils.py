import pandas as pd                         
import numpy as np                          
import datetime as dt                       
import statsmodels.api as sm                
import seaborn as sns                       
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from matplotlib.pyplot import figure
import re
import pydotplus
import csv
import os
from sklearn.model_selection import train_test_split
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from os import walk
from os import path
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
import constants
import joblib



dtypes={
    constants.FECHA:        "string",
    constants.ENCAP:        "string",
    constants.ICMP:         "string",
    constants.RE_PERD:      "string",
    constants.C_VENTANA:    "string",
    constants.ECN:          "string",
    constants.FRAG:         "string",
    constants.OP_IP:        "string",
    constants.TCP_M:        "int64",
    constants.TCP_w:        "int64",
    constants.TCP_s:        "int64",
    constants.TCP_a:        "int64",
    constants.TCP_e:        "int64",
    constants.TCP_E:        "int64",
    constants.TCP_T:        "int64",
    constants.TCP_c:        "int64",
    constants.TCP_N:        "int64",
    constants.TCP_O:        "int64",
    constants.TCP_S:        "int64",
    constants.TCP_D:        "int64",
    constants.PROTO:        "string",
    constants.SRCADDR:      "string",
    constants.DIR:          "string",
    constants.DSTADDR:      "string",
    constants.PKTS:         "int64",
    constants.SRCPKTS:      "int64",
    constants.DSTPKTS:      "int64",
    constants.STATE:        "string",
    constants.SRCLOAD:      "float64",
    constants.DSTLOAD:      "float64",
    constants.LOSS:         "float64",
    constants.RATE:         "float64",
    constants.MEAN:         "float64",
    constants.STDDEV:       "float64",
    constants.RUNTIME:      "float64",
    constants.IDLE:         "float64",
    constants.TRANS:        "int64",
    constants.SUM:          "float64",
    constants.STTL:         "float64",
    constants.DTTL:         "float64",
    constants.PCR:          "float64",
    constants.TCPRTT:       "float64",
    constants.SYNACK:       "float64",
    constants.ACKDAT:       "float64",
    constants.SRCWIN:       "int64",
    constants.DSTWIN:       "int64"
};

columnsToEncode = [
    constants.ICMP,
    constants.RE_PERD,
    constants.C_VENTANA,
    constants.ECN,
    constants.OP_IP,
    constants.PROTO,
    constants.STATE
];




def flowPreprocesor(flow):
    flow=flow.strip()
    # Flags field
    flags = addFlagsSeparators(flow, 28, 7)
    # TCP opt field
    tcpOpt = addFlagsSeparators(flow, 37, 12, True)
    flow = flow[:27] + flags + " " + tcpOpt + flow[37+12:]
    flow=flow.split()
    flow[1] = flow[0]+" "+flow[1]
    flow.pop(0)
    while (len(flow) < len(dtypes.keys())):
        flow.append('0');
        
    return flow

def addFlagsSeparators(flowStr,start, length, binary = False):
    newFlags=""
    for i in range(start, start+length):
        if(flowStr[i]==' '):
            newFlags+= '0' if binary else '-'
        else:
            newFlags+= '1' if binary else flowStr[i]
        newFlags+=' '
    return newFlags





def createCSV(dataDir, filterCriteria = None):
    csvPath = f'{dataDir}/netflow.csv'
    try:
        os.remove(csvPath);
        print('...deleting prev CSV')
    except:
        var=0

        
    open(csvPath, "w").close()
    for (dirpath, dirnames, filenames) in walk(dataDir):
        for fname in filenames:
            match = re.search(".+.txt$", fname);
            if match is not None:
                print(fname)
                argusOutputFile = open(f'{dirpath}/{fname}', "r");
            
                #do nothing with header
                argusOutputFile.readline();
                
                fileContent = [i for i in argusOutputFile.readlines()]

                if filterCriteria is not None:
                    fileContent = list(filter(filterCriteria, fileContent))
                
                with open(f'{csvPath}', "a") as csvFile:
                    for line in fileContent:
                        writer = csv.writer(csvFile)
                        writer.writerow(flowPreprocesor(line))
    print("txt's >> CSV DONE")


    

def uniqueValuesOfColumns(df):
    for column in df:
        print(column, df[column].unique())


def getFlowDataFrame(flow):
    flow=flowPreprocesor(flow)
    
    df = pd.DataFrame(
        np.array([flow]),
        columns=dtypes.keys()
    )
    for t in dtypes:
        df[t] = df[t].astype(dtypes[t])
    return df

def createDataFrame(source, forceCsvCreation = False):
    csvPath = f'{source}/netflow.csv'
    csvExists = os.path.exists(csvPath)
    
    if not csvExists or forceCsvCreation:
        print ('no CSV, creating one');
        createCSV(source)
    
    df = pd.read_csv(
        csvPath,
        header=None,
        names=dtypes.keys(),
        dtype=dtypes
        
    )

    print("CSV >> DF DONE")
    return df


def preprocesDataFrame(df):
    df = df.drop(columns=[
            constants.FECHA,
            constants.SRCADDR,
            constants.DSTADDR,
            constants.DIR
        ])

    df= pd.get_dummies(df, columns=df.select_dtypes("string").columns.tolist());
    #print("get dummies DONE")
    return df


def dimensionalityReduction(df):    
    lowVarianceFilter = VarianceThreshold(0.01)
    lowVarianceFilter.fit(df, df['es_escaneo'])
    print(df.columns[[not e for e in lowVarianceFilter.get_support()]])
    
    df=pd.DataFrame(data=lowVarianceFilter.transform(df),
                  columns=df.columns[lowVarianceFilter.get_support()])
    
    print("low variance columns REMOVED")

    
    corr  = df.corr()
    correlated=[]
    plotCorrelation(df)
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.95:
                print("correlation: ", corr.columns[i], corr.columns[j])
                colname = corr.columns[i]
                correlated.append(colname)
    
    df = df.drop(columns=correlated)
    print("corralated columns REMOVED")

    print(df.columns.tolist())
    print(len(df.columns.tolist())) 
    return df

def  dimensionalityReduction_2(df):    
    
    # Separar la columna objetivo
    target_column = 'es_escaneo'
    target = df[target_column]
    features = df.drop(columns=[target_column])  # Excluir la columna objetivo
    
    # Paso 1: Filtro de baja varianza
    lowVarianceFilter = VarianceThreshold(0.1)
    lowVarianceFilter.fit(features)  # Ajustar sólo en las características
    print(features.columns[[not e for e in lowVarianceFilter.get_support()]])
    
    features = pd.DataFrame(data=lowVarianceFilter.transform(features),
                            columns=features.columns[lowVarianceFilter.get_support()])
    
    print("low variance columns REMOVED")

    # Paso 2: Eliminar características correlacionadas
    corr = features.corr()
    correlated = []
    plotCorrelation(features)
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.95:
                print("correlation: ", corr.columns[i], corr.columns[j])
                colname = corr.columns[i]
                correlated.append(colname)
    
    features = features.drop(columns=correlated)
    print("correlated columns REMOVED")

    # Volver a unir las características con la columna objetivo
    df = pd.concat([features, target], axis=1)

    print(df.columns.tolist())
    print(len(df.columns.tolist())) 
    return df




def filterByIP(line):
    return "2021-05-" in line or line.find(windowsIp) == -1



#### plot functions
def plotConfusionMatrix(dtree, X_test, Y_test):
    y_pred = dtree.predict(X_test)
    cm = confusion_matrix(X_test, y_pred);
     
    #disp = plot_confusion_matrix(confusion_matrix=cm) 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm) 
    #print(disp.ConfusionMatrixDisplay)
    disp.plot(cmap=plt.cm.Blues)
    FN=0
    TP=0
    if len(disp.confusion_matrix) == 2:
        FN = disp.confusion_matrix[1][0]
        TP = disp.confusion_matrix[1][1] if len(disp.confusion_matrix[1]) == 2 else 0
                
    TN = disp.confusion_matrix[0][0]
    FP = disp.confusion_matrix[0][1] if len(disp.confusion_matrix[0]) == 2 else 0
    
    
    print("Precision: ", (TP/(TP + FP)) if TP+FP > 0 else 1);
    print("Recall: ", (TP/(TP + FN)) if TP + FN > 0 else 1);
    print("Tasa de error: ", (FP + FN)/(TP + TN + FP + FN));
    print("Precision total: ", (TP + TN)/(TP + TN + FP + FN));
    plt.show()

def plotConfusionMatrix_2(dtree, X_test, Y_test):
    # Realiza la predicción
    y_pred = dtree.predict(X_test)
    
    # Calcula la matriz de confusión
    cm = confusion_matrix(Y_test, y_pred)
    
    # Muestra la matriz de confusión usando ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm) 
    disp.plot(cmap=plt.cm.Blues)
    
    # Calcula métricas
    FN = cm[1][0] if len(cm) == 2 else 0
    TP = cm[1][1] if len(cm) == 2 and len(cm[1]) == 2 else 0
    TN = cm[0][0]
    FP = cm[0][1] if len(cm[0]) == 2 else 0
    
    # Muestra las métricas calculadas
    print("Precision: ", (TP / (TP + FP)) if TP + FP > 0 else 1)
    print("Recall: ", (TP / (TP + FN)) if TP + FN > 0 else 1)
    print("Tasa de error: ", (FP + FN) / (TP + TN + FP + FN))
    print("Precision total: ", (TP + TN) / (TP + TN + FP + FN))
    
    # Muestra la gráfica
    plt.show()


def plotFeatureImportance (clf, X_train):
    f_importances=[]
    if not hasattr(clf, 'feature_importances_'):
        f_importances = np.mean([
            tree.feature_importances_ for tree in clf.estimators_
        ], axis=0)
    else:
        f_importances = clf.feature_importances_

    feat_importances = pd.Series(f_importances, index=X_train.columns)
    plot = feat_importances.nlargest(len(X_train.columns)).plot(kind='barh')
    plt.show()

    
    

def plotCorrelation(df):
    corr=df.corr()
    print(corr)
    plt.matshow(corr)
    plt.show()
