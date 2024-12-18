#Interfaz para probar el modelo en realtime
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
from colorama import Fore, Style
from os import walk
from sklearn import tree, preprocessing
import joblib
from utils import *
from bashUtils import *
import constants
import io
import sys
import json
demoDirPath = variables["demoData"];



def addMissingEncodedColumns(df):
    treeColumns = joblib.load('columns.txt')
    df = df.drop(columns=list(filter(lambda c: not (c in treeColumns), df.columns.tolist())))
    for c in treeColumns:
        if not (c in df):
            df[c] = 0

    df = df[treeColumns]
    return df

process = createArgusDaemonOutput(demoDirPath)
clf = joblib.load('rForest.pkl')
print("Real time netflow");
while True:
    header=True
    lines = io.TextIOWrapper(process.stdout, encoding="utf-8")
    #for line in io.TextIOWrapper(process.stdout, encoding="utf-8"):
    next(lines)
    for line in lines:
        if not header:
            df = getFlowDataFrame(line)

            output = '{:^22}'.format(df[constants.FECHA][0]) + '{:^10}'.format(df[constants.PROTO][0]) + '{:^35}'.format(df[constants.SRCADDR][0])+ '{:^35}'.format(df[constants.DSTADDR][0])+ '{:^5}'.format(df[constants.STATE][0])+ '{:^10}'.format(df[constants.SUM][0])+"\n"
            df = preprocesDataFrame(df)

            df = addMissingEncodedColumns(df)
            #print(df)

            if clf.predict(df) == 1:
                #sys.stdout.shell.write(output, "COMMENT")
             print(Fore.RED+output+Style.RESET_ALL)
            else:
                #sys.stdout.shell.write(output, "STRING")
             print(Fore.GREEN+output+Style.RESET_ALL)

        else:
            header=False

    if input() == 'STOP':
        process.kill()
        break;
