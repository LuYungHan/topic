import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy import signal

np.random.seed(202205456)

data0506 = [
            "../0506data/手2022-05-06 114900-4.csv",
            "../0506data/手2022-05-06 113600-4.csv",
            "../0506data/手2022-05-06 115200-2.csv",
            "../0506data/腳2022-05-06 120000-4.csv",
            "../0506data/腳2022-05-06 120300-3.csv",
            "../0506data/腳2022-05-06 120500-4.csv"]
                                                    
data0510 = [
            "../0510data/手腳2022-05-10 104600.csv",
            "../0510data/手腳2022-05-10 104400.csv",
            "../0510data/手腳2022-05-10 104800.csv",
            "../0510data/手腳2022-05-10 104000.csv",
            "../0510data/手2022-05-10 103600.csv",
            "../0510data/手2022-05-10 103800.csv",
            "../0510data/手2022-05-10 104000.csv",
            "../0510data/腳2022-05-10 103600.csv",
            "../0510data/腳2022-05-10 103800.csv",
            "../0510data/腳2022-05-10 104000.csv",
            "../0510data/手2022-05-10 103600優化.csv",
            "../0510data/手2022-05-10 103800優化.csv",
            "../0510data/手2022-05-10 103600優化.csv",#12
            "../0510data/手2022-05-10 103800優化.csv",#13
            "../0510data/手2022-05-10 104000優化.csv",#14
            "../0510data/腳2022-05-10 103600優化.csv",#15
            "../0510data/腳2022-05-10 103800優化.csv",#16
            "../0510data/腳2022-05-10 104000優化.csv"]#17

dataFft = ["../Add_fft_features_data/AfterFft_36_Foot.csv",#0
          "../Add_fft_features_data/AfterFft_38_Foot.csv",#1
          "../Add_fft_features_data/AfterFft_40_Foot.csv",#2
          "../Add_fft_features_data/AfterFft_36_Hand.csv",#3
          "../Add_fft_features_data/AfterFft_38_Hand.csv",#4
          "../Add_fft_features_data/AfterFft_40_Hand.csv",#5
          "../Add_fft_features_data/AfterFft_陀螺儀_36_Hand.csv",#6
          "../Add_fft_features_data/AfterFft_陀螺儀_38_Hand.csv",#7
          "../Add_fft_features_data/AfterFft_陀螺儀_40_Hand.csv",#8
          "../Add_fft_features_data/AfterFft_陀螺儀_36_Foot.csv",#9
          "../Add_fft_features_data/AfterFft_陀螺儀_38_Foot.csv",#10
          "../Add_fft_features_data/AfterFft_陀螺儀_40_Foot.csv",#11
           "../Add_fft_features_data/AfterFft_fft_all_36_Hand.csv",#12
          "../Add_fft_features_data/AfterFft_fft_all_38_Hand.csv",#13
          "../Add_fft_features_data/AfterFft_fft_all_40_Hand.csv",#14
          "../Add_fft_features_data/AfterFft_fft_all_36_Foot.csv",#15
          "../Add_fft_features_data/AfterFft_fft_all_38_Foot.csv",#16
           "../Add_fft_features_data/AfterFft_fft_all_40_Foot.csv",#17
           "../Add_fft_features_data/AfterFft_fft_arctan_36_Hand.csv",#18
          "../Add_fft_features_data/AfterFft_fft_arctan_38_Hand.csv",#19
          "../Add_fft_features_data/AfterFft_fft_arctan_40_Hand.csv",#20
          "../Add_fft_features_data/AfterFft_fft_arctan_36_Foot.csv",#21
          "../Add_fft_features_data/AfterFft_fft_arctan_38_Foot.csv",#22
           "../Add_fft_features_data/AfterFft_fft_arctan_40_Foot.csv",#23
           "../Add_fft_features_data/AfterFft_fft_sqrt_arctan_36_Hand.csv",#24
          "../Add_fft_features_data/AfterFft_fft_sqrt_arctan_38_Hand.csv",#25
          "../Add_fft_features_data/AfterFft_fft_sqrt_arctan_40_Hand.csv",#26
          "../Add_fft_features_data/AfterFft_fft_sqrt_arctan_36_Foot.csv",#27
          "../Add_fft_features_data/AfterFft_fft_sqrt_arctan_38_Foot.csv",#28
           "../Add_fft_features_data/AfterFft_fft_sqrt_arctan_40_Foot.csv",#29

          ]
def SVM(data,data1):     
    data = np.array(pd.read_csv(data))
    labelSeries = data[:,16]
    dataSeries = np.append(data[:,1:4], data[:,7:13] , axis=1)
    data1 = np.array(pd.read_csv(data1))
    labelSeries1 = data1[:,16]
    dataSeries1 = np.append(data1[:,1:4], data1[:,7:13] , axis=1)
    svm = SVC(kernel='poly', degree=3, probability=True)
    svm.fit(dataSeries, labelSeries)
    predict = svm.predict(dataSeries1)
    svm_score = svm.score(dataSeries1,labelSeries1) 
    print(classification_report(labelSeries1,predict))
    return svm_score

def DT(data,data1):
    data = np.array(pd.read_csv(data))
    labelSeries = data[:,16]
    dataSeries = np.append(data[:,1:4], data[:,7:13] , axis=1)
    data1 = np.array(pd.read_csv(data1))
    labelSeries1 = data1[:,16]
    dataSeries1 = np.append(data1[:,1:4], data1[:,7:13] , axis=1)
    decisionTreeModel = DecisionTreeClassifier()
    decisionTreeModel.fit(dataSeries, labelSeries)
    predict= decisionTreeModel.predict(dataSeries1)
    DT_score = decisionTreeModel.score(dataSeries1,labelSeries1)
    print(classification_report(labelSeries1,predict))
    return DT_score

def RF(data,data1):
    fft = []
    fft1 = []
    amp = 2 * np.sqrt(2)
    data = np.array(pd.read_csv(data))
    data_len = len(data[:,1])
    labelSeries = data[1:data_len-16,16]
    dataSeries = np.append(data[1:data_len-16,1:4], data[1:data_len-16,7:10] , axis=1)
#     dataSeries = np.append(dataSeries, data[1:data_len-16,17:23] , axis=1)
    dataSeries = np.append(dataSeries, data[1:data_len-16,24:30] , axis=1)
    data1 = np.array(pd.read_csv(data1))
    data_len1 = len(data1[:,1])
    labelSeries1 = data1[1:data_len1-16,16]
    dataSeries1 = np.append(data1[1:data_len1-16,1:4], data1[1:data_len1-16,7:10] , axis=1)
#     dataSeries1 = np.append(dataSeries1, data1[1:data_len1-16,17:23] , axis=1)
    dataSeries1 = np.append(dataSeries1, data1[1:data_len1-16,24:30] , axis=1)
#     for i in range(data_len-17):
#         fft.append([])
#     for i in range(data_len1-17):
#         fft1.append([])
#     f, t, Zxx = signal.stft(data[:,3],fs=1,noverlap=17,nperseg=18,nfft=1000,return_onesided= False,padded=True,detrend=False,boundary=None)
#     f1, t1, Zxx1 = signal.stft(data1[:,3],fs=1,noverlap=17,nperseg=18,nfft=1000,return_onesided= False,padded=True,detrend=False,boundary=None)
#     for i in range (len(Zxx[100,:])):
#         fft[i].append(Zxx[100,i].real)
#         fft[i].append(Zxx[600,i].real)
#         fft[i].append(Zxx[800,i].real)
#         #add[i].append((Zxx[100,i].conjugate()*Zxx[100,i]).real)
#         #add[i].append((Zxx[600,i].conjugate()*Zxx[600,i]).real)
#         #add[i].append((Zxx[800,i].conjugate()*Zxx[800,i]).real)
#     for i in range (len(Zxx1[100,:])):
#         #add1[i].append((Zxx1[100,i].conjugate()*Zxx1[100,i]).real)
#         #add1[i].append((Zxx1[600,i].conjugate()*Zxx1[600,i]).real)
#         #add1[i].append((Zxx1[800,i].conjugate()*Zxx1[800,i]).real)
#         fft1[i].append(Zxx1[100,i].real)
#         fft1[i].append(Zxx1[600,i].real)
#         fft1[i].append(Zxx1[800,i].real)
#     dataSeries = np.append(dataSeries, fft , axis=1)
#     dataSeries1 = np.append(dataSeries1,fft1, axis=1)
    randomForestModel = RandomForestClassifier(
        random_state=1111,
        oob_score= True,
        n_estimators=100,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf = 50,
        criterion = 'entropy',
        max_features = 0.2,
        )
    randomForestModel.fit(dataSeries, labelSeries)
    predict = randomForestModel.predict(dataSeries1)
    RF_score = randomForestModel.score(dataSeries1,labelSeries1)
    print(classification_report(labelSeries1,predict))
    return RF_score
#Support Vector Machine支援向量機
'''print("--------Support Vector Machine ---------")
for i in range(1):
    print("Hand and foot:",SVM(data0510[i+2],data0510[i+3]))
    print("")
# for i in range(len(data0510)//2):
#     print("Hand:",SVM(data0510[i]))
#     print("")
#     print("Foot:",SVM(data0510[i+2]))
#     print("________________________________________________________")

print("--------------------------------\nDecision Treeeeeeeeeeeeeeeeeeee\n-------------------------------")
#for i in range(len(data0510)//2):
for i in range(1):
    print("Hand and foot:",DT(data0510[i+2],data0510[i+3]))
    print("")
    #print("Hand:",DT(data0510[i]))
    #print("")
    #print("Foot:",DT(data0510[i+2]))
    #print("________________________________________________________")'''

print("--------------------------------\nRandomForest\n-------------------------------")
#for i in range(len(data0506)//2):
for i in range(1):
    print("36train_38test hand:",RF(dataFft[i+24],dataFft[i+26]))
#     print("Hand and foot:",RF(data0510[i+15],data0510[i+16]))
    print("")
    #print("Foot:",RF(data0506[i+2],data0506[i+3]))
    #print("________________________________________________________")

