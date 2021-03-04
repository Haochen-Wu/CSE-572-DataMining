import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
pickle_in = open('svm_classifier.pickle', 'rb')
pickle_clf = pickle.load(pickle_in)
testData = pd.read_csv("test.csv",header=None)
testData["label"] = 1
def getPeak(row):
    data = [x for x in list(row) if type(x)!=str and x > 0]
    maxData = max(data)
    return maxData
def getStd(row):
    data = [x for x in list(row) if type(x)!=str and x > 0]
    std = np.mat(data).std()
    return std
def getGN(row):
    meal = row[5]
    peak = getPeak(row)
    gn = (peak - meal) / meal
    return gn
def fft(row):
    data = [x for x in list(row) if type(x)!=str and x > 0]
    fftArray = np.fft.fft(data)
    freqArray = np.fft.fftfreq(len(data))
    fftDict = dict(zip(fftArray,freqArray))
    fftArray= sorted(fftArray)
    fft1 = fftArray[-3]
    fft2 = fftArray[-5]
    freq1 = fftDict.get(fft1)
    freq2 = fftDict.get(fft2)
    return [abs(fft1),abs(freq1),abs(fft2),abs(freq2)]
def fft1(row):
    fft1,fft2,fft3,fft4 = fft(row)
    return fft1
def fft2(row):
    fft1,fft2,fft3,fft4 = fft(row)
    return fft2
def fft3(row):
    fft1,fft2,fft3,fft4 = fft(row)
    return fft3
def fft4(row):
    fft1,fft2,fft3,fft4 = fft(row)
    return fft4

def getGT(row):
    index = row.index[1:-1]
    data = [x for x in list(row)[1:-1] if type(x)!=str and x > 0]
    Map = dict(zip(row.values[1:-1],index))
    peak = max(data)
    peakLoc = Map.get(peak)
    mealLoc = Map.get(row[5])
    if peakLoc == 5:
        return 0
    else:
        GT = (peak - row[5]) / (abs(peakLoc - 5)*5*60)
        return GT


def featureExtractor(data):
    peak = data.apply(lambda row:getPeak(row),axis=1)
    std = data.apply(lambda row:getStd(row),axis=1)
    GN = data.apply(lambda row:getGN(row),axis=1)
    Fft1 = data.apply(lambda row:fft1(row),axis=1)
    Fft2 = data.apply(lambda row:fft2(row),axis=1)
    Fft3 = data.apply(lambda row:fft3(row),axis=1)
    Fft4 = data.apply(lambda row:fft4(row),axis=1)
    GT = data.apply(lambda row:getGT(row),axis=1)
    return pd.DataFrame({"peak":peak,"std":std,"GN":GN,"fft1":Fft1,"fft2":Fft2,"fft3":Fft3,"fft4":Fft4,"GT":GT})

x_test = featureExtractor(testData)
x_test = x_test.dropna()
scaler =  MinMaxScaler()
X_test_scaled = scaler.fit_transform(x_test)
result = pd.DataFrame(pickle_clf.predict(X_test_scaled))
result.to_csv("Result.csv",header=None,index=False)