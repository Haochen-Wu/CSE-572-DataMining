import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import pickle

gcm1 = pd.read_csv("CGMData.csv")
gcm2 = pd.read_csv("CGM_patient2.csv")
insulin1 = pd.read_csv("InsulinData.csv")
insulin2 = pd.read_csv("Insulin_patient2.csv")


def transformTime(date, time):
    dateTime = date + " " + time
    try:
        dateTime = datetime.datetime.strptime(dateTime, "%m/%d/%Y %H:%M:%S")
    except:
        try:
            dateTime = datetime.datetime.strptime(dateTime, "%Y-%m-%d %H:%M:%S")
        except:
            dateTime = datetime.datetime.strptime(dateTime, "%Y-%m-%d 00:00:00 %H:%M:%S")
    return dateTime


gcm1["datetime"] = gcm1.apply(lambda row: transformTime(row["Date"], row["Time"]), axis=1)
gcm2["datetime"] = gcm2.apply(lambda row: transformTime(row["Date"], row["Time"]), axis=1)
insulin1["datetime"] = insulin1.apply(lambda row: transformTime(row["Date"], row["Time"]), axis=1)
insulin2["datetime"] = insulin2.apply(lambda row: transformTime(row["Date"], row["Time"]), axis=1)


def getTm(insulin):
    intakeData = insulin[insulin["BWZ Carb Input (grams)"] > 0]
    intakeTimes = list(intakeData["datetime"])[::-1]
    startTime = []
    for index1 in range(len(intakeTimes)):
        tm = intakeTimes[index1]
        timedelta = datetime.timedelta(minutes=120)
        end = tm + timedelta
        isOverlaped = False
        for index2 in range(index1 + 1, len(intakeTimes)):
            if tm < intakeTimes[index2] < end:
                isOverlaped = True
                break
        if isOverlaped:
            continue
        else:
            startTime.append(tm)
    return startTime


def getDataMatrix(startTime, cgm):
    mealMatrix = []
    noMealMatrix = []
    for timeIndex in range(len(startTime)):
        time = startTime[timeIndex]
        mealEnd = time + datetime.timedelta(minutes=120)
        mealStart = time - datetime.timedelta(minutes=30)
        meal = cgm[cgm["datetime"].between(mealStart, mealEnd)]
        mealMatrix.append(meal["Sensor Glucose (mg/dL)"].values)
        noStart = startTime[timeIndex] + datetime.timedelta(minutes=120)
        noEnd = noStart + datetime.timedelta(minutes=120)
        timeLabel = timeIndex
        while (timeLabel < len(startTime) - 1 and noEnd < startTime[timeIndex + 1]):
            noMealdata = cgm[cgm["datetime"].between(noStart, noEnd)]
            noMealMatrix.append(noMealdata["Sensor Glucose (mg/dL)"].values)
            noStart = noEnd
            noEnd = noEnd + datetime.timedelta(minutes=120)
            timeLabel += 1
    return mealMatrix, noMealMatrix


mealStart1 = getTm(insulin1)
mealStart2 = getTm(insulin2)
mealMatrix1, noMealMatrix1 = getDataMatrix(mealStart1, gcm1)
mealMatrix2, noMealMatrix2 = getDataMatrix(mealStart2, gcm2)
mealMatrix1 = [x for x in mealMatrix1 if len(x) == 30]
mealMatrix2 = [x for x in mealMatrix2 if len(x) == 30]
noMealMatrix1 = [x for x in noMealMatrix1 if len(x) == 24]
noMealMatrix2 = [x for x in noMealMatrix2 if len(x) == 24]
dfMeal1 = pd.DataFrame(mealMatrix1)
dfMeal2 = pd.DataFrame(mealMatrix2)
dfNoMeal1 = pd.DataFrame(noMealMatrix1)
dfNoMeal2 = pd.DataFrame(noMealMatrix2)


def handleMissingData(data):
    for row in data.itertuples():
        if pd.Series(list(row)).isnull().sum() / len(row) > 0.2:
            data = data.drop(index=row[0])
    return data


meal1 = handleMissingData(dfMeal1)
meal2 = handleMissingData(dfMeal2)
noMeal1 = handleMissingData(dfNoMeal1)
noMeal2 = handleMissingData(dfNoMeal2)
meal1["label"] = 1
meal2["label"] = 1
noMeal1["label"] = 0
noMeal2["label"] = 0
meal = pd.concat([meal1, meal2], axis=0).reset_index()
noMeal = pd.concat([noMeal1, noMeal2], axis=0).reset_index()


def getPeak(row):
    data = [x for x in list(row) if type(x) != str and x > 0]
    maxData = max(data)
    return maxData


def getStd(row):
    data = [x for x in list(row) if type(x) != str and x > 0]
    std = np.mat(data).std()
    return std


def getGN(row):
    meal = row[5]
    peak = getPeak(row)
    gn = (peak - meal) / meal
    return gn


def fft(row):
    data = [x for x in list(row) if type(x) != str and x > 0]
    fftArray = np.fft.fft(data)
    freqArray = np.fft.fftfreq(len(data))
    fftDict = dict(zip(fftArray, freqArray))
    fftArray = sorted(fftArray)
    fft1 = fftArray[-3]
    fft2 = fftArray[-5]
    freq1 = fftDict.get(fft1)
    freq2 = fftDict.get(fft2)
    return [abs(fft1), abs(freq1), abs(fft2), abs(freq2)]


def fft1(row):
    fft1, fft2, fft3, fft4 = fft(row)
    return fft1


def fft2(row):
    fft1, fft2, fft3, fft4 = fft(row)
    return fft2


def fft3(row):
    fft1, fft2, fft3, fft4 = fft(row)
    return fft3


def fft4(row):
    fft1, fft2, fft3, fft4 = fft(row)
    return fft4


def getGT(row):
    index = row.index[1:-1]
    data = [x for x in list(row)[1:-1] if type(x) != str and x > 0]
    Map = dict(zip(row.values[1:-1], index))
    peak = max(data)
    peakLoc = Map.get(peak)
    mealLoc = Map.get(row[5])
    if peakLoc == 5:
        return 0
    else:
        GT = (peak - row[5]) / (abs(peakLoc - 5) * 5 * 60)
        return GT


def featureExtractor(data):
    peak = data.apply(lambda row: getPeak(row), axis=1)
    std = data.apply(lambda row: getStd(row), axis=1)
    GN = data.apply(lambda row: getGN(row), axis=1)
    Fft1 = data.apply(lambda row: fft1(row), axis=1)
    Fft2 = data.apply(lambda row: fft2(row), axis=1)
    Fft3 = data.apply(lambda row: fft3(row), axis=1)
    Fft4 = data.apply(lambda row: fft4(row), axis=1)
    GT = data.apply(lambda row: getGT(row), axis=1)
    return pd.DataFrame(
        {"peak": peak, "std": std, "GN": GN, "fft1": Fft1, "fft2": Fft2, "fft3": Fft3, "fft4": Fft4, "GT": GT})


mealFeature = featureExtractor(meal)
noMealFeature = featureExtractor(noMeal)
mealFeature["label"] = meal.label
noMealFeature["label"] = noMeal.label
combData = pd.concat([mealFeature, noMealFeature])
combineData = combData.sample(frac=1)
trainData = combineData.iloc[0:round(len(combineData.index) * 0.8)]
testData = combineData.iloc[round(len(combineData.index) * 0.8):]
x_train = trainData
x_test = testData
y_train = x_train.label
y_test = x_test.label
x_train = x_train.iloc[:, :-1]
x_test = x_test.iloc[:, :-1]
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
x_train_scaled = x_train_sacled.fillna(method="pad")
clf = SVC(kernel='rbf', gamma=5, C=40).fit(X_train_scaled, y_train)
with open('svm_classifier.pickle', 'wb') as dump_var:
    pickle.dump(clf, dump_var)
