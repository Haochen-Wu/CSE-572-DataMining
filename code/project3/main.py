import numpy as np
import pandas as pd
import datetime

insulin = pd.read_csv("InsulinData.csv")
cgm = pd.read_csv("CGMData.csv")
inputCarb = insulin["BWZ Carb Input (grams)"]
minCarb = min(insulin[insulin["BWZ Carb Input (grams)"] > 0]["BWZ Carb Input (grams)"])
maxCarb = max(insulin[insulin["BWZ Carb Input (grams)"] > 0]["BWZ Carb Input (grams)"])
numClusters = round((maxCarb - minCarb) / 20)


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
        # get noMealData
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


insulin["datetime"] = insulin.apply(lambda row: transformTime(row["Date"], row["Time"]), axis=1)
cgm["datetime"] = cgm.apply(lambda row: transformTime(row["Date"], row["Time"]), axis=1)
tm = getTm(insulin)
mealMatrix, noMealMatrix = getDataMatrix(tm, cgm)
dfMeal = pd.DataFrame(mealMatrix)


def getCarb(tm, insulin):
    carbs = []
    for time in tm:
        carb = insulin[insulin["datetime"] == time]["BWZ Carb Input (grams)"].values
        carbs.append(carb)
    aux = []
    for carb in carbs:
        if len(carb) == 1:
            aux.append(carb[0])
        else:
            temp = 0
            for c in carb:
                if c > 0:
                    temp += c
            aux.append(temp)
    carbs = aux
    return carbs


carbs = getCarb(tm, insulin)
dfMeal["carbs"] = carbs


def handleMissingData(data):
    for row in data.itertuples():
        if pd.Series(list(row)).isnull().sum() / len(row) > 0.2:
            data = data.drop(index=row[0])
    return data


dfMeal = handleMissingData(dfMeal)
labels = dfMeal["carbs"]


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


mealFeatures = featureExtractor(dfMeal)
mealFeatures["carbs"] = labels


def getLabel(carb, sep, n, minCarb, maxCarb):
    label = 1
    currCarb = minCarb
    while (currCarb < maxCarb):
        if currCarb <= carb <= currCarb + sep:
            return label
        currCarb += sep
        label += 1
    return np.nan


mealFeatures["groundTruth"] = mealFeatures.apply(lambda row: getLabel(row["carbs"], 20, numClusters, minCarb, maxCarb),
                                                 axis=1)
mealFeatures = mealFeatures.dropna()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
data = mealFeatures.iloc[:, :8]
scaler = StandardScaler()
dataScaled = scaler.fit_transform(data)
kmeans = KMeans(init="random", n_clusters=numClusters, n_init=numClusters, random_state=0)
kmeans.fit(dataScaled)
kmeansPredict = list(kmeans.predict(dataScaled))
kmeansPredict = [x + 1 for x in kmeansPredict]


def getDistance(data1, data2):
    sqrtValue = []
    for i in range(len(data1)):
        sqrtValue.append(pow(data1[i] - data2[i], 2))
    distance = pow(sum(sqrtValue), 0.5)
    return distance


def getSSE(data):
    centroid = pd.DataFrame(data).mean().to_list()
    dist = 0
    for dataPoint in data:
        dist += pow(getDistance(dataPoint, centroid), 2)
    return dist


def iterativeDbscan(n, data):
    scaler = StandardScaler()
    dataScaled = scaler.fit_transform(data)
    db = DBSCAN(eps=2, min_samples=20).fit(dataScaled)
    labels = db.labels_
    numClusters = len(set(labels))
    tempData = data.copy()
    predict = list(db.labels_)
    tempData["dbscan"] = predict
    tempContainer = []
    while (numClusters < n):
        sse = []
        sse2 = []
        if numClusters == 2:
            for index in set([x for x in predict]):
                tempContainer.append(tempData[tempData["dbscan"] == index])
        for i in range(len(tempContainer)):
            dfScaled = StandardScaler().fit_transform(tempContainer[i])
            sse.append((getSSE(dfScaled), i))
        sse.sort(key=lambda x: x[0])
        tempDF = tempContainer[sse[-1][1]]
        tempContainer = tempContainer[:sse[-1][1]] + tempContainer[sse[-1][1] + 1:]
        km = KMeans(init="random", n_clusters=2, n_init=2, random_state=0)
        km.fit(StandardScaler().fit_transform(tempDF))
        tempDF["biscetK"] = list(km.predict(StandardScaler().fit_transform(tempDF)))
        tempDF1 = tempDF[tempDF["biscetK"] == 0]
        tempDF2 = tempDF[tempDF["biscetK"] == 1]
        tempContainer.append(tempDF1)
        tempContainer.append(tempDF2)
        numClusters += 1

    auxContainer = []
    for i in range(len(tempContainer)):
        tempDF3 = tempContainer[i]
        tempDF3["DBSCAN"] = i
        auxContainer.append(tempDF3)
    return pd.concat(auxContainer)


dbscanData = iterativeDbscan(numClusters, data)
dbscanData = dbscanData[["peak", "std", "GN", "fft1", "fft2", "fft3", "fft4", "GT", "DBSCAN"]]

kmeansPredict = [x - 1 for x in kmeansPredict]
kmeansResult = data.copy()
kmeansResult["KMEANS"] = kmeansPredict
kmeansResult = kmeansResult.join(mealFeatures["groundTruth"])
dbscanResult = dbscanData.join(mealFeatures["groundTruth"])


def modifyGD(row):
    return row - 1


def validationMatrix(result, n):
    data = []
    for i in range(n):
        data.append([])
    for index in range(n):
        tempData = result[result.iloc[:, 8] == index]
        for j in range(n):
            auxData = tempData[tempData["groundTruth"] == j]
            data[index].append(len(auxData))
    return data


kmeansResult["groundTruth"] = kmeansResult.apply(lambda row: modifyGD(row["groundTruth"]), axis=1)
dbscanResult["groundTruth"] = dbscanResult.apply(lambda row: modifyGD(row["groundTruth"]), axis=1)
kmeansMatrix = np.array(validationMatrix(kmeansResult, numClusters))
dbscanMatrix = np.array(validationMatrix(dbscanResult, numClusters))


def calcEntropy(matrix):
    total = 0
    rowSum = []
    E = []
    for i in range(len(matrix)):
        total += matrix[i].sum()
        E.append([])
        rowSum.append(matrix[i].sum())
        for j in range(len(matrix)):
            rate = matrix[i][j] / rowSum[i]
            if rate == 0:
                E[i].append(0)
            else:
                E[i].append((-np.log2(rate) * rate))
        E[i] = sum(E[i])
    result = []
    for k in range(len(E)):
        result.append(E[i] / rowSum[i])
    return sum(result)


def calcPurity(matrix):
    total = 0
    rowSum = []
    P = []
    columnSum = []
    transpose = matrix.transpose()
    for index in range(len(transpose)):
        columnSum.append(sum(transpose[index]))
    for i in range(len(matrix)):
        total += matrix[i].sum()
        P.append([])
        rowSum.append(matrix[i].sum())
        rate = matrix.max() / rowSum[i]
        P.append(rate)
    result = []
    for k in range(len(P)):
        result.append((columnSum[i] / total) * P[i])
    return sum(result) * 10


def calcSSE(dataResult):
    SSE = []
    for index in range(numClusters):
        tempData = dataResult[dataResult.iloc[:, 8] == index]
        SSE.append(getSSE(StandardScaler().fit_transform(tempData)))
    return sum(SSE)


kmeansEntropy = calcEntropy(kmeansMatrix)
dbscanEntropy = calcEntropy(dbscanMatrix)

kmeansPurity = calcPurity(kmeansMatrix)
dbscanPurity = calcPurity(dbscanMatrix)

kmeansSSE = calcSSE(kmeansResult)
dbscanSSE = calcSSE(dbscanResult)

finalResult = []
finalResult.append(kmeansSSE)
finalResult.append(dbscanSSE)
finalResult.append(kmeansEntropy)
finalResult.append(dbscanEntropy)
finalResult.append(kmeansPurity)
finalResult.append(dbscanPurity)

pd.DataFrame([finalResult]).to_csv("Result.csv", header=None, index=False)