import pandas as pd
import numpy as np
from datetime import datetime
cgmData = pd.read_csv("CGMData.csv")
islData = pd.read_csv("InsulinData.csv")

def transform_time(date,time):
    dateTime = date + " " + time
    dateTime = datetime.strptime(dateTime,"%m/%d/%Y %H:%M:%S")
    return dateTime

date = islData[islData["Alarm"]=="AUTO MODE ACTIVE PLGM OFF"]["Date"].values[-1]
time = islData[islData["Alarm"]=="AUTO MODE ACTIVE PLGM OFF"]["Time"].values[-1]
autoStartTime = transform_time(date,time)
newDates = islData.apply(lambda row:transform_time(row["Date"],row["Time"]),axis=1)
cgmTime = cgmData.apply(lambda row:transform_time(row["Date"],row["Time"]),axis=1)
cgmData["dateTime"] = cgmTime
cgmAutoTime = cgmData[cgmData["dateTime"] > autoStartTime]["dateTime"].values[-1]
manualData = cgmData[cgmData["dateTime"] < cgmAutoTime]
autoData = cgmData[cgmData["dateTime"] >= cgmAutoTime]

def get_day(datetime):
    return datetime.day

def get_hour(datetime):
    return datetime.hour

manualData["day"] = manualData.apply(lambda row:get_day(row["dateTime"]),axis=1)
manualData["hour"] = manualData.apply(lambda row:get_hour(row["dateTime"]),axis=1)
autoData["day"] = autoData.apply(lambda row:get_day(row["dateTime"]),axis=1)
autoData["hour"] = autoData.apply(lambda row:get_hour(row["dateTime"]),axis=1)

def handle_missingData(data):
    dates = data["Date"].unique()
    removalList = []
    for date in dates:
        subData = data.groupby("Date").get_group(date)
        length = len(subData)
        numMissing = subData["Sensor Glucose (mg/dL)"].isnull().sum()
        missRatio = numMissing / length
        if missRatio <= 0.2:
            continue
        else:
            removalList.append(subData["Date"].values[0])
    finalData = data.copy()
    for removalDate in removalList:
        index = finalData[finalData["Date"]==removalDate].index
        finalData = finalData.drop(index)
    return finalData

modifiedManual = handle_missingData(manualData)
modifiedAuto = handle_missingData(autoData)

def getPerc(data,isWhole=False):
    range1 = data[data["Sensor Glucose (mg/dL)"] > 180]
    range2 = data[data["Sensor Glucose (mg/dL)"] > 250]
    range3 = data[data["Sensor Glucose (mg/dL)"].between(70,180)]
    range4 = data[data["Sensor Glucose (mg/dL)"].between(70,150)]
    range5 = data[data["Sensor Glucose (mg/dL)"] < 70]
    range6 = data[data["Sensor Glucose (mg/dL)"] < 54]
    if isWhole:
        return len(range1)/288,len(range2)/288,len(range3)/288,len(range4)/288,len(range5)/288,len(range6)/288
    else:
        if len(data) == 0:
            return 0,0,0,0,0,0
        else:
            return len(range1)/len(data),len(range2)/len(data),len(range3)/len(data),len(range4)/len(data),len(range5)/len(data),len(range6)/len(data)


def getMetric(data):
    dates = data["Date"].unique()
    overnightMetric = [[0],[0],[0],[0],[0],[0]]
    daytimeMetric = [[0],[0],[0],[0],[0],[0]]
    wholeMetric = [[0],[0],[0],[0],[0],[0]]
    for date in dates:
        subData = data.groupby("Date").get_group(date)
        overnight = subData[subData["hour"].between(0,6)]
        daytime = subData[subData["hour"].between(6,23)]
        whole = subData
        overnightPerc = list(getPerc(overnight))
        daytimePerc = list(getPerc(daytime))
        wholePerc = list(getPerc(whole,True))
        for i in range(6):
            overnightMetric[i].append(overnightPerc[i])
            daytimeMetric[i].append(daytimePerc[i])
            wholeMetric[i].append(wholePerc[i])
    metrics = [overnightMetric,daytimeMetric,wholeMetric]
    answer = []
    for metric in metrics:
        for i in range(6):
            answer.append(sum(metric[i]) / len(dates))
    return answer

manualResult = np.array(getMetric(modifiedManual))
autoResult = np.array(getMetric(modifiedAuto))
result = pd.DataFrame([manualResult,autoResult])
result.to_csv("Results.csv",index=0,header=0)
