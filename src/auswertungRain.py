# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 14:22:09 2023

@author: jakob
"""

import matplotlib.pyplot as plt
import os
import copy

FOLDER_NAMES = ["DLRT(Rank4)TestDim1-500ep250batch256lr1e-3Noise20Dataset10", "DLRT(Rank6)TestDim1-500ep250batch256lr1e-3Noise20Dataset10", "DLRT(Rank10)TestDim1-500ep250batch256lr1e-3Noise20Dataset10", "DLRT(Rank20)TestDim1-500ep250batch256lr1e-3Noise20Dataset10", "DLRT(Rank50)TestDim1-500ep250batch256lr1e-3Noise20Dataset10","DLRT(Rank100)TestDim1-500ep250batch256lr1e-3Noise20Dataset10"]
FOLDER_NAME_FR = "TestDim1-500ep250batch256lr1e-3Noise20Dataset10"
RANKS = [4,6,10,20,50,100]
EPOCH = 250
LOG_SCALE = True

plt.rcParams["figure.figsize"] = (16,9)

allErrors = []
allxValues = []

for count, el in enumerate(FOLDER_NAMES):
    tests = []
    trains = []
    allTests = []
    allTrains = []
    names = []    
    leng = 0
    lengFiles = 0
    epochs = 0
    linkToRun = "finishedTests\\" + el

    for path in os.listdir(linkToRun):
        leng += 1
    
    for path in os.listdir(linkToRun + "\Run1\historyLogs"):
        names.append(path)
        lengFiles += 1
    
    for j in range(1,leng+1):
        for i in range(lengFiles):
            link = linkToRun + "\Run" + str(j) + "\historyLogs\\" + names[i]
            with open(link, "r") as file:
                lines = file.readlines()
            
            rows = []
            test = []
            train = []
                        
            for line in lines:
                rows.append(line.split(";"))
            
            for row in rows:
                try:
                    test.append(1-float(row[5]))
                    train.append(1-float(row[1]))
                except:
                    continue
                
            tests.append(test)
            trains.append(train)
            
            epochs = len(test)
            
        allTests.append(tests)
        allTrains.append(trains)
        tests = []
        trains = []
    
    allTests.append(copy.deepcopy(allTests[0]))
    allTrains.append(copy.deepcopy(allTrains[0]))
    
    for h in range(1, leng):
        for k in range(len(allTests[0])):
            for l in range(len(allTests[0][k])):
                allTests[0][k][l] += allTests[h][k][l]
                allTrains[0][k][l] += allTrains[h][k][l]
                if h == leng-1:
                    allTests[0][k][l] /= leng
                    allTrains[0][k][l] /= leng
    
    errors = []
    xValues = []
    for i in range(len(allTests[0])):
        if (i+RANKS[count]) % 50 == 0 or i == 0:
            summe = 0
            j = 0
            start = i-25
            end = i+25
            if start < 0:
                start = 0
            if i == 0:
                end = 1
            if end >= len(allTests[0]):
                end = len(allTests[0])-1
            for el in allTests[0][start:end]:
                summe += el[EPOCH-1]
                j += 1
            summe /= j
            errors.append(summe)
            xValues.append(RANKS[count]*(0+2*(i+RANKS[count]))*16*1e-6)
    xValues[0] = ((RANKS[count])**2)*16*1e-6
    allErrors.append(errors)
    allxValues.append(xValues)

tests = []
trains = []
allTests = []
allTrains = []
names = []    
leng = 0
lengFiles = 0
epochs = 0
linkToRun = "finishedTests\\" + FOLDER_NAME_FR

for path in os.listdir(linkToRun):
    leng += 1

for path in os.listdir(linkToRun + "\Run1\historyLogs"):
    names.append(path)
    lengFiles += 1

for j in range(1,leng+1):
    for i in range(lengFiles):
        link = linkToRun + "\Run" + str(j) + "\historyLogs\\" + names[i]
        with open(link, "r") as file:
            lines = file.readlines()
        
        rows = []
        test = []
        train = []
                    
        for line in lines:
            rows.append(line.split(";"))
        
        for row in rows:
            try:
                test.append(1-float(row[5]))
                train.append(1-float(row[1]))
            except:
                continue
            
        tests.append(test)
        trains.append(train)
        
        epochs = len(test)
        
    allTests.append(tests)
    allTrains.append(trains)
    tests = []
    trains = []

allTests.append(copy.deepcopy(allTests[0]))
allTrains.append(copy.deepcopy(allTrains[0]))

for h in range(1, leng):
    for k in range(len(allTests[0])):
        for l in range(len(allTests[0][k])):
            allTests[0][k][l] += allTests[h][k][l]
            allTrains[0][k][l] += allTrains[h][k][l]
            if h == leng-1:
                allTests[0][k][l] /= leng
                allTrains[0][k][l] /= leng
                
for i in range(len(RANKS)):
    allErrors[i][0] = allTests[0][RANKS[i]-1][EPOCH-1]

colors = ["black", "red", "green", "yellow", "orange", "pink", "brown"]
for i in range(len(allErrors)):
    plt.plot(allxValues[i][1:], allErrors[i][1:], '-o', color=colors[i], label="Rank " + str(RANKS[i]))
    plt.plot(allxValues[i][0], allErrors[i][0], '-o', color=colors[i], markersize=10)

plt.title("The error of low rank solutions at epoch " + str(EPOCH))
plt.ylabel("Test Error")
plt.xlabel("Memory footprint (MB)")
plt.legend()
plt.show()






