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
FOLDER_NAMES_EXTRA = ["TestDim600ep250batch256lr1e-3Noise20Dataset10", "TestDim700ep250batch256lr1e-3Noise20Dataset10", "TestDim800ep250batch256lr1e-3Noise20Dataset10",
                      "TestDim900ep250batch256lr1e-3Noise20Dataset10", "TestDim1000ep250batch256lr1e-3Noise20Dataset10", "TestDim1500ep250batch256lr1e-3Noise20Dataset10", 
                      "TestDim2000ep250batch256lr1e-3Noise20Dataset10", "TestDim2500ep250batch256lr1e-3Noise20Dataset10", "TestDim3000ep250batch256lr1e-3Noise20Dataset10",
                      "TestDim3500ep250batch256lr1e-3Noise20Dataset10", "TestDim4000ep250batch256lr1e-3Noise20Dataset10", "TestDim4500ep250batch256lr1e-3Noise20Dataset10",
                      "TestDim5000ep250batch256lr1e-3Noise20Dataset10"]
FOLDER_NAME_REG = "TestDim1-500ep250batch256lr1e-3Noise20Dataset10Reg1"
RANKS = [4,6,10,20,50,100]
EPOCH = 250
EARLY_STOPPING = True
MEMORY_COST = True

plt.rcParams["figure.figsize"] = (16,9)
plt.rcParams.update({'font.size': 22})

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
            start = i-5
            end = i+5
            if start < 0:
                start = 0
            if i == 0:
                end = 1
            if end >= len(allTests[0]):
                end = len(allTests[0])-1
            for el in allTests[0][start:end]:
                if EARLY_STOPPING:
                    summe += min(el)
                else:
                    summe += el[EPOCH-1]
                j += 1
            summe /= j
            errors.append(summe)
            if MEMORY_COST:
                xValues.append(RANKS[count]*(2*i+RANKS[count])*16*1e-6)
            else:
                xValues.append(((RANKS[count])**2)*2*i)
    if MEMORY_COST:
        xValues[0] = ((RANKS[count])**2)*16*1e-6
    else:
        xValues[0] = RANKS[count]**2
    allErrors.append(errors)
    allxValues.append(xValues)
    print("1 Number " + str(count+1))

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
    if EARLY_STOPPING:
        allErrors[i][0] = min(allTests[0][RANKS[i]-1])
    else:
        allErrors[i][0] = allTests[0][RANKS[i]-1][EPOCH-1]
        
extraxValues = []
extrayValues = []
for i in range(2, 6):
    size = 100*i
    if MEMORY_COST:
        extraxValues.append((size**2)*16*1e-6)
    else:
        extraxValues.append((size**2))
    if EARLY_STOPPING:
        extrayValues.append(min(allTests[0][size-1]))
    else:
        extrayValues.append(allTests[0][size-1][EPOCH-1])
print("2 Number 1")


extraxxValues = []
extrayyValues = []
for count, el in enumerate(FOLDER_NAMES_EXTRA):
    tests = []
    trains = []
    allTests = []
    allTrains = []
    names = []    
    leng = 0
    lengFiles = 1
    epochs = 0
    linkToRun = "finishedTests\weitereDim\\" + el
    size = (count+6)*100
    if count > 4:
        size = (count-4)*500+1000
    
    for path in os.listdir(linkToRun):
        leng += 1
    
    for j in range(1,leng+1):
        link = linkToRun + "\Run" + str(j) + "\historyLogs\history_001_.csv"
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
                    
    if MEMORY_COST:
        extraxxValues.append((size**2)*16*1e-6)
    else:
        extraxxValues.append(size**2)
    if EARLY_STOPPING:
        extrayyValues.append(min(allTests[0][0]))
    else:
        extrayyValues.append(allTests[0][0][EPOCH-1])
    print("3 Number " + str(count+1))

        
fullRankx = []
fullRanky = []
for i in range(len(allxValues)):
    fullRankx.append(allxValues[i][0])
    fullRanky.append(allErrors[i][0])    
for i in range(len(extraxValues)):
    fullRankx.append(extraxValues[i])
    fullRanky.append(extrayValues[i])
for i in range(len(extraxxValues)):
    fullRankx.append(extraxxValues[i])
    fullRanky.append(extrayyValues[i])

lowRankx = []
lowRanky = []
for i in range(len(allxValues)):
    lowRankx.append(allxValues[i][-8])
    lowRanky.append(allErrors[i][-8])
    
    
tests = []
trains = []
allTests = []
allTrains = []
names = []    
leng = 0
lengFiles = 0
epochs = 0
linkToRun = "finishedTests\\" + FOLDER_NAME_REG

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
        
extraxValuesReg = []
extrayValuesReg = []
for i in range(3, 11):
    size = 50*i
    if MEMORY_COST:
        extraxValuesReg.append((size**2)*16*1e-6)
    else:
        extraxValuesReg.append((size**2))
    if EARLY_STOPPING:
        extrayValuesReg.append(min(allTests[0][size-1]))
    else:
        extrayValuesReg.append(allTests[0][size-1][EPOCH-1])
print("4 Number 1")



plt.plot(fullRankx, fullRanky, '-o', color="black", label="Reference network")
plt.plot(extraxValuesReg, extrayValuesReg, '-o', color="cyan", label="Regularized \u03BB=0.01")
plt.plot(lowRankx, lowRanky, '--o', color="red", label="Width 150")

colors = ["blue", "orange", "green", "purple", "yellow", "brown"]
for i in range(len(allErrors)):
    plt.plot(allxValues[i][1:], allErrors[i][1:], '-o', color=colors[i], label="Rank " + str(RANKS[i]))
    #plt.plot(allxValues[i][0], allErrors[i][0], '-o', color=colors[i], markersize=10)

"""
colors2 = ["pink", "yellow", "olive", "cyan"]
for i in range(2, 6):
    plt.plot(extraxValues[i-2], extrayValues[i-2], '-o', color=colors2[i-2], markersize=10, label="Rank " + str(i*100))

colors3 = ["black", "dodgerblue", "greenyellow", "gray", "peru"]    
for i in range(len(extraxxValues)):
    plt.plot(extraxxValues[i], extrayyValues[i], '-o', markersize=10, label="Rank " + str((i+6)*100))
"""

if EARLY_STOPPING:
    plt.title("The error of low rank solutions with early stopping")
else:
    plt.title("The error of low rank solutions at epoch " + str(EPOCH))
plt.ylabel("Test Error")
if MEMORY_COST:
    plt.xlabel("Memory footprint (MB)")
else:
    plt.xlabel("Computational cost (operations)")
plt.xscale("log")
plt.legend()
plt.show()






