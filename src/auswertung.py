# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 14:22:09 2023

@author: jakob
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import copy
from matplotlib.colors import LogNorm

FOLDER_NAME = "TestDim491-501ep250batch256lr0.001Noise20.0Dataset10.0RegL1-0.1Run1"
SHOW_ALL_DIM = False
SHOW_SINGLE_DIM = 250
SHOW_SINGLE_EPOCH = 150
SHOW_ALL_EPOCH = False
MARK_STD_ABOVE = 0.075
LOG_SCALE_X_Y_Z = [False, False, False]

plt.rcParams["figure.figsize"] = (16,9)

def heatmap2d(arr: np.ndarray, title, log_scale):
    if log_scale[2]:
        plt.imshow(arr, cmap='plasma', norm=LogNorm())
    else:
        plt.imshow(arr, cmap='plasma')
    if log_scale[1]:
        plt.ylim([500,1])
        plt.yscale('log', base=1.013)
        plt.yticks([2,4,8,16,32,64,128,256,500],[2,4,8,16,32,64,128,256,500])
    if log_scale[0]:
        plt.xlim([1,250])
        plt.xscale('log', base=1.022)
        plt.xticks([2,4,8,16,32,64,128,250],[2,4,8,16,32,64,128,250])
    plt.title(title)
    plt.ylabel("Width")
    plt.xlabel("Epochs")
    plt.colorbar()
    plt.show()
    
tests = []
trains = []
allTests = []
allTrains = []
names = []    
linkToRun = "finishedTests\\" + FOLDER_NAME
leng = 0
lengFiles = 0
epochs = 0

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

allDeviations = []
deviationsList = []
for k in range(len(allTests[0])):
    deviations = []
    for l in range(len(allTests[0][k])):
        mu = 0
        deviation = 0
        for h in range(1, leng+1):
            mu += allTests[h][k][l]
        mu /= leng
        for h in range(1, leng+1):
            deviation += ((mu - allTests[h][k][l])**2)
        deviation /= leng
        deviation = np.sqrt(deviation)
        deviation /= mu
        deviations.append(deviation)
        deviationsList.append(deviation)
    allDeviations.append(deviations)
    
sumDev = 0
lenDev = 0
maxDev = 0
minDev = 100
maxy = 0
maxx = 0
miny = 0
minx = 0
for count, dev in enumerate(allDeviations):
    if max(dev) > maxDev:
        maxDev = max(dev)
        maxx = dev.index(maxDev)
        maxy = count
    if min(dev) < minDev:
        minDev = min(dev)
        minx = dev.index(minDev)
        miny = count
    sumDev += sum(dev)
    lenDev += len(dev)
avgDev = sumDev/lenDev

print(str(leng) + " Runs containing " + str(lengFiles) + " dimensions each with " + str(epochs) + " epochs each found and analyzed.")
print("Total amount of analyzed datapoints: ", leng*lengFiles*epochs*6)
print("Average standard deviation: " + str(avgDev))
print("Maximum standard deviation: " + str(maxDev) + " at Dim " + str(maxy+1) + " and epoch " + str(maxx+1))
print("Minimum standard deviation: " + str(minDev) + " at Dim " + str(miny+1) + " and epoch " + str(minx+1))
print("Quartile: ", np.quantile(deviationsList, q = np.arange(0.25, 1.25, 0.25)))

heatmap2d(allTests[0], "Test Error", LOG_SCALE_X_Y_Z)
heatmap2d(allTrains[0], "Train Error", LOG_SCALE_X_Y_Z)

if SHOW_ALL_DIM == True:
    for i in range(len(allTests[0])):
        plt.plot(allTests[0][i])
        plt.plot(allTrains[0][i])
        plt.ylabel("Test/Train Error")
        plt.xlabel("Epochs")
        plt.title("Width " + str(i+1))
        plt.show()

elif SHOW_SINGLE_DIM != 0:
    plt.plot(allTests[0][SHOW_SINGLE_DIM-1])
    plt.plot(allTrains[0][SHOW_SINGLE_DIM-1])
    plt.ylabel("Test/Train Error")
    plt.xlabel("Epochs")
    plt.title("Width " + str(SHOW_SINGLE_DIM))
    plt.show()

if SHOW_SINGLE_EPOCH != 0:
    dimsTest = []
    dimsTrain = []
    dimsTestAll = []
    dimsTrainAll = []
    for j in range(len(allTests[0][0])):
        for i in range(len(allTests[0])):
            dimsTest.append(allTests[0][i][j])
            dimsTrain.append(allTrains[0][i][j])
        dimsTestAll.append(dimsTest)
        dimsTest = []
        dimsTrainAll.append(dimsTrain)
        dimsTrain = []
    if SHOW_ALL_EPOCH == True:
        for i in range(len(dimsTestAll)):
            plt.plot(dimsTestAll[i])
            plt.plot(dimsTrainAll[i])
            plt.ylabel("Test/Train Error")
            plt.xlabel("Width")
            plt.title("Epoch " + str(i+1))
            plt.show()
    else:
        plt.plot(dimsTestAll[SHOW_SINGLE_EPOCH-1])
        plt.plot(dimsTrainAll[SHOW_SINGLE_EPOCH-1])
        plt.ylabel("Test/Train Error")
        plt.xlabel("Width")
        plt.title("Epoch " + str(SHOW_SINGLE_EPOCH))
        plt.show()

heatmap2d(allDeviations, "Standard Deviation", LOG_SCALE_X_Y_Z)
for h in range(len(allDeviations)):
    for i in range(len(allDeviations[h])):
        if allDeviations[h][i] > MARK_STD_ABOVE:
            allDeviations[h][i] = 1
        else:
            allDeviations[h][i] = 0
heatmap2d(allDeviations, "Standard Deviation above " + str(MARK_STD_ABOVE), LOG_SCALE_X_Y_Z)





