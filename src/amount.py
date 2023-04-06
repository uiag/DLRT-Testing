# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 21:51:30 2023

@author: jakob
"""

import os

link = "finishedTests"
pathAmount = []
pathCount = 0

for path in os.listdir(link):
    countRuns = 0
    runs = 0
    for run in os.listdir(link + "\\" + path):
        countRuns += 1
        countExcel = 0
        excelName = ""
        for excel in os.listdir(link + "\\" + path + "\Run1\historyLogs"):
            countExcel += 1
            excelName = excel 
        link2 = link + "\\" + path + "\Run1\historyLogs\\" + excelName    
        with open(link2, "r") as file:
            lines = file.readlines()
        
        rows = []
        test = []
                    
        for line in lines:
            rows.append(line.split(";"))
        
        for row in rows:
            try:
                test.append(1-float(row[5]))
            except:
                continue
        epochs = len(test)
        
        runs = countExcel*epochs
    pathAmount.append(runs*countRuns)
    pathCount += 1

total = 0
for el in pathAmount:
    total += el

print("Total Count of Tests: ", pathCount)
print("Total Datapoints: ", total*6)    