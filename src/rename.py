# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:09:27 2023

@author: jakob
"""

import os
import shutil

def getName(i):
    if i < 10:
        return "history_00"+str(i)+"_.csv"
    elif i < 100:
        return "history_0"+str(i)+"_.csv"
    else:
        return "history_" + str(i)+"_.csv"

folders = os.listdir()
folders = folders[:-1]
"""
for el in folders:
    i = int(el.split("-")[0].split("m")[1])
    for j in reversed(range(1, len(os.listdir(el + "/historyLogs"))+1)):
        name = el + "/historyLogs/" + getName(j)
        newName = el + "/historyLogs/" + getName(j+i-1)
        os.rename(name, newName)
"""
os.mkdir("Run")
os.mkdir("Run/historyLogs")
for el in folders:
    for file_name in os.listdir(el + "/historyLogs"):
        source = el + "/historyLogs/" + file_name
        destination = "Run/historyLogs/" + file_name
        shutil.copy(source, destination)
