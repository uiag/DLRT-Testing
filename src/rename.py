# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:09:27 2023

@author: jakob
"""

import os

def getName(i):
    if i < 10:
        return "history_00"+str(i)+"_"
    elif i < 100:
        return "history_0"+str(i)+"_"
    else:
        return "history_" + str(i)+"_"

for i in reversed(range(50)):
    name = getName(i)
    newName = getName(i+5)
    os.rename(name, newName)