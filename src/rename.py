# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:43:15 2023

@author: jakob
"""

import os

def getNumber(i):
    if i < 10:
        return "00" + str(i)
    if i < 100:
        return "0" + str(i)
    return str(i)

for i in reversed(range(1, 496)):
    name = "history_" + getNumber(i) + "_.csv"
    new = "history_" + getNumber(i+5) + "_.csv"
    os.rename(name, new)