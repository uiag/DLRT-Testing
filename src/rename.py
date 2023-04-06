# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:43:15 2023

@author: jakob
"""

import os

for i in reversed(range(305, 501)):
    name = "history_" + str(i) + "_.csv"
    new = "history_" + str(i-1) + "_.csv"
    os.rename(name, new)