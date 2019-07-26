# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:26:47 2017

@author: Payam
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import sklearn.linear_model as linmod

# Obtaining the path to the current python script
dir_path = os.path.dirname(os.path.realpath(__file__))

# Loading the csv file into mdata dataframe
mdata = pd.read_csv(dir_path + '/data/main_data.csv')
mdata.rename(columns = {'Date/Time': 'DateTime'}, inplace = True)
mdata.head()
mdata.tail()

# Checking if there is any null values in any column
mdata.isnull().any()







