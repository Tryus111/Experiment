#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:30:20 2023

@author: user
"""

import pandas as pd
import numpy as np

raw_data = pd.read_csv('project-Nasa.csv')
df_rain = raw_data.replace(-999, np.nan)

# Drop missing value
df_rain = df_rain.dropna()

# Date
date  = pd.concat([df_rain['YEAR'], df_rain['MO'], df_rain['DY']], axis = 1)
date['DATE'] = date['YEAR'].astype(str) + date['MO'].astype(str).str.zfill(2) + date['DY'].astype(str).str.zfill(2)
date['DATE'] = pd.to_datetime(date['DATE'], format = '%Y %m %d')

# Concat date and data
df_no_date = df_rain.drop(df_rain[['YEAR', 'MO', 'DY']], axis = 1)
df_preprocessing = pd.concat([date['DATE'], df_no_date], axis = 1)

# Save file
df_preprocessing.to_csv('data_cleaned.csv', index = False)

