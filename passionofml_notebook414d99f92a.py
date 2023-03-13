#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




# Lets check the train folder
print(check_output(["ls","../input/train"]).decode("utf8"))




check_output(["ls", "../input/train/ALB"]).decode("utf8")




sub_folders = check_output(["ls", "../input/train"]).decode("utf8").strip().split('\n')
folderdict = {}
print(sub_folders)
for sub_folder in sub_folders:
    print("folders Name:", sub_folder)
    files = check_output(["ls", "../input/train/"+sub_folder]).decode("utf8").strip().split('\n')
    folderdict[sub_folder] = len(files)
    print("number of images:", folderdict[sub_folder])

