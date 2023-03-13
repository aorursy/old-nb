# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
	
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
	
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
	
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
	
# Any results you write to the current directory are saved as output.
	
animals = pd.read_csv('../input/train.csv')
	
otv = animals["OutcomeType"].value_counts()
print(otv)
	
labels = 'Adoption', 'Transfer', 'Return to Owner', 'Euthanasia', 'Died'
sizes = [10769, 9422, 4786, 1555, 197]
colors = ['yellowgreen', 'mediumpurple', 'lightskyblue', 'lightcoral', 'orange'] 
explode = (0, 0, 0, 0,0)    # proportion with which to offset each wedge
	
plt.pie(sizes,              # data
	        explode=explode,    # offset parameters 
	        labels=labels,      # slice labels
	        colors=colors,      # array of colours
	        autopct='%1.1f%%',  # print the values inside the wedges
	        shadow=True,        # enable shadow
	        startangle=70       # starting angle
)

plt.axis('equal')
	
plt.show()



