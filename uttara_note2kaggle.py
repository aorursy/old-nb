import pandas as pd

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print(train.describe())

import pandas as pd
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
train[(train["SchoolHoliday"] == 1) & (train["Open"] == 1) & (train["Promo"] == 0) ].describe()
regr = linear_model.LinearRegression()
#regr.fit(np.array(train), diabetes_y_train)


store_ids = set(train["Store"])
    
