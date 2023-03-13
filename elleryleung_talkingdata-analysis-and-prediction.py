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



# http://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
# http://stackoverflow.com/questions/22258491/read-a-small-random-sample-from-a-big-csv-file-into-a-python-data-frame
event_file_path = "../input/events.csv"
nlinesfile = 3252951
#nlinesfile = file_len(event_file_path)
nlinesrandomsample = 200000
lines2skip = np.random.choice(np.arange(1,nlinesfile+1), (nlinesfile-nlinesrandomsample), replace=False)

# events.csv, app_events.csv - when a user uses TalkingData SDK, the event gets logged in this data. 
# Each event has an event id, location (lat/long), and the event corresponds to a list of apps in app_events.
# This is a ~190MB file
event = pd.read_csv(event_file_path, skiprows=lines2skip, encoding="utf-8", header=0)

app_events_file_path = "../input/app_events.csv"
nlinesfile = 32473068
#nlinesfile = file_len(app_events_file_path)
nlinesrandomsample = 200000
lines2skip = np.random.choice(np.arange(1,nlinesfile+1), (nlinesfile-nlinesrandomsample), replace=False)

# This is a ~1GB file
app_events = pd.read_csv(app_events_file_path, skiprows=lines2skip, encoding="utf-8", header=0)

# gender_age_train.csv, gender_age_train.csv - the training and test set
# group: this is the target variable you are going to predict
gender_age_train = pd.read_csv("../input/gender_age_train.csv", encoding="utf-8")
gender_age_test = pd.read_csv("../input/gender_age_test.csv", encoding="utf-8")

# app_labels.csv - apps and their labels, the label_id's can be used to join with label_categories
# label_categories.csv - apps' labels and their categories in text
# phone_brand_device_model.csv - device ids, brand, and models
app_labels = pd.read_csv("../input/app_labels.csv", encoding="utf-8")
label_categories = pd.read_csv("../input/label_categories.csv", encoding="utf-8")
phone_brand_device_model = pd.read_csv("../input/phone_brand_device_model.csv", encoding="utf-8")

app_labels_label_categories = pd.merge(app_labels, label_categories, how="inner", on="label_id")
app_events_app_labels_label_categories = pd.merge(app_events, app_labels_label_categories, how="left", on="app_id")
event_app_events_app_labels_label_categories = pd.merge(event, app_events_app_labels_label_categories, how="inner", on="event_id")
gender_age_train_event_app_events_app_labels_label_categories = pd.merge(gender_age_train, event_app_events_app_labels_label_categories, 
                                                                   how="inner", on="device_id")
gender_age_test_event_app_events_app_labels_label_categories = pd.merge(gender_age_test, event_app_events_app_labels_label_categories, 
                                                                   how="inner", on="device_id")
train = gender_age_train_event_app_events_app_labels_label_categories
test = gender_age_test_event_app_events_app_labels_label_categories

cols = train['group'].unique()
cols.sort()
cols = dict(enumerate(cols))

# Convert group to number
def set_num_group(group):
    for i, v in cols.items():
        if(v == group):
            return i
        
train['group'] = train['group'].apply(set_num_group)

# Convert gender to number
train['gender'] = train['gender'].apply(lambda x: 1 if(x == 'M') else 0)

#Convert category
cat_col = train['category'].unique()
cat_col = dict(enumerate(cat_col))

# Convert group to number
def set_num_category(cat):
    for i, v in cat_col.items():
        if(v == cat):
            return i
        
train['category'] = train['category'].apply(set_num_category)

#Convert datetime        
train['timestamp'] = train['timestamp'].apply(lambda x: pd.to_datetime(x).value)
y = train['group']

# Train the model
from sklearn import svm
svc = svm.SVC(kernel='linear', C=1.0).fit(train, y)

# Test for accuracy
results = svc.predict(gender_age_test)
num_correct = (results == test_n_group).sum()
recall = num_correct / len(test_n_group)
print "model accuracy (%): ", recall * 100, "%"
