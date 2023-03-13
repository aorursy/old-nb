#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import datetime
from kaggle.competitions import nflrush
import tqdm
import re
from string import punctuation
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
import keras
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 150)

sns.set_style('darkgrid')
mpl.rcParams['figure.figsize'] = [15,10]




env = nflrush.make_env()




train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})




train.head()




train.describe()




train.select_dtypes(include=["float", 'int']).columns




train.select_dtypes(exclude=["float", 'int']).columns




train['PlayId'].value_counts().describe()




train['Position'].value_counts()




train['Yards'].describe()




ax = sns.distplot(train['Yards'])
plt.vlines(train['Yards'].mean(), plt.ylim()[0], plt.ylim()[1], color='r', linestyles='--');
plt.text(train['Yards'].mean()+5, plt.ylim()[1]-0.02, "Mean yards travaled", size=15, color='r')
plt.title("Yards travaled distribution", size=20);




cat_features = []
int_features = []
float_features = []
bool_features = []
uint8_features = []
for col in train.columns:
    #print(train[col].dtype) #print out to check datatypes
    if train[col].dtype =='object':
        cat_features.append((col, len(train[col].unique())))
    if train[col].dtype =='int64':
        int_features.append((col, len(train[col].unique())))
    if train[col].dtype =='float64':
        float_features.append((col, len(train[col].unique())))
    if train[col].dtype =='bool':
        bool_features.append((col, len(train[col].unique())))
    if train[col].dtype =='uint8':
        uint8_features.append((col, len(train[col].unique())))
        
''' contained datatypes
int64
object
float64
'''

print("{} Categorical Features found.".format(len(cat_features)))
print("{} Integer Features found.".format(len(int_features)))
print("{} Float Features found.".format(len(float_features)))
print("{} Boolean Features found.".format(len(bool_features)))
print("{} uint8 OneHot Features found.".format(len(uint8_features)))




cat_features




def convert_to_seconds(txt): #converts string to number of seconds. range = 0-15min 0-900sec
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
    return ans




def classify_offense_formation(txt): #can be extended
    if pd.isna(txt):
        return "unknown"
    txt = txt.lower()
    if (txt.find("single") != -1): #singleback
        return "singleback"
    elif (txt.find("shot") != -1): #shotgun
        return "shotgun"
    elif (txt.find("i_form") != -1): #I_Formation
        return "i_form"
    elif (txt.find("pistol") != -1): #Pistol
        return "pistol"
    elif (txt.find("jumbo") != -1): #Jumbo
        return "jumbo"
    elif (txt.find("wildcat") != -1): #Wildcat
        return "wildcat"
    elif (txt.find("empty") != -1): #Empty
        return "empty"
    elif (txt.find("ace") != -1): #Ace
        return "ace"
    else:
        return "unknown"




offense_dict = {'DB' : "offense_DB", 'DL' : "offense_DL", 
                'LB' : "offense_LB", 'OL' : "offense_OL", 
                'QB' : "offense_QB", 'RB' : "offense_RB", 
                'TE' : "offense_TE", 'WR' : "offense_WR"}

def ProcessOffensePersonnel(row):
    for whole_offense in row["OffensePersonnel"].split(","):
        #print("Offense Item: {}".format(whole_offense))
        offense_item = whole_offense.strip().split(" ")
        if (offense_item[1] in offense_dict.keys()):
            if offense_item[0].isdigit():
                row[offense_dict[offense_item[1]]] = int(offense_item[0])
            else:
                row[offense_item[offense_item[1]]] = 0
        else:
            if offense_item[0].isdigit():
                row["offense_UNKNOWN"] += int(offense_item[0])
            else:
                row["offense_UNKNOWN"] = 0
    return row




defense_dict = {'DB' : "defense_DB", 'DL' : "defense_DL", 
                'LB' : "defense_LB", 'OL' : "defense_OL"}

def ProcessDefensePersonnel(row):
    for whole_defense in row["DefensePersonnel"].split(","):
        defense_item = whole_defense.strip().split(" ")
        if (defense_item[1] in defense_dict.keys()):
            if defense_item[0].isdigit():
                row[defense_dict[defense_item[1]]] = int(defense_item[0])
            else:
                row[defense_dict[defense_item[1]]] = 0
        else:
            if defense_item[0].isdigit():
                row["defense_UNKNOWN"] += int(defense_item[0])
            else:
                row["defense_UNKNOWN"] = 0
    return row




#convert to boolean outdoor = True, indoow = False
def classify_stadium_type(txt):
    if pd.isna(txt):
        return True
    txt = txt.lower()
    if (txt.find("ou") != -1):
        return True
    if (txt.find("in") != -1):
        return False
    if (txt.find("op") != -1):
        return True
    if (txt.find("cl") != -1):
        return False
    return True




#convert to boolean natural grass = True, artificial = False
def classify_turf_type(txt):
    if pd.isna(txt):
        print("isna")
        return True
    txt = txt.lower()
    if (txt.find("gras") != -1):
        return True
    if (txt.find("nat") != -1):
        return True
    #print("false")
    return False




def classify_weather(txt):
    if pd.isnull(txt):
        #print("isnull")
        return "cloudy" #default
    txt = txt.lower()
    if ((txt.find("sun") != -1) or (txt.find("cle") != -1)): #sunny or clear
        return "sunny"
    elif ((txt.find("oud") != -1) or (txt.find("clo") != -1)): #cloudy
        return "cloudy"
    elif ((txt.find("rai") != -1) or (txt.find("sho") != -1)): #rain or shower
        return "rainy"
    elif ((txt.find("indo") != -1) or (txt.find("clim") != -1)): #indoor climatized
        return "indoor"
    elif (txt.find("sno") != -1): #snow
        return "snow"
    else:
        return "cloudy" #default




def new_X(x_coordinate, play_direction): #play direction boolean (right=True / left=False)
    if play_direction == False: #direction = left --> switch play direction, so play direction is always from left to right
        return 120.0 - x_coordinate
    else:
        return x_coordinate #direction = right --> ok


def new_orientation(angle, play_direction): #play direction boolean (right=True / left=False)
    if play_direction == False: #direction = left --> field was switched --> turn player around
        new_angle = 360.0 - angle
        if new_angle == 360.0:
            return 0.0
        else:
            return new_angle
    else:
        if angle == 360.0:
            return 0.0
        else:
            return angle

        
def line_of_scrimmage_in_coordinates(rush_team, field_position, yardline):
    #rushin team is always from left to right
    if rush_team == field_position:
        # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
        return 10.0 + yardline
    else:
        # half the field plus the yards between midfield and the line of scrimmage
        return 60.0 + (50 - yardline)

    
def euclidean_distance(x1,y1,x2,y2):
    x_diff = (x1-x2)**2
    y_diff = (y1-y2)**2
    return np.sqrt(x_diff + y_diff)


def is_looking_back(orientation): #looking back means from right to left, means direction > 180
    if orientation > 180.0:
        return True
    else:
        return False
    
def recalculate_yardline(df): #transforms YardLine to global field coordinates
    new_yardline = df[df['NflId'] == df['NflIdRusher']].copy()
    #enthält je game einen rusher
    new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: line_of_scrimmage_in_coordinates(x[0],x[1],x[2]), axis=1)
    new_yardline = new_yardline[['GameId','PlayId','YardLine']]
    #enthält je play die neu berechnete yardline
    df = df.drop('YardLine', axis=1)
    df = pd.merge(df, new_yardline, on=['GameId','PlayId'], how='inner')
    return df

def recalculate_pos_and_dir(df): #transforms position direction and orientation to "rush from left to right"
    df['X'] = df[['X','PlayDirection']].apply(lambda x: new_X(x[0],x[1]), axis=1)
    df['Orientation'] = df[['Orientation','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)
    df['Dir'] = df[['Dir','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)
    return df





def features_relative_coordinates(df):
    rusher_features = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','NflIdRusher','X','Y','Orientation','Dir','YardLine']].copy()
    rusher_features['rusher_yards_to_go'] = rusher_features['YardLine'] - rusher_features['X'] #num yards behind LineOfScrimmage
    rusher_features['rusher_oriented_back'] = rusher_features['Orientation'].apply(lambda x: is_looking_back(x))
    rusher_features['rusher_moving_back'] = rusher_features['Dir'].apply(lambda x: is_looking_back(x))
    rusher_features = rusher_features.rename(columns={'X':'rusher_X', 'Y':'rusher_Y'})
    rusher_features = rusher_features[['GameId','PlayId','NflIdRusher','rusher_X','rusher_Y','rusher_yards_to_go',
                         'rusher_oriented_back','rusher_moving_back']]
    player_distance = df[['GameId','PlayId','NflId','X','Y']].copy() #from all players
    player_distance = pd.merge(player_distance, rusher_features, on=['GameId','PlayId'], how='inner')
    #player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']] #leave this line out ???
    player_distance['dist_to_rusher'] = player_distance[['X','Y','rusher_X','rusher_Y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
    player_distance = player_distance.groupby(['GameId','PlayId','rusher_yards_to_go','rusher_oriented_back','rusher_moving_back'])                                            .agg({'dist_to_rusher':['min','max','mean','std']})                                             .reset_index()
    player_distance.columns = ['GameId','PlayId','rusher_yards_to_go','rusher_oriented_back','rusher_moving_back',
                                   'min_dist','max_dist','mean_dist','std_dist']    
    df = pd.merge(player_distance,df,on=['GameId','PlayId'],how='inner')
    return df
    #adds new features 'rusher_yards_to_go', 'rusher_oriented_back','rusher_moving_back'
    #adds new features 'min_dist','max_dist','mean_dist','std_dist'
    #for the future: add each players distance to rusher


def defense_features(df):
    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']].copy()
    rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

    defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
    defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X','Y','RusherX','RusherY']]
    defense['def_dist_to_rusher'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

    defense = defense.groupby(['GameId','PlayId'])                         .agg({'def_dist_to_rusher':['min','max','mean','std']})                         .reset_index()
    defense.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist']
    
    df = pd.merge(defense,df,on=['GameId','PlayId'],how='inner')
    return df
    #adds new features 'def_min_dist','def_max_dist','def_mean_dist','def_std_dist'

    
def dist_to_rusher_features(df):
    rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','X','Y']].copy()
    rusher.columns = ['GameId','PlayId','RusherX','RusherY']

    defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
    defense = defense[['GameId','PlayId',"NflId",'X','Y','RusherX','RusherY']]
    defense['dist_to_rusher'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)
    defense.drop(["X", "Y", "RusherX", "RusherY"], axis=1, inplace=True)
    df = pd.merge(defense,df,on=['GameId','PlayId',"NflId"],how='inner')
    return df
    
    
def calc_new_position(X, Y, S, A, Dir):
    D1 = S + A/2 #acceleration remains unchanged within 1 second
    D3 = D1 + 2*S + 2*A #no further acceleration from second 1 - 3
    X1 = X + D1 * np.sin(Dir/180*np.pi)
    Y1 = Y + D1 * np.cos(Dir/180*np.pi)
    X3 = X + D3 * np.sin(Dir/180*np.pi)
    Y3 = Y + D3 * np.cos(Dir/180*np.pi)
    return X1, Y1, X3, Y3


def additional_time_series_features(df):
    #Annahme: Beschleunigung dauert 1 Sekunde an und die Geschwindigkeit bleibt dann gleich
    features = df[['GameId','PlayId','NflId','X','Y','S','A','Dir']].copy()
    features[['X1sec', 'Y1sec', 'X3sec', 'Y3sec']] = features[['X','Y','S','A', 'Dir']].apply(lambda x: calc_new_position(x[0],x[1],x[2],x[3],x[4]), axis=1, result_type="expand")
    features.drop(["X", "Y", "S", "A", "Dir"], axis=1, inplace=True)
    df = pd.merge(features,df,on=['GameId','PlayId','NflId'],how='inner')
    return df


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C


def calc_intersection(L1P1X, L1P1Y, L1P2X, L1P2Y, L2P1X, L2P1Y, L2P2X, L2P2Y):
    L1 = line([L1P1X,L1P1Y],[L1P2X,L1P2Y])
    L2 = line([L2P1X,L2P1Y],[L2P2X,L2P2Y])
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        if ((x > L1P1X) and (x > L1P2X)):
            return False
        elif ((x < L1P1X) and (x < L1P2X)):
            return False
        elif ((x < L2P1X) and (x < L2P2X)):
            return False
        elif ((x > L2P1X) and (x > L2P2X)):
            return False
        else:
            if ((y > L1P1Y) and (y > L1P2Y)):
                return False
            elif ((y < L1P1Y) and (y < L1P2Y)):
                return False
            if ((y > L2P1Y) and (y > L2P2Y)):
                return False
            elif ((y < L2P1Y) and (y < L2P2Y)):
                return False
            else:
                return True
    else:
        return False
    
        
        
        
        

def calc_crossing_movements(df):
    rusher_features = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','X','Y','X1sec','Y1sec','X3sec','Y3sec']].copy()
    rusher_features = rusher_features.rename(columns={'X':'rusher_X', 'Y':'rusher_Y', 'X1sec':'rusher_X1sec', 'Y1sec':'rusher_Y1sec', 'X3sec':'rusher_X3sec', 'Y3sec':'rusher_Y3sec'})
    #print(rusher_features.head())
    features = df[['GameId','PlayId','NflId','X','Y','X1sec','Y1sec','X3sec','Y3sec']].copy()
    features = pd.merge(features,rusher_features,on=['GameId','PlayId'],how='inner')
    #print(features.head())
    df["CrossesWithin_1"] = features[['X','Y','X1sec','Y1sec','rusher_X','rusher_Y','rusher_X1sec','rusher_Y1sec']].apply(lambda x: calc_intersection(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]), axis=1)
    df["CrossesWithin_3"] = features[['X','Y','X3sec','Y3sec','rusher_X','rusher_Y','rusher_X3sec','rusher_Y3sec']].apply(lambda x: calc_intersection(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]), axis=1)
    return df





from sklearn.base import BaseEstimator, TransformerMixin

class CategoricalFeaturesProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X):
        return self #do nothing
    def transform(self, X):
        X['Team'] = X['Team'].apply(lambda x: x.strip()=='home')
        X.drop(['DisplayName'], axis=1, inplace=True)
        X['GameClock'] = X['GameClock'].apply(convert_to_seconds)
        
        X["PossessionTeam"] = X["PossessionTeam"].replace("ARZ", "ARI")
        X["PossessionTeam"] = X["PossessionTeam"].replace("BLT", "BAL")
        X["PossessionTeam"] = X["PossessionTeam"].replace("CLV", "CLE")
        X["PossessionTeam"] = X["PossessionTeam"].replace("HST", "HOU")
        X["FieldPosition"] = X["FieldPosition"].replace("ARZ", "ARI")
        X["FieldPosition"] = X["FieldPosition"].replace("BLT", "BAL")
        X["FieldPosition"] = X["FieldPosition"].replace("CLV", "CLE")
        X["FieldPosition"] = X["FieldPosition"].replace("HST", "HOU")
        X["FieldPosition"] = X["FieldPosition"].replace(np.nan, "UNKNOWN")
        
        X['HomePossession'] = X['PossessionTeam'] == X['HomeTeamAbbr']
        X['HomeField'] = X['FieldPosition'] == X['HomeTeamAbbr']
        X['PossessionInOwnField'] = X['FieldPosition'] == X['PossessionTeam']
        X = recalculate_yardline(X)
        X.drop(["PossessionTeam", "HomeTeamAbbr", "VisitorTeamAbbr", "FieldPosition"], axis=1, inplace=True)
        
        X["OffenseFormation"] = X["OffenseFormation"].replace(np.nan, "UNKNOWN")
        X['OffenseFormation'] = X['OffenseFormation'].apply(classify_offense_formation)
        #do one hot encoding
        X['off_form_singleback'] = X['OffenseFormation'] == "singleback"
        X['off_form_shotgun'] = X['OffenseFormation'] == "shotgun"
        X['off_form_i_form'] = X['OffenseFormation'] == "i_form"
        X['off_form_pistol'] = X['OffenseFormation'] == "pistol"
        X['off_form_jumbo'] = X['OffenseFormation'] == "jumbo"
        X['off_form_wildcat'] = X['OffenseFormation'] == "wildcat"
        X['off_form_empty'] = X['OffenseFormation'] == "empty"
        X['off_form_ace'] = X['OffenseFormation'] == "ace"
        X['off_form_unknown'] = X['OffenseFormation'] == "unknown"
        X.drop(["OffenseFormation"], axis=1, inplace=True)
        
        offense_dict = {'DB' : "offense_DB", 'DL' : "offense_DL", 
                        'LB' : "offense_LB", 'OL' : "offense_OL", 
                        'QB' : "offense_QB", 'RB' : "offense_RB", 
                        'TE' : "offense_TE", 'WR' : "offense_WR"}
        for key in offense_dict.keys():
            X[offense_dict[key]] = np.zeros((X.shape[0],1))
        X["offense_UNKNOWN"] = np.zeros((X.shape[0],1))
        X = X.apply(ProcessOffensePersonnel, axis=1)
        
        defense_dict = {'DB' : "defense_DB", 'DL' : "defense_DL", 
                        'LB' : "defense_LB", 'OL' : "defense_OL"}
        for key in defense_dict.keys():
            X[defense_dict[key]] = np.zeros((X.shape[0],1))
        X["defense_UNKNOWN"] = np.zeros((X.shape[0],1))
        X = X.apply(ProcessDefensePersonnel, axis=1)
        
        X.drop(["OffensePersonnel", "DefensePersonnel"], axis=1, inplace=True)
            
        X['PlayDirection'] = X['PlayDirection'].apply(lambda x: x.strip() == 'right')
        X = recalculate_pos_and_dir(X)
        
        X = features_relative_coordinates(X)
        X = defense_features(X)
        X = dist_to_rusher_features(X)
        X = additional_time_series_features(X)
        X = calc_crossing_movements(X)
        
        X['TimeHandoff'] = X['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
        X['TimeSnap'] = X['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
        X['PlayerBirthDate'] = X['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
        X['TimeSnapToHandoff'] = X.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
        seconds_in_year = 60*60*24*365
        X['PlayerAge'] = X.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
        X.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate'], axis=1, inplace=True)
        
        X['PlayerHeight'] = X['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
        
        X['PlayerBMI'] = 703*(X['PlayerWeight']/(X['PlayerHeight'])**2)
        
        X.drop(["PlayerCollegeName"], axis=1, inplace=True)
        X.drop(["Position"], axis=1, inplace=True)
        X.drop(["Stadium"], axis=1, inplace=True)
        
        X['StadiumType'] = X['StadiumType'].apply(classify_stadium_type)
        
        X['Turf'] = X['Turf'].apply(classify_turf_type)
        
        X.drop(["Location"], axis=1, inplace=True)
        
        X['GameWeather'] = X['GameWeather'].apply(classify_weather)
        #Do One Hot Encoding
        X['weather_cloudy'] = X['GameWeather'] == "cloudy"
        X['weather_sunny'] = X['GameWeather'] == "sunny"
        X['weather_indoor'] = X['GameWeather'] == "indoor"
        X['weather_rainy'] = X['GameWeather'] == "rainy"
        X['weather_snow'] = X['GameWeather'] == "snow"
        X.drop(["GameWeather"], axis=1, inplace=True)
        
        X.drop(["WindSpeed", "WindDirection"], axis=1, inplace=True)
        
        return X
    
cat_feat_processor = CategoricalFeaturesProcessor()




int_features




class NumericalFeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X):
        return self #do nothing
    def transform(self, X):
        X['IsRusher'] = X['NflId'] == X['NflIdRusher']
        X.drop(['NflId', 'NflIdRusher'], axis=1, inplace=True)
        
        temp_train = X[X["IsRusher"]][["Team", "PlayId"]].rename(columns={"Team":"RusherTeam"})
        X = X.merge(temp_train, on = "PlayId")
        X["RusherTeam"] = X["Team"] == X["RusherTeam"]
        
        X.drop(["Season", "Week"], axis=1, inplace=True)
        
        X['YardsLeft'] = X.apply(lambda row: 100-row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)
        X['YardsLeft'] = X.apply(lambda row: row['YardsLeft'] if row['PlayDirection'] else 100-row['YardsLeft'], axis=1)
        
        X["diffScoreBeforePlay"] = X["HomeScoreBeforePlay"] - X["VisitorScoreBeforePlay"]
        
        X["Orientation"].fillna(0, inplace=True)
        X["Dir"].fillna(0, inplace=True)
        X["DefendersInTheBox"].fillna(train["DefendersInTheBox"].median(), inplace=True)
        X["Temperature"].fillna(train["Temperature"].mean(), inplace=True)
        X["Humidity"].fillna(train["Humidity"].mean(), inplace=True)
        X.fillna(-999, inplace=True) #default
        
        return X
    
num_feat_processor = NumericalFeatureProcessor()




float_features




train[train["PlayId"] == 20170907000118]




train = cat_feat_processor.fit_transform(train)
train = num_feat_processor.fit_transform(train)

cat_features = []
int_features = []
float_features = []
bool_features = []
uint8_features = []
for col in train.columns:
    #print(train[col].dtype) #print out to check datatypes
    if train[col].dtype =='object':
        cat_features.append((col, len(train[col].unique())))
    if train[col].dtype =='int64':
        int_features.append((col, len(train[col].unique())))
    if train[col].dtype =='float64':
        float_features.append((col, len(train[col].unique())))
    if train[col].dtype =='bool':
        bool_features.append((col, len(train[col].unique())))
    if train[col].dtype =='uint8':
        uint8_features.append((col, len(train[col].unique())))
        
''' contained datatypes
int64
object
float64
'''
print("{} columns in total".format(train.shape[1]))
print("{} Categorical Features found.".format(len(cat_features)))
print("{} Integer Features found.".format(len(int_features)))
print("{} Float Features found.".format(len(float_features)))
print("{} Boolean Features found.".format(len(bool_features)))
print("{} uint8 OneHot Features found.".format(len(uint8_features)))

print("Integer Features:\n{}".format(int_features))
print("Floating Features:\n{}".format(float_features))
print("Boolean Features:\n{}".format(bool_features))
print("Uint OneHot Features:\n{}".format(uint8_features))




train.head()




train[train["PlayId"] == 20170907000118]




print(calc_intersection(-1, 0, 1, 0, 0, 1, 0, -1))




train.describe()




train["CrossesWithin_1"].value_counts()




class TrainingDataPreparation(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X):
        return self #do nothing
    def transform(self, X):
        X = X.sort_values(by=['PlayId', 'RusherTeam', 'IsRusher', 'dist_to_rusher']).reset_index()
        
        unused_columns = ["GameId","PlayId","Team","IsRusher", "index"]
        unique_columns = ["X", "Y", "S", "A", "X1sec", "Y1sec", "X3sec", "Y3sec", "RusherTeam", "CrossesWithin_1", "CrossesWithin_3", "dist_to_rusher", "Dis", "Orientation", "Dir", "JerseyNumber", "PlayerHeight", "PlayerWeight", "PlayerAge", "PlayerBMI"]
        
        #params that are fed into NN:
        
        # 20*22 = 440 params for every player individually, sorted by RusherTeam first, Rusher first, distance_to_rusher
        # X, Y, S, A, X1, Y1, X3, Y3, CrossesWithin_1, CrossesWithin_3,
        # dist_to_rusher,Dis,Orientation,Dir,JerseyNumber,PlayerHeight,PlayerWeight,PlayerAge,PlayerBMI,RusherTeam
        
        #global params:
        #4 def_min_dist, def_max_dist, def_mean_dist, def_std_dist, 
        #3 rusher_yards_to_go, rusher_oriented_back, rusher_moving_back,
        #4 min_dist, max_dist, mean_dist, std_dist, 
        #8 Quarter, GameClock, Down, Distance, HomeScoreBeforePlay, VisitorScoreBeforePlay, DefendersInTheBox, PlayDirection,
        #9 Yards(!!!), StadiumType, Turf, Temperature, Humidity, HomePossession, HomeField, PossessionInOwnField, YardLine, 
        #5 off_form_singleback, off_form_shotgun, off_form_i_form, off_form_pistol, off_form_jumbo,
        #4 off_form_wildcat, off_form_empty, off_form_ace, off_form_unknown, 
        #9 offense_DB, offense_DL, offense_LB, offense_OL, offense_QB, offense_RB, offense_TE, offense_WR, offense_UNKNOWN, 
        #5 defense_DB, defense_DL, defense_LB, defense_OL, defense_UNKNOWN,
        #1 TimeSnapToHandoff, 
        #5 weather_cloudy, weather_sunny, weather_indoor, weather_rainy, weather_snow, 
        #2 YardsLeft, diffScoreBeforePlay
        #in total 59 global params
        
        #499 params in total
        
        
        #original columns
        #2 'GameId', 'PlayId', --> unused
        #5 'X1', 'Y1', 'X3', 'Y3', 'dist_to_rusher', --> player
        #4 'def_min_dist', 'def_max_dist', 'def_mean_dist', 'def_std_dist', --> global
        #3 'rusher_yards_to_go', 'rusher_oriented_back', 'rusher_moving_back', --> global
        #4 'min_dist', 'max_dist', 'mean_dist', 'std_dist', --> global
        #1 'Team', --> unused
        #8 'X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'JerseyNumber', --> player
        #5 'Quarter', 'GameClock', 'Down', 'Distance', 'HomeScoreBeforePlay', --> global
        #3 'VisitorScoreBeforePlay', 'DefendersInTheBox', 'PlayDirection',  --> global
        #1 'Yards', --> target
        #2 'PlayerHeight', 'PlayerWeight', --> player
        #3 'StadiumType', 'Turf', 'Temperature', --> global
        #5 'Humidity', 'HomePossession', 'HomeField', 'PossessionInOwnField', 'YardLine', --> global
        #5 'off_form_singleback', 'off_form_shotgun', 'off_form_i_form', 'off_form_pistol', 'off_form_jumbo', --> global
        #4 'off_form_wildcat', 'off_form_empty', 'off_form_ace', 'off_form_unknown', --> global
        #9 'offense_DB','offense_DL','offense_LB','offense_OL','offense_QB','offense_RB','offense_TE','offense_WR','offense_UNKNOWN', --> global
        #5 'defense_DB', 'defense_DL', 'defense_LB', 'defense_OL', 'defense_UNKNOWN', --> global
        #2 'CrossesWithin_1', 'CrossesWithin_3', --> player
        #1 'TimeSnapToHandoff', --> global
        #2 'PlayerAge', 'PlayerBMI', --> player
        #5 'weather_cloudy', 'weather_sunny', 'weather_indoor', 'weather_rainy', 'weather_snow', --> global
        #1 'IsRusher', --> unused
        #1 'RusherTeam', --> player
        #2 'YardsLeft', 'diffScoreBeforePlay' --> global
        #58 globals + yards as target
        #20 player
        #4 unused + index
        
        
        training_cols = []
        for c in X.columns:
            if c not in unique_columns + unused_columns:
                training_cols.append(c)
        for c in unique_columns:
            for i in range(22):
                training_cols.append(c+str(i))
        
        #print("{} columns\n{}".format(len(training_cols), training_cols))
        
        training_data=np.zeros((X.shape[0]//22,len(training_cols)))
        for i in range(0,X.shape[0],22):#for all plays
            count=0
            for c in training_cols:
                if c in X: #not in unique_columns and not in unused_columns
                    training_data[i//22][count] = X[c][i]
                    count+=1
            for c in unique_columns:
                for j in range(22):
                    training_data[i//22][count] = X[c][i+j]
                    count+=1
        X = pd.DataFrame(data=training_data, columns=training_cols)
        
        return X
    
train_dat_prep = TrainingDataPreparation()









print(train.shape)
train.columns




train_transformed = train_dat_prep.fit_transform(train)
print(train_transformed.shape)
print(train_transformed.describe())




y_train = train_transformed["Yards"]
X_train = train_transformed.drop(["Yards"], axis=1)
print(y_train.shape)
print(X_train.shape)




scaler_X = PowerTransformer()
#scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)




#from sklearn.preprocessing import StandardScaler
#scaler_y = StandardScaler()
#y_train = y_train.values
#scaler_y.fit(y_train.reshape(-1,1))
#y_train = scaler_y.transform(y_train.reshape(-1, 1)).flatten()




batch_size=512
print("Shape of X_train: {}".format(X_train.shape))
print("Shape of y_train: {}".format(y_train.shape))

X_train_nn = X_train

y_train_nn = np.zeros(shape=(X_train.shape[0], 199))
for i,yard in enumerate(y_train.astype(int)):
    y_train_nn[i, yard+99:] = np.ones(shape=(1, 100-yard))

print("Shape of X_train for NN: {}".format(X_train_nn.shape))
print("Shape of y_train for NN: {}".format(y_train_nn.shape))




#from https://www.kaggle.com/davidcairuz/nfl-neural-network-w-softmax
def crps(y_true, y_pred):
    return K.mean(K.square(y_true - K.cumsum(y_pred, axis=1)), axis=1)




'''def get_model():
    x = keras.layers.Input(shape=[X_train.shape[1]])
    fc1 = keras.layers.Dense(units=384, input_shape=[X_train.shape[1]])(x)
    act1 = keras.layers.PReLU()(fc1)
    #bn1 = keras.layers.BatchNormalization()(act1)
    dp1 = keras.layers.Dropout(0.7)(act1)
    concat1 = keras.layers.Concatenate()([x, dp1])
    fc2 = keras.layers.Dense(units=320)(concat1)
    act2 = keras.layers.PReLU()(fc2)
    bn2 = keras.layers.BatchNormalization()(act2)
    dp2 = keras.layers.Dropout(0.6)(bn2)
    concat2 = keras.layers.Concatenate()([x, dp2])
    fc3 = keras.layers.Dense(units=256)(concat2)
    act3 = keras.layers.PReLU()(fc3)
    #bn3 = keras.layers.BatchNormalization()(act3)
    dp3 = keras.layers.Dropout(0.5)(act3)
    output = keras.layers.Dense(units=199, activation='softmax')(dp3)
    model = keras.models.Model(inputs=[x], outputs=[output])
    return model'''

'''def get_model():
    x = keras.layers.Input(shape=[X_train.shape[1]])
    fc1 = keras.layers.Dense(units=512, activation="relu", input_shape=[X_train.shape[1]])(x)
    dp1 = keras.layers.Dropout(0.4)(fc1)
    fc2 = keras.layers.Dense(units=256, activation="relu")(dp1)
    dp2 = keras.layers.Dropout(0.2)(fc2)
    output = keras.layers.Dense(units=199, activation='softmax')(dp2)
    model = keras.models.Model(inputs=[x], outputs=[output])
    return model'''

def get_model():
    x = keras.layers.Input(shape=[X_train.shape[1]])
    fc1 = keras.layers.Dense(units=384, activation="relu", input_shape=[X_train.shape[1]])(x)
    bn1 = keras.layers.BatchNormalization()(fc1)
    dp1 = keras.layers.Dropout(0.5)(bn1)
    fc2 = keras.layers.Dense(units=256, activation="relu")(dp1)
    bn2 = keras.layers.BatchNormalization()(fc2)
    dp2 = keras.layers.Dropout(0.5)(bn2)
    output = keras.layers.Dense(units=199, activation='softmax')(dp2)
    model = keras.models.Model(inputs=[x], outputs=[output])
    return model

'''def get_model():
    x = keras.layers.Input(shape=[X_train.shape[1]])
    fc1 = keras.layers.Dense(units=384, input_shape=[X_train.shape[1]])(x)
    act1 = keras.layers.PReLU()(fc1)
    dp1 = keras.layers.Dropout(0.7)(act1)
    fc2 = keras.layers.Dense(units=320)(dp1)
    act2 = keras.layers.PReLU()(fc2)
    dp2 = keras.layers.Dropout(0.6)(act2)
    fc3 = keras.layers.Dense(units=256)(dp2)
    act3 = keras.layers.PReLU()(fc3)
    dp3 = keras.layers.Dropout(0.5)(act3)
    output = keras.layers.Dense(units=199, activation='softmax')(dp3)
    model = keras.models.Model(inputs=[x], outputs=[output])
    return model'''

'''def get_model():
    x = keras.layers.Input(shape=[X_train.shape[1]])
    fc1 = keras.layers.Dense(units=1024, activation="relu", input_shape=[X_train.shape[1]])(x)
    bn1 = keras.layers.BatchNormalization()(fc1)
    dp1 = keras.layers.Dropout(0.5)(bn1)
    fc2 = keras.layers.Dense(units=512, activation="relu")(dp1)
    bn2 = keras.layers.BatchNormalization()(fc2)
    dp2 = keras.layers.Dropout(0.5)(bn2)
    fc3 = keras.layers.Dense(units=512, activation="relu")(dp2)
    bn3 = keras.layers.BatchNormalization()(fc3)
    dp3 = keras.layers.Dropout(0.5)(bn3)
    output = keras.layers.Dense(units=199, activation='softmax')(dp3)
    model = keras.models.Model(inputs=[x], outputs=[output])
    return model'''


earlyStopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    verbose=1,
    mode='min',
    restore_best_weights=True,
    min_delta=1e-5
)

reduceLR = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
    min_lr=1e-6,
    verbose=1,
    mode='min'
)




def train_model(X_train, y_train, X_val, y_val):
    model = get_model()
    #model.compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-6), loss=crps)
    model.compile(optimizer=Adam(lr=1e-4), loss=crps)
    checkPoint = ModelCheckpoint('best_model.h5',monitor='val_loss',mode='min', save_best_only=True, verbose=1, save_weights_only=True)
    model.fit(X_train, y_train, epochs=150, callbacks=[earlyStopping, reduceLR, checkPoint], validation_data=[X_val, y_val], batch_size=batch_size)
    return model




from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=5, n_repeats=1)

models = []

for tr_idx, vl_idx in rkf.split(X_train_nn, y_train_nn):
    
    x_tr, y_tr = X_train_nn[tr_idx], y_train_nn[tr_idx]
    x_vl, y_vl = X_train_nn[vl_idx], y_train_nn[vl_idx]
    
    model = train_model(x_tr, y_tr, x_vl, y_vl)
    model.load_weights("best_model.h5")
    models.append(model)




plot_model(models[0], to_file="mymodel.png")




def make_pred_nn(df, models):
    tdata_local = cat_feat_processor.fit_transform(df)
    tdata_local = num_feat_processor.fit_transform(tdata_local)
    tdata_local = train_dat_prep.fit_transform(tdata_local)
    #y_train = tdata_local["Yards"] #not used
    #print("Shape after transform: {}".format(tdata_local.shape))
    if ("Yards" in tdata_local.columns):
        tdata_local = tdata_local.drop(["Yards"], axis=1)
    tdata_local = scaler_X.transform(tdata_local)
    
    #print("Training Data shape: {}".format(tdata_local.shape))
    #print("Models shape: {}".format(len(models)))
    y_pred = np.zeros((tdata_local.shape[0], 199))        
    for model in models:
        y_pred += np.cumsum(model.predict(tdata_local), axis=1)
    y_pred = y_pred/len(models)
    #print("Output shape: {}".format(y_pred.shape))
    return y_pred
    

def make_pred_env_nn(df, sample_prediction, env, models):
    y_pred = make_pred_nn(df, models)
    env.predict(pd.DataFrame(data=y_pred.clip(0,1),columns=sample_prediction.columns))




#train_evaluate_nn = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})




'''
y_prediction_nn = np.zeros((X_train_nn.shape[0],199))
y_true_nn = np.zeros((X_train_nn.shape[0],199))

num_plays = train_evaluate_nn["PlayId"].unique().shape[0]
print("Evaluating {} plays.".format(num_plays))

index = 0
for play in tqdm.tqdm(train_evaluate_nn["PlayId"].unique()):
    current_play = train_evaluate_nn[train_evaluate_nn["PlayId"] == play].copy()
    #print("Play-ID: {} Shape before transform: {}".format(play, current_play.shape))
    y_pred = make_pred_nn(current_play, models)
    #print("Predicted y: {}".format(y_pred))
    y_prediction_nn[index] = y_pred[0]
    y_true_val = current_play["Yards"].values[0]
    y_true = np.zeros((1, 199))
    y_true[0, y_true_val+99:] = np.ones((1, 100-y_true_val))
    y_true_nn[index] = y_true[0]
    #print("True y: {}\n{}".format(y_true_val, y_true))  
    index = index + 1
'''    




#print("Validation score NN:",np.sum(np.power(y_prediction_nn-y_true_nn,2))/(199*(num_plays)))




for test, sample_prediction in tqdm.tqdm(env.iter_test()):
     make_pred_env_nn(test, sample_prediction, env, models)




env.write_submission_file()






