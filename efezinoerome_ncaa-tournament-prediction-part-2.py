#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.plotly as py
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.




temp = os.listdir("../input")
all_csv = {}
for i in range(0,len(temp)):
#for i in range(0,33):
    if(temp[i].split(".")[1] == "csv"):
        all_csv[temp[i].split(".")[0]] = pd.read_csv("../input/"+temp[i],encoding = 'ISO-8859-1')

print(all_csv.keys())




[i for i in all_csv.keys() if 'Prelim' in i]




def summ(df):
    print(df.head())
    print(100*"*")
    print(df.info())
    print(100*"*")
    if(len(df.select_dtypes(include = ['O']).columns) == 0):
        print("No Objects in DataFrame")
    else:
        print(df.describe(include = ['O']))
    print(100*"*")
    if(len(df.select_dtypes(exclude = ['O']).columns) == 0):
        print("No Integers/Floats in DataFrame")
    else:
        print(df.describe(exclude =['O']))
    print(100*"*")
    if(np.sum(df.isnull().sum().values) == 0):
        print('No missing values')
    else:
        print(df.isnull().sum()[df.isnull().sum().values== 1])
    return(None)




def heatmap(df, height_ratio = (.9, .03), hspace = .15, fig_w_h = (18,25)):
    grid_kws = {"height_ratios": height_ratio, "hspace": hspace}
    fig1, (ax1, cbar1) = plt.subplots(2, gridspec_kw=grid_kws)
    fig1.set_size_inches(fig_w_h)
    ax = sns.heatmap(df.select_dtypes(exclude = ['object']).corr(),                  ax=ax1, cbar_ax=cbar1,                  cbar_kws={"orientation": "horizontal"},vmin = -1, vmax = 1,                 linewidths = 1,annot = True,fmt='.2f',
                annot_kws = {"size": 11})
    cbar1.set_title("Correlation Between Features")
    ax.tick_params(labelsize=12)
    return(ax)
    #scale figure appropriately




def analyze_cont(df,col,x = 'Season',hue = "WLoc",type_plot = "box",wid = 11, hght = 8):
    if (type_plot == "hist"):
        temp_len = len(df[hue].unique())
        hue_val = df[hue].unique()
        #print(hue_val)
        for i in range(len(col)):
            fig,ax = plt.subplots(1,1,figsize = (wid,hght)) 
            #ax.append(plt.subplot2grid(shape=(nx,ny),rowspan = 1, colspan = 1, loc = (xv[i],yv[i])))
            for j in range(0,temp_len):
                sns.distplot(df.loc[df[hue]==hue_val[j],col[i]],                     ax = ax,label = hue_val[j], hist = False, kde = True)
        #ax.xaxis.set_tick_params(rotation=45)
        ax.legend(loc = 0)
        plt.show()
    elif(type_plot == "box"):
        for i in range(len(col)):
            fig,ax = plt.subplots(1,1,figsize = (wid,hght)) 
            #ax.append(plt.subplot2grid(shape=(nx,ny),rowspan = 1, colspan = 1, loc = (xv[i],yv[i])))
            sns.boxplot(y = col[i], x = x, data = df,                        ax = ax, hue = hue, orient = 'v')
            ax.xaxis.set_tick_params(rotation=45)
            ax.legend(loc = 0)
            plt.show()
    elif(type_plot == "violin"):
        for i in range(len(col)):
            fig,ax = plt.subplots(1,1,figsize = (wid,hght)) 
            #ax.append(plt.subplot2grid(shape=(nx,ny),rowspan = 1, colspan = 1, loc = (xv[i],yv[i])))
            sns.violinplot(y = col[i], x = x, data = df,                        ax = ax, hue = hue, orient = 'v')
            ax.xaxis.set_tick_params(rotation=45)
            ax.legend(loc = 0)
            plt.show()
    else:
        print("Invalid Plot Type")
    return(None)




def rankCount_season(df, wid, hght, x_val, y_val, col):
    fig, ax = plt.subplots(13,2, figsize = (wid, hght))
    plt_range = [(i,j) for i in range(13) for j in range(2)]
    for k,m in zip(plt_range,range(1,26)):
        sns.barplot(data = df[df[col]==m].sort_values(by = ["rankCount"], ascending = False), 
                    x = x_val, y = y_val, ax = ax[k[0]][k[1]])
        ax[k[0]][k[1]].xaxis.label.set_size(15)
        ax[k[0]][k[1]].yaxis.label.set_size(15)
        #ax[k[0]][k[1]].set_xticklabels(ax[k[0]][k[1]].get_xticklabels(), fontsize=30)
        ax[k[0]][k[1]].set_yticklabels(ax[k[0]][k[1]].get_yticklabels(), fontsize=15)
        ax[k[0]][k[1]].set_title("Rank: " + str(m), size = 20)
    title ="AP POLL Season: "+str(df.Season.unique()[0])
    #print(title)
    plt.suptitle(title, fontsize = 30)
    plt.subplots_adjust(top=0.97, wspace = 0.35, hspace = 0.35)
   #plt.tight_layout(h_pad = 2)
    return(None)




summ(all_csv['RegularSeasonDetailedResults_Prelim2018'])




heatmap(all_csv['RegularSeasonDetailedResults_Prelim2018'])




all_csv['RegularSeasonDetailedResults_Prelim2018']['WLoc'].unique().tolist()




all_csv['RegularSeasonDetailedResults_Prelim2018'].columns




all_csv['RegularSeasonDetailedResults_Prelim2018'][[i for i in     all_csv['RegularSeasonDetailedResults_Prelim2018'].columns if i not in     ['Season','DayNum','WTeamID','LTeamID']]].head()




col = all_csv['RegularSeasonDetailedResults_Prelim2018'][[i for i     in all_csv['RegularSeasonDetailedResults_Prelim2018'].columns if i not in     ['Season','DayNum','WTeamID','LTeamID','WLoc']]].columns




analyze_cont(df = all_csv['RegularSeasonDetailedResults_Prelim2018'],col = col,             hue = 'WLoc',type_plot = 'box',x = 'Season',wid = 10, hght = 8)




analyze_cont(df = all_csv['RegularSeasonDetailedResults_Prelim2018'],col = col,             hue = 'WLoc',type_plot = 'hist',x = 'Season',wid = 10, hght = 8)




all_csv['RegularSeasonDetailedResults_Prelim2018'].columns




Losing_Team = all_csv['RegularSeasonDetailedResults_Prelim2018'][[i for i in                 all_csv['RegularSeasonDetailedResults_Prelim2018'].columns                                                    if i[0]=='L' or i in ['Season','WLoc']]]




Losing_Team.head()




for name_i in Losing_Team.columns.tolist()[1:]:
    Losing_Team = Losing_Team.rename(index = str, columns =                   {name_i:name_i[1:]})
Losing_Team.head()




Losing_Team["Outcome"] = "Losing Team"




Losing_Team.head()




Winning_Team = all_csv['RegularSeasonDetailedResults_Prelim2018'][[i for i in                 all_csv['RegularSeasonDetailedResults_Prelim2018'].columns                                                    if i[0]=='W' or i in ['Season']]]




Winning_Team.head()




for name_i in Winning_Team.columns.tolist()[1:]:
    Winning_Team = Winning_Team.rename(index = str, columns =                   {name_i:name_i[1:]})
Winning_Team.head()




Winning_Team["Outcome"] = "Winning Team"




Winning_Team.head()




Winning_Losing_Team = pd.concat([Winning_Team, Losing_Team], axis = 0)




Winning_Losing_Team.head()




ax = sns.factorplot(x="Loc", y='FGM',
                  hue="Outcome", col="Season", col_wrap = 4,\
                   data=Winning_Losing_Team, \
                   orient="v", palette="Set3", \
                   kind="violin", dodge=True, size = 4, aspect = 0.8)




for name_i in [i for i in Winning_Losing_Team.columns.tolist() if i not in                ['Season','Loc','Outcome','TeamID']]:
    ax = sns.factorplot(x="Loc", y=name_i,
                   hue="Outcome", col="Season", col_wrap = 4,\
                    data=Winning_Losing_Team, \
                    orient="v", palette="Set3", \
                    kind="box", dodge=True)
    #g[name_i]=ax




summ(all_csv['NCAATourneyDetailedResults'])




heatmap(all_csv['NCAATourneyDetailedResults'])




tour_losing_team =all_csv['NCAATourneyDetailedResults'][[i for i in                 all_csv['NCAATourneyDetailedResults'].columns                if i[0]=='L' or i in ['Season','WLoc']]]
for name_i in tour_losing_team.columns.tolist()[1:]:
    tour_losing_team = tour_losing_team.rename(index = str, columns =                   {name_i:name_i[1:]})
tour_losing_team["Outcome"] = "Losing Team"
tour_losing_team.head()




tour_winning_team =all_csv['NCAATourneyDetailedResults'][[i for i in                 all_csv['NCAATourneyDetailedResults'].columns                if i[0]=='W' or i in ['Season','WLoc']]]
for name_i in tour_winning_team.columns.tolist()[1:]:
    tour_winning_team = tour_winning_team.rename(index = str, columns =                   {name_i:name_i[1:]})
tour_winning_team["Outcome"] = "Winning Team"
tour_winning_team.head()




tour_winning_losing_team = pd.concat([tour_winning_team,tour_losing_team], axis = 0)
tour_winning_losing_team.head()




ax = sns.factorplot(x="Loc", y='FGM',
                  hue="Outcome", col="Season", col_wrap = 4,\
                   data=tour_winning_losing_team, \
                   orient="v", palette="Set3", \
                   kind="bar", dodge=True, size = 4, aspect = 0.8)




summ(all_csv['Cities_Prelim2018'])




all_csv['Cities_Prelim2018'][all_csv['Cities_Prelim2018'].State.isnull()]




all_csv['Cities_Prelim2018'].State.fillna('PR',inplace = True)




all_csv['Cities_Prelim2018'].isnull().sum()




all_csv['Cities_Prelim2018'].State.value_counts().index




fig, ax = plt.subplots(1,1, figsize = (15,5))
sns.countplot(all_csv['Cities_Prelim2018']['State'], ax = ax,
             order = all_csv['Cities_Prelim2018'].State.value_counts().index)
ax.xaxis.set_tick_params(rotation = 90)




all_csv['Cities_Prelim2018'].State.value_counts()[all_csv['Cities_Prelim2018'].State.value_counts().values>15]




100*all_csv['Cities_Prelim2018'].State.value_counts()[all_csv['Cities_Prelim2018'].State.value_counts().values>15].sum()/   all_csv['Cities_Prelim2018'].State.value_counts().sum()




summ(all_csv['GameCities_Prelim2018'])




ax = sns.barplot(x = "CRType", y = "Percentage",            data = pd.DataFrame(100*all_csv['GameCities_Prelim2018'].CRType.value_counts()/all_csv['GameCities_Prelim2018'].CRType.value_counts().sum()).reset_index().           rename(index = str, columns = {"index":"CRType","CRType":"Percentage"}))
pos = range(len(all_csv['GameCities_Prelim2018'].CRType.value_counts().values))
max_val = [np.around(i,2) for i in (100*all_csv['GameCities_Prelim2018'].CRType.value_counts()/all_csv['GameCities_Prelim2018'].CRType.value_counts().sum()).tolist()]
for tick,label in zip(pos,ax.get_xticklabels()):
    ax.text(pos[tick], max_val[tick] + 0.03, str(max_val[tick])+"%",     horizontalalignment='center', color='black', weight='semibold')
plt.show()




cities = all_csv['Cities_Prelim2018'].merge(all_csv['GameCities_Prelim2018'],on = "CityID")




summ(cities)




#fig, ax = plt.subplots(1,1)
#g = sns.factorplot("CRType", col = 'State',data = cities,\
#              kind = "count", col_wrap = 4)
cities.groupby(['State','CRType'])['CRType'].count().apply(lambda x: 100*x/cities.shape[0]).sort_values(ascending = False).head(10)




sns.factorplot(data = cities.groupby(['State','CRType'])['City'].count().apply(lambda x: 100*x/cities.shape[0]).reset_index(level = "CRType").reset_index(), col = 'State', col_wrap = 4, x = "CRType", y = "City", kind = "bar")




cities[cities.CRType =='NCAA'].State.value_counts().apply(lambda x: 100 * x/cities[cities.CRType =='NCAA'].shape[0]).head(10)




cities[cities.CRType =='NCAA'].State.value_counts().apply(lambda x: 100 * x/cities[cities.CRType =='NCAA'].shape[0]).reset_index().head(10)




fig1, ax1 = plt.subplots(1,1, figsize = (15,5))
sns.barplot(data =cities[cities.CRType =='NCAA'].State.value_counts().apply(lambda x: 100 * x/cities[cities.CRType =='NCAA'].shape[0]).reset_index(), x = "index", y = "State",
order = cities[cities.CRType =='NCAA'].State.value_counts().sort_values(ascending = False).index,\
ax = ax1)
ax1.set_xlabel('STATES')
ax1.set_ylabel('Percentage (%)')
plt.show()




summ(all_csv['MasseyOrdinals_Prelim2018'])




masseyOrdinals = all_csv['MasseyOrdinals_Prelim2018'].merge(all_csv['Teams'][['TeamID','TeamName']], on = 'TeamID')




100*masseyOrdinals.SystemName.value_counts().sort_values(ascending = False).head(10)/masseyOrdinals.SystemName.value_counts().sum()




masseyOrdinals.groupby(['Season','SystemName'])['OrdinalRank'].agg({"Min": np.min, "Max":np.max}).head(10)




masseyOrdinals_AP = masseyOrdinals[masseyOrdinals.SystemName == 'AP']
masseyOrdinals_AP.head()




fig, ax = plt.subplots(8,2)
fig.set_size_inches(30,150)
wid_hei = [(j,k) for j in range(9) for k in range(2)]
for i,j in zip(np.sort(masseyOrdinals_AP.Season.unique()),wid_hei):
    sns.boxplot(data = masseyOrdinals_AP[masseyOrdinals_AP.Season == i],                 y = 'TeamName', x = 'OrdinalRank', ax = ax[j[0]][j[1]])
    ax[j[0]][j[1]].yaxis.label.set_size(20)
    ax[j[0]][j[1]].xaxis.label.set_size(20)
    ax[j[0]][j[1]].set_ylabel('Team Name')
    ax[j[0]][j[1]].set_yticklabels(ax[j[0]][j[1]].get_yticklabels(), fontsize=20)
    ax[j[0]][j[1]].set_title("Season: " + str(i), size = 30)
    #ax[j[0]][j[1]].set_xticklabels(ax[j[0]][j[1]].get_xticklabels(), fontsize=20)
plt.tight_layout()




masseyOrdinals_AP_rankcount = masseyOrdinals_AP.groupby(    ['Season','TeamName','OrdinalRank'])['SystemName'].count()
masseyOrdinals_AP_rankcount.head()




masseyOrdinals_AP_rankcount = masseyOrdinals_AP_rankcount.reset_index(level =     ["TeamName","OrdinalRank","Season"]).rename(index=str, columns={"SystemName": "rankCount"})




rankCount_season(df = masseyOrdinals_AP_rankcount[masseyOrdinals_AP_rankcount.Season == 2010],
                wid = 20, hght = 60, y_val = "TeamName", x_val = "rankCount", col = "OrdinalRank")




plt_val = [(i,j) for i in range(3) for j in range(2)]
fig, ax = plt.subplots(3,2, figsize = (20,35))
temp = masseyOrdinals_AP.groupby(        ['OrdinalRank','TeamName'])['SystemName'].count().reset_index(level =         ['TeamName','OrdinalRank'])
#print(temp.head())
for i,j in zip(plt_val, range(1,7)):
    sns.barplot(data = temp[temp.OrdinalRank == j].sort_values(by = 'SystemName', ascending = False),                 y = "TeamName", x = "SystemName",                 ax = ax[i[0]][i[1]])
    ax[i[0]][i[1]].yaxis.label.set_size(15)
    ax[i[0]][i[1]].xaxis.label.set_size(15)
    ax[i[0]][i[1]].set_xlabel('AP Rank: '+str(j)+" count")
    ax[i[0]][i[1]].set_yticklabels(ax[i[0]][i[1]].get_yticklabels(), fontsize=12)
    ax[i[0]][i[1]].set_title("Rank: "+str(j), size = 20)
plt.suptitle("AP Total Rank Since 2003", fontsize=30)
plt.subplots_adjust(wspace = 0.3, hspace = 0.125, top=0.965)




summ(all_csv['TeamCoaches_Prelim2018'])




teamCoaches = all_csv['TeamCoaches_Prelim2018'].merge(all_csv['Teams'][['TeamID','TeamName']])
teamCoaches.head()




teamCoaches_count = teamCoaches.groupby(['TeamName','CoachName'])['Season'].count()
teamCoaches_count.sort_values(ascending = False).head(10)




all_csv['TeamCoaches_Prelim2018'].groupby(["Season"])




summ(all_csv['Conferences'])




summ(all_csv['TeamConferences'])




summ(all_csv['ConferenceTourneyGames'])




summ(all_csv['SecondaryTourneyTeams'])




summ(all_csv['SecondaryTourneyCompactResults'])




summ(all_csv['TeamSpellings'])




summ(all_csv['NCAATourneySlots'])




summ(all_csv['NCAATourneySeedRoundSlots'])






