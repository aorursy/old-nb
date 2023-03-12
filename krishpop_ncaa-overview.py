import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifierCV
import xgboost as xgb
# Only using the Teams, and Detailed Tournament Results files

detailed_tourney_df = pd.read_csv('../input/TourneyDetailedResults.csv')
compact_tourney_df = pd.read_csv('../input/TourneyCompactResults.csv')
teams_df = pd.read_csv('../input/Teams.csv')
seasons = pd.read_csv('../input/Seasons.csv')
regular_season = pd.read_csv('../input/RegularSeasonCompactResults.csv')
regular_season16 = regular_season.loc[regular_season['Season']==2016]
tourney_seeds = pd.read_csv('../input/TourneySeeds.csv')
tourney_seeds16 = tourney_seeds.loc[tourney_seeds['Season']==2016]
tourney_slots = pd.read_csv('../input/TourneySlots.csv')
tourney_slots16 = tourney_slots.loc[tourney_slots['Season']==2016]

team_dict = dict(zip(teams_df['Team_Id'].values, teams_df['Team_Name'].values))
tourney_df = detailed_tourney_df
detailed_tourney_df['Wteam_name'] = detailed_tourney_df['Wteam'].map(team_dict)
detailed_tourney_df['Lteam_name'] = detailed_tourney_df['Lteam'].map(team_dict)
# Distribution density of wins and losses.

tourney_wins = tourney_df.loc[tourney_df['Wteam'] !=0, 'Wteam'].value_counts()
tourney_losses = tourney_df.loc[tourney_df['Wteam'] !=0, 'Lteam'].value_counts()
tourney_df['Wwins'] = tourney_df['Wteam'].map(tourney_wins)
tourney_df['Lwins'] = tourney_df['Lteam'].map(tourney_wins)
tourney_df = tourney_df.replace(np.nan, 0)
# Look at a key statistic "Aggressiveness Percentage (AP)" - 
# [Free throws to Field goals taken ratio], <code>[Three pointers taken to Field 
# goals ratio], for both the winning and the losing teams.

wft_wfg = (tourney_df['Wfta'].values * 0.66) / tourney_df['Wfga'].values 
wtp_wfg = tourney_df['Wfga3'].values / tourney_df['Wfga'].values

# arbitraty penalty term = 0.66 - some ft attempts stem from fg attempts
# this can obviously be changed to a different value

lft_lfg = (tourney_df['Lfta'].values * 0.66) / tourney_df['Lfga'].values
ltp_lfg = tourney_df['Lfga3'].values / tourney_df['Lfga'].values
# Correlation coefficient between these 3 features

print('Correlation matrix for wins:')
print('==============================')
print(np.corrcoef((wft_wfg, tourney_df['Wscore'], wtp_wfg)))
print('\nCorrelation matrix for losses:')
print('===============================')
print(np.corrcoef((lft_lfg, tourney_df['Lscore'], ltp_lfg)))
# Finding each team's "Last Ten" -> lt
tourney_df = compact_tourney_df
# seeds16 = tourney_seeds16[tourney_seeds16.columns[1]].str.extract("(?P<Region>[A-Z+a-z])(?P<Seed>[0-9]+)")
tourney_seeds16["seed"].apply(str.extract("(?P<Region>[A-Z+a-z])(?P<Seed>[0-9]+)"))
#tourney_seeds16 = pd.concat(tourney_seeds16, tourney_seeds16

# regular_season16.loc[regular_season16['WTeam']==1101, regular_season16['LTeam']==1101].head()
# In winning teams, [Ft : Fg] ratio correlation to Points scored is much 
# stronger (9.78) than it is in losing teams (0.22)r. Further, in winning teams, 
# Points scored is much more negatively correlated to [Tp : Fg] ratio, than it is 
# in losing teams.

# True shooting percentage
wtsp = tourney_df['Wscore'] / (2 * (tourney_df['Wfga'] + (0.44 * tourney_df['Wfta'])))
ltsp = tourney_df['Lscore'] / (2 * (tourney_df['Lfga'] + (0.44 * tourney_df['Lfta'])))

# Offensive effeciency rating
w_off_rating = 100 * tourney_df['Wscore'] / (tourney_df['Wfga'] + 0.40 * tourney_df['Wfta'] - 1.07 * \
                                            (tourney_df['Wor'] / (tourney_df['Wor'] + tourney_df['Wdr'])) * \
                                            (tourney_df['Wfga'] + tourney_df['Wfgm']) + tourney_df['Wto'])
l_off_rating = 100 * tourney_df['Lscore'] / (tourney_df['Lfga'] + 0.40 * tourney_df['Lfta'] - 1.07 * \
                                            (tourney_df['Lor'] / (tourney_df['Lor'] + tourney_df['Ldr'])) * \
                                            (tourney_df['Lfga'] + tourney_df['Lfgm']) + tourney_df['Lto'])
# Heatmap correlation matrix showing which stats have highest correlation to 
# offensive rating. Focusing on the last two columns.

tourney_df['Worating'] = w_off_rating
tourney_df['Lorating'] = l_off_rating

plt.figure(figsize=(20, 15))
cm = np.corrcoef(tourney_df.iloc[:, 8:34].values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', \
                 yticklabels=tourney_df.columns.values[8:34], xticklabels=tourney_df.columns.values[8:34])
# Note: Number of wins for a team in the "Wteam" column will equal number of wins of 
# that same team if that team is in the "Lteam" column. Number of tournament wins is just, 
# "How many times has this "Team_Id" won an NCAA tourney game".

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

ax.set_title('Correlation: Offensive rating & NCAAM tourney wins')
ax.scatter(w_off_rating, tourney_df['Wwins'], wft_wfg, c='#0080ff', alpha=0.5, edgecolors='None', label='Winning Teams')
ax.scatter(l_off_rating, tourney_df['Lwins'], lft_lfg, c='#ff6600', alpha=0.5, edgecolors='None', label='Losing Teams')
ax.set_xlabel('Offensive Rating')
ax.set_ylabel('Tournament Wins')
ax.set_zlabel('True Shooting Percentage')
ax.view_init(azim=110)

print('==================================================')
print('Correlation between Winning team offensive rating')
print('and amount of NCAAM tourney wins winning team has:')
print('==================================================')
print(np.corrcoef((w_off_rating, tourney_df['Wwins'], wft_wfg)))

# While I was surprised to see that Offensive Rating didn't have more of a 
# correlation with Number of Tourney Wins, I'm not surprised at Offensive Rating's 
# relatively strong correlation to the Aggressiveness Percentage (AP) "[FT's : FG's]/[3s:FGs]", 
# which just shows how key of a stat AP really is (in my opinion).
# See how the average stats from all games measure up against the top teams.

all_means = []

# Get mean of all games played 
for column in tourney_df.columns.values[8:34]:
    all_means.append([column, np.mean(tourney_df[column])])

# Get top teams who've won at least 20 NCAA tournament games
bt = tourney_df.loc[(tourney_df['Wwins'] >= 20) & (tourney_df['Lwins'] >= 20), ['Wteam_name', 'Lteam_name']]
bt1 = bt['Wteam_name'] 
bt2 = bt['Lteam_name']
best_teams = np.concatenate((bt1, bt2))

best_means = []

# Just picking a single random team out of the list of best teams
num = random.randint(0,21)
bt_stats = tourney_df.loc[tourney_df['Wteam_name'] == best_teams[num]]

for column in bt_stats.columns.values[8:34]:
    best_means.append([column, np.mean(bt_stats[column])])

x1 = [item[0] for item in all_means]   
y1 = [item[1] for item in all_means]
x2 = [item[0] for item in best_means]
y2 = [item[1] for item in best_means]

plt.figure(figsize=(20,5))
sns.set_style('whitegrid')

plt.title('The Best vs. The Field')
sns.barplot(x=x1, y=y1, color='#0080ff', alpha=0.5, label='Means of All Games')
sns.barplot(x=x2, y=y2, color='#ff6600', alpha=0.5, label='Means of Games from Top Team')
plt.xlabel('Statistic')
plt.ylabel('Statistic Count')
plt.legend(loc='upper right')

print('===============')
print('Best Teams:')
print('===============')
print(pd.Series(best_teams).unique())