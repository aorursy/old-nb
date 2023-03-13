#!/usr/bin/env python
# coding: utf-8



get_ipython().system(' apt install libgeos-dev')
get_ipython().system(' pip uninstall -y shapely; pip install --no-binary :all: shapely==1.6.4')
get_ipython().system(' pip uninstall -y cartopy; pip install --no-binary :all: cartopy==0.17.0')
get_ipython().system(' pip install geoviews==1.6.6 hvplot==0.5.2 panel==0.8.0 bokeh==1.4.0')




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from operator import add, mul
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import hvplot.pandas
import holoviews as hv
import cartopy.crs as ccrs
import geopandas as gpd
from toolz.curried import map, partial, pipe, reduce
from statsmodels.regression.linear_model import OLS

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

hv.extension('bokeh')




countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).replace('United States of America', 'US')
data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv', parse_dates=['Date'], index_col='Id')
data




weeks = (data
         .assign(dayofweek = lambda df: df.Date.dt.dayofweek)
         .set_index('Date')
         .drop(columns=['Province/State', 'Lat', 'Long'])
         .groupby(['Country/Region', pd.Grouper(freq='W')]).agg({'ConfirmedCases':'sum', 'Fatalities':'sum', 'dayofweek':'max'})
         .reset_index()
         .where(lambda df: df.ConfirmedCases > 0)
         .dropna(0)
         .groupby('Country/Region')
         .apply(lambda df: (df
                            .sort_values('Date')
                            .assign(week_of_infection = lambda df: pd.np.arange(df.shape[0]))))
         .where(lambda df: df.dayofweek >= 6)
         .dropna(0)
         .reset_index(drop=True)
         .merge(countries, left_on='Country/Region', right_on='name'))
weeks




weeks.hvplot.line(x='Date', y='ConfirmedCases', by='Country/Region', title='Confirmed Cases', width=800, height=400, legend=False)




gpd_weeks = gpd.GeoDataFrame(weeks, geometry='geometry')
gpd_weeks.hvplot(geo=True, c=gpd_weeks.ConfirmedCases, title='Confirmed Cases', cmap='Spectral_r') +gpd_weeks.hvplot(geo=True, c=gpd_weeks.Fatalities, title='Fatalities', cmap='Spectral_r')




gpd_weeks.hvplot(geo=True, c=gpd_weeks.ConfirmedCases / gpd_weeks.pop_est, title='Confirmed Cases per Capita', cmap='Spectral_r') +gpd_weeks.hvplot(geo=True, c=gpd_weeks.Fatalities / gpd_weeks.pop_est, title='Fatalities per Capita', cmap='Spectral_r')




gpd_weeks.hvplot(geo=True, color=(gpd_weeks.ConfirmedCases / gpd_weeks.pop_est) / (gpd_weeks.pop_est / gpd_weeks.area), title='Confirmed Cases per Capita, per Population Density', cmap='Spectral_r') +gpd_weeks.hvplot(geo=True, color=(gpd_weeks.Fatalities / gpd_weeks.pop_est) / (gpd_weeks.pop_est / gpd_weeks.area), title='Fatalities per Capita, per Population Density', cmap='Spectral_r')




week_of_infection = (weeks)

percapita_perdensity = (gpd_weeks
                         .assign(infectionspercapita_populationdensity = lambda df: (df.ConfirmedCases / df.pop_est) / (df.pop_est / df.area))
                         .pipe(pd.DataFrame)
                         .groupby('Country/Region')
                         .apply(lambda df: (df
                                            .sort_values('Date')
                                            .assign(week_of_infection = lambda df: pd.np.arange(df.shape[0])))))

percapita_perdensity_top_mask = (percapita_perdensity
                            .groupby('Country/Region')
                            .max()
                            .nlargest(10, 'infectionspercapita_populationdensity')
                            .sort_values('infectionspercapita_populationdensity', ascending=True)
                            .index
                            .to_series())

percapita_perdensity_top = (percapita_perdensity
                            .merge(percapita_perdensity_top_mask,
                                   left_on='Country/Region',
                                   right_index=True, how='right'))

((percapita_perdensity_top
 .hvplot.line(x='Date', y='infectionspercapita_populationdensity',
              by='Country/Region',
              xlabel='Date', ylabel = 'Infections per Capita, per Population Density',
              title='Top 10 Confirmed Cases Per Capita, per Population Density',
              width=800, height=400,
              legend='right', logy=False)) + \

(percapita_perdensity_top
 .hvplot.line(x='week_of_infection', y='infectionspercapita_populationdensity',
              by='Country/Region',
              xlabel='Week of Local Infection', ylabel = 'Infections per Capita, per Population Density',
              title='Top 10 Confirmed Cases Per Capita, per Population Density by Week of Local Infection',
              width=800, height=400,
              legend='right', logy=False)))




X = pd.concat([gpd_weeks.week_of_infection.to_frame(), pd.get_dummies(gpd_weeks.loc[:,'Country/Region'])], axis=1).assign(const = 1)
y = ((gpd_weeks.ConfirmedCases / gpd_weeks.pop_est) / (gpd_weeks.pop_est / gpd_weeks.area)).apply(np.log).to_frame()

model = OLS(y, X).fit()
model.summary()




gpd_weeks_coef = gpd_weeks.merge(model.params.rename('coefficient').to_frame(), left_on='Country/Region', right_index=True)
gpd_weeks_coef.hvplot(geo=True, color='coefficient', 
                      title='Excess Growth in Confirmed Cases per Capita, per Population Density controlling for Week of Infection in Country', 
                      width=1000, height=600,
                      cmap='Spectral_r')

