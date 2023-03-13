#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install pydub')




#Libraries Required
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly_express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import wave
import os
import requests
import re
from pydub import AudioSegment
import IPython.display as ipd
import struct
from scipy.io import wavfile as wav
from colorama import Fore, Back, Style
import requests
import json
from bs4 import BeautifulSoup
import ipywidgets as widgets
from itertools import product




df_train = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')
df_test = pd.read_csv('/kaggle/input/birdsong-recognition/test.csv')
df = pd.read_csv('/kaggle/input/birdsongrecognitiondetails/bird_details.csv')
media_path = '/kaggle/input/birdsong-recognition/train_audio/'




def get_elevation(val):
    """Derive the elevation value from the string. Also, I have 
    kept negative elevation values as below sea level is also a possibility."""
    l = re.findall('[~\?]?(-?\d+[\.,]?\d*)-?(\d*)',val)
    val1=0
    val2=0
    if l:
        if l[0][0]:
            val1=float(l[0][0].replace(',',''))
        if l[0][1]:
            val2=float(l[0][1].replace(',',''))
        if val1!=0 and val2!=0:
            return (val1+val2)/2
        return val1
    else:
        return float('nan')
df_train.elevation=df_train.elevation.apply(lambda x: get_elevation(x))




def wav_plotter(full_path,data):   
    rate, wav_sample = wav.read(full_path)
    wave_file = open(full_path,"rb")
    riff_fmt = wave_file.read(36)
    bit_depth_string = riff_fmt[-2:]
    bit_depth = struct.unpack("H",bit_depth_string)[0]
    print(Fore.CYAN+data['title'].upper())
    print('_'*len(data['title']))
    print('')
    print('Scientific Name:',data['sci_name'])
    print('Country recorded: ',data['country'])
    print('Recordist: ',data['author'])
    print('Rating: ',data['rating'])
    print('Sampling rate: ',rate,'Hz')
    print('Bit depth: ',bit_depth)
    print('Number of channels: ',wav_sample.shape[1] if len(wav_sample.shape)>1 else 1)
    print('Duration: ',wav_sample.shape[0]/rate,' second')
    print('Number of samples: ',len(wav_sample))
    plt.figure(figsize=(12, 4))
    plt.plot(wav_sample)
    return ipd.Audio(full_path)

def plot_wav(sp):
    data = df_train[df_train['species']==sp]
    idx = np.random.choice(data.index,1)[0]
    sound_data = data.loc[idx,:]
    src = os.path.join('/kaggle/input/birdsong-recognition/train_audio/',sound_data['ebird_code'],sound_data['filename'])
    sound_mp3 = AudioSegment.from_mp3(src)
    filename=sound_data['filename'].split('.')[0]+'.wav'
    sound_mp3.export(filename,format='wav')
    return wav_plotter(filename,sound_data)




plot_wav('Alder Flycatcher')




plot_wav('House Wren')




plot_wav('American Robin')




plot_wav('Ovenbird')




plot_wav('Northern Flicker')




#Common backdrop to be used for all the plots
PAPER_BGCOLOR='rgb(255,255,255)'
PLOT_BGCOLOR='rgb(255,255,255)'




ratings = df_train.groupby('rating',as_index=False)['title'].count().sort_values('rating')
fig = go.Figure()
fig.add_trace(go.Bar(x=ratings['rating'],y=ratings['title'],marker_line_color='black',marker_line_width=1.5,text=ratings['title'],textposition='auto'))
fig.update_layout(template='seaborn',height=300,title='Ratings count',paper_bgcolor=PAPER_BGCOLOR,plot_bgcolor=PLOT_BGCOLOR,
                 xaxis=dict(title='Ratings',nticks=20,mirror=True,linewidth=2,linecolor='black'),
                 yaxis=dict(title='Counts',mirror=True,linewidth=2,linecolor='black',gridcolor='darkgrey'))
fig.show()




df_train['year'] = df_train['date'].apply(lambda x: int(x.split('-')[0]))
group = df_train.groupby(['year','species'],as_index=False).agg({'rating':'mean','ebird_code':'count'})    .sort_values(['year','ebird_code'],axis=0)
group = group[group['year']>=1979]

data = np.array(list(product(group.year.unique().tolist(),group.species.unique().tolist())))
df_data = pd.DataFrame(np.vstack(data), columns=['year','species'])
df_data['year'] = df_data['year'].astype(int)
df_data = pd.merge(df_data,group,on=['year','species'],how='left')
df_data.fillna(0,inplace=True)
df_data.rename(columns={'ebird_code':'Recordings'},inplace=True)
df_data.Recordings = df_data.Recordings.astype(int)
fig = px.bar(df_data,y='species',x='Recordings',animation_frame='year',orientation='h')
fig.update_layout(template='seaborn',height=800,width=700,title='Recordings registered per year',
                  paper_bgcolor=PAPER_BGCOLOR,plot_bgcolor=PLOT_BGCOLOR,
                 xaxis=dict(range=[0,48],title='Number of Recordings',mirror=True,linewidth=2,linecolor='black',gridcolor='darkgrey'),
                 yaxis=dict(title='Bird Species',mirror=True,linewidth=2,linecolor='black'))
fig.show()




ratings_species = df_train.groupby('species',as_index=False)    .agg({'rating':'mean','country':'nunique','author':'nunique','duration':'mean','elevation':'mean'})
ratings_species['rating']=np.round(ratings_species['rating'],2)
ratings_species['duration']=np.round(ratings_species['duration'],2)
ratings_species['elevation']=np.round(ratings_species['elevation'],2)
ratings_species.sort_values('rating',ascending=False,inplace=True)




fig = make_subplots(rows=3,cols=2,specs=[[{'type':'table','colspan':2},None],[{},{}],[{},{}]],
                   vertical_spacing=0.03,horizontal_spacing=0.03)
fig.add_trace(go.Table(
        columnorder=[1,2,3,4,5,6],
        columnwidth=[170,60,130,90,130,120],
        header=dict(
            values=["<b>Species</b>", "<b>Mean Rating</b>", "<b>No. of Countries</b><br>where bird is found",
                    "<b>No. of Recordists</b>",'<b>Mean Duration</b><br>of chirp in seconds',
                   "<b>Mean Elevation</b><br>in meters"],
            line_color='darkslategray',
            fill_color='royalblue',
            font=dict(color='white', size=10),
            align=['center']
        ),
        cells=dict(
            values=[ratings_species[k].tolist() for k in ratings_species.columns],
            align = "center",
            line_color='darkslategray',
            fill=dict(color=['paleturquoise', 'white']))
    ),1,1)
fig.add_trace(go.Scatter(name='Author',x=ratings_species.rating,y=ratings_species.author,mode='markers',
                        marker_size=10,marker_line_width=1,
                        text=ratings_species['species'],
                        textposition='bottom center'),2,1)
fig.add_trace(go.Scatter(name='Duration',x=ratings_species.rating,y=ratings_species.duration,mode='markers',
                        marker_size=10,marker_line_width=1,
                        text=ratings_species['species'],
                        textposition='bottom center'),2,2)
fig.add_trace(go.Scatter(name='Country',x=ratings_species.rating,y=ratings_species.country,mode='markers',
                        marker_size=10,marker_line_width=1,
                        text=ratings_species['species'],
                        textposition='bottom center'),3,1)
fig.add_trace(go.Scatter(name='Elevation',x=ratings_species.rating,y=ratings_species.elevation,mode='markers',
                        marker_size=10,marker_line_width=1,
                        text=ratings_species['species'],
                        textposition='bottom center'),3,2)

fig.update_xaxes(linecolor='black',linewidth=2,showline=True,
                 showgrid=False,mirror=True,ticks='inside',tickfont=dict(size=10),row=2,col=1)
fig.update_xaxes(linecolor='black',linewidth=2,showline=True,
                 showgrid=False,mirror=True,ticks='inside',tickfont=dict(size=10),row=2,col=2)
fig.update_xaxes(linecolor='black',linewidth=2,showline=True,
                 showgrid=False,mirror=True,ticks='inside',tickfont=dict(size=10),row=3,col=1)
fig.update_xaxes(linecolor='black',linewidth=2,showline=True,
                 showgrid=False,mirror=True,ticks='inside',tickfont=dict(size=10),row=3,col=2)

fig.update_yaxes(title_text='No. of Recordists',linecolor='black',linewidth=2,showline=True,
                 showgrid=False,mirror=True,ticks='inside',tickfont=dict(size=10),row=2,col=1)
fig.update_yaxes(title_text='Mean Duration',linecolor='black',linewidth=2,showline=True,side='right',
                 showgrid=False,mirror=True,ticks='inside',tickfont=dict(size=10),row=2,col=2)
fig.update_yaxes(title_text='No. of Countries',linecolor='black',linewidth=2,showline=True,
                 showgrid=False,mirror=True,ticks='inside',tickfont=dict(size=10),row=3,col=1)
fig.update_yaxes(title_text='Mean Elevation',linecolor='black',linewidth=2,showline=True,side='right',
                 showgrid=False,mirror=True,ticks='inside',tickfont=dict(size=10),row=3,col=2)

fig.update_layout(template='seaborn',width=700,height=800,title='Bird Chirp Rating Analysis',
                  showlegend=False,plot_bgcolor=PLOT_BGCOLOR,paper_bgcolor=PAPER_BGCOLOR)
fig.show()                                            




for graph in fig.data:
    graph_type = type(graph).__name__
    if graph_type == 'Bar' or graph_type == 'Scatter':
        print(graph.marker)
    




countries = df_train.groupby(['country','bird_seen'],as_index=False).agg({'title':'count','rating':'mean'})    .sort_values('title',ascending=False).reset_index()
countries = countries.loc[:50,:]
seen_color = {'yes':'rgb(93, 217, 93)','no':'rgb(239, 58, 56)'}
fig = go.Figure()
for seen in ['yes','no']:
    fig.add_trace(go.Bar(name=seen,y=countries[countries['bird_seen']==seen]['country'],
                         x=countries[countries['bird_seen']==seen]['title'],orientation='h',
                         marker_line_color='black',marker_line_width=1.5,
                         text=np.round(countries[countries['bird_seen']==seen]['rating'],2),textposition='outside',
                         marker_color=seen_color[seen]))
fig.update_layout(height=800,template='seaborn',paper_bgcolor=PAPER_BGCOLOR,plot_bgcolor=PLOT_BGCOLOR,barmode='stack',
                  hovermode='y unified',width=700,
                 xaxis=dict(title='Number of Recordings',type='log',mirror='allticks',linewidth=2,linecolor='black',
                            showgrid=True,gridcolor='darkgray'),
                 yaxis=dict(mirror=True,linewidth=2,linecolor='black',tickfont=dict(size=8)),
                 legend=dict(title='<b>Was the bird seen?</b>',x=0.71,y=0.95,bgcolor='rgba(255, 255, 255, 0)',
                             bordercolor='rgba(255, 255, 255, 0)'),
                 title='<b>Number of Recordings per Country [Top 50]</b><br>(Along with average ratings)')
fig.show()




from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("gmaps")
secret_value_1 = user_secrets.get_secret("mapboxtoken")
df_train.latitude = df_train.latitude.str.replace('Not specified','nan').astype(np.float16)
df_train.longitude = df_train.longitude.str.replace('Not specified','nan').astype(np.float16)




px.set_mapbox_access_token(secret_value_1)
fig = px.scatter_mapbox(df_train,
                lat='latitude',
                lon='longitude',
                size='duration',
                color='rating',
                hover_name='species',
                hover_data=['duration','country','elevation'],
                color_continuous_scale=px.colors.sequential.Viridis,
                mapbox_style='open-street-map',
                zoom=0.5)
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="mercator")
fig.update_layout(height=500,width=700,margin={"r":0,"t":50,"l":0,"b":0})
fig.update_layout(title='<b>Recording Locations</b>',template='seaborn',
                  hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"))
fig.show()




def normalize(x):
    xmin = x.min()
    xmax = x.max()
    return (x-xmin)/(xmax-xmin)

recordists = df_train.groupby('author',as_index=False).agg({'rating':'mean','species':'nunique','country':'nunique'}).sort_values(['species','rating'],ascending=False)
recordists['rating'] = np.round(recordists['rating'],2)
recordists['rating_norm'] = normalize(recordists['rating'])
recordists['species_norm'] = normalize(recordists['species'])
recordists['country_norm'] = normalize(recordists['country'])
recordists['total'] = recordists['species_norm']*0.5 + recordists['country_norm']*0.3 + recordists['rating_norm']*0.2
recordists.drop(['rating_norm','species_norm','country_norm'],axis=1,inplace=True)

fig = make_subplots(rows=2,cols=1,specs=[[{'type':'table','rowspan':1}],[{'rowspan':1}]],
                   vertical_spacing=0.03,horizontal_spacing=0.03,shared_xaxes=True)
fig.add_trace(go.Table(
        columnorder=[1,2,3,4],
        columnwidth=[250,150,150,150],
        header=dict(
            values=["<b>Recordist</b>", "<b>Mean Rating</b>", "<b>Species Covered</b>",
                    "<b>Countries Covered</b>"],
            line_color='darkslategray',
            fill_color='royalblue',
            font=dict(color='white', size=12),
            align=['center']
        ),
        cells=dict(
            values=[recordists[k].tolist() for k in recordists.columns[:-1]],
            align = "center",
            line_color='darkslategray',
            fill=dict(color=['paleturquoise', 'white']))
    ),1,1)

rec = recordists.nlargest(30,'total')
fig.add_trace(go.Bar(name='Species',x=rec.author,y=rec.species,
                    marker_line_width=1.5,
                    marker_line_color='black',
                    marker_color='#F1EA49'),2,1)
fig.add_trace(go.Bar(name='Countries',x=rec.author,y=rec.country,
                    marker_line_width=1.5,
                    marker_line_color='black',
                    marker_color='#3893D2',
                    text=rec.rating,
                    textposition='outside'),2,1)
fig.update_xaxes(linecolor='black',linewidth=2,showline=True,
                 showgrid=False,mirror=True,ticks='outside',tickfont=dict(size=10),row=2,col=1)
fig.update_yaxes(title='Species + Countries covered',linecolor='black',linewidth=2,showline=True,
                 showgrid=False,mirror=True,ticks='outside',tickfont=dict(size=10),row=2,col=1)
fig.update_layout(template='seaborn',width=700,height=800,title='Recordist Analysis',
                  legend=dict(title='<b>   Top 30<br>Recordists</b>',x=0.1,y=0.49,bgcolor='rgba(255, 255, 255, 0)',
                             bordercolor='rgba(255, 255, 255, 0)',orientation='h'),
                  plot_bgcolor=PLOT_BGCOLOR,paper_bgcolor=PAPER_BGCOLOR,barmode='stack',
                 hovermode='x unified')
fig.show()                                            




"""Covert mp3 to wav format"""
# path = '/kaggle/input/birdsong-recognition/train_audio/'
# out_dir = '/kaggle/working/train_audio_wav'
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)
# for i in range(df_train.shape[0]):
#     src = os.path.join(path,df_train.loc[i,'ebird_code'],df_train.loc[i,'filename'])
#     dst = os.path.join(out_dir,df_train.loc[i,'ebird_code'])
#     if not os.path.exists(dst):
#         os.makedirs(dst)
#     sound = AudioSegment.from_mp3(src)
#     os.chdir(dst)
#     sound.export(df_train.loc[i,'filename'].split('.')[0]+'.wav', format='wav')    

