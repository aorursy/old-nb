
north_pole = (90,0)
weight_limit = 1000
sleigh_weight = 10


import pandas as pd
import numpy as np
from haversine import haversine

def weighted_trip_length(stops, weights): 
    tuples = [tuple(x) for x in stops.values]
    # adding the last trip back to north pole, with just the sleigh weight
    tuples.append(north_pole)
    weights.append(sleigh_weight)
    
    dist = 0.0
    prev_stop = north_pole
    prev_weight = sum(weights)
    for location, weight in zip(tuples, weights):
        dist = dist + haversine(location, prev_stop) * prev_weight
        prev_stop = location
        prev_weight = prev_weight - weight
    return dist

def weighted_reindeer_weariness(all_trips):
    uniq_trips = all_trips.TripId.unique()
    
    if any(all_trips.groupby('TripId').Weight.sum() > weight_limit):
        raise Exception("One of the sleighs over weight limit!")
 
    dist = 0.0
    for t in uniq_trips:
        this_trip = all_trips[all_trips.TripId==t]
        dist = dist + weighted_trip_length(this_trip[['Latitude','Longitude']], this_trip.Weight.tolist())
    
    return dist    



gifts = pd.read_csv('../input/gifts.csv')
sample_sub = pd.read_csv('https://raw.githubusercontent.com/wbyouth/kaggle/master/santa_sub.csv')

all_trips = sample_sub.merge(gifts, on='GiftId')

print(weighted_reindeer_weariness(all_trips))
