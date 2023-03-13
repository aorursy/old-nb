#!/usr/bin/env python
# coding: utf-8



#----- Matrix
import numpy as np

# ---- Image Analysis
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
from skimage import measure
import scipy as sp
from scipy.signal import argrelextrema

# ---- Image Plotting
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pylab import rcParams

import pandas as pd
from sklearn import metrics

# ---- Networking
import cv2
import networkx as nx
from collections import defaultdict
from itertools import chain




img = mpimg.imread('../input/images/1562.jpg')
img = np.divide(img,np.max(np.max(img)))
skeleton = skeletonize(img)




def zhang_suen_node_detection(skel):
    
    def check_pixel_neighborhood(x, y, skel):
        
        accept_pixel_as_node = False
        item = skel.item
        p2 = item(x - 1, y) / 255
        p3 = item(x - 1, y + 1) / 255
        p4 = item(x, y + 1) / 255
        p5 = item(x + 1, y + 1) / 255
        p6 = item(x + 1, y) / 255
        p7 = item(x + 1, y - 1) / 255
        p8 = item(x, y - 1) / 255
        p9 = item(x - 1, y - 1) / 255
        
        components = (p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) +                      (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1) +                      (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) +                      (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1)
        if (components >= 3) or (components == 1):
            accept_pixel_as_node = True
        return accept_pixel_as_node

    graph = nx.Graph()
    w, h = skel.shape
    item = skel.item
    for x in range(1, w - 1):
        for y in range(1, h - 1):            
            if item(x, y) != 0 and check_pixel_neighborhood(x, y, skel):
                graph.add_node((x, y))
    return graph




graph = zhang_suen_node_detection(skeleton*255)




def breadth_first_edge_detection(skel, segmented, graph):

    def neighbors(x, y):
        item = skel.item
        width, height = skel.shape
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if (dx != 0 or dy != 0) and                                         0 <= x + dx < width and                                         0 <= y + dy < height and                                 item(x + dx, y + dy) != 0:
                    yield x + dx, y + dy

    # compute edge length
    label_node = dict()
    queues = []
    label = 1
    label_length = defaultdict(int)
    for x, y in graph.nodes_iter():
        for a, b in neighbors(x, y):
            label_node[label] = (x, y)
            label_length[label] = 1.414214 if abs(x - a) == 1 and                                               abs(y - b) == 1 else 1
            queues.append((label, (x, y), [(a, b)]))
            label += 1

    # bfs over the white pixels.
    # One phase: every entry in queues is handled
    # Each label grows in every phase.
    # If two labels meet, we have an edge.
    edges = set()
    edge_trace = np.zeros(skel.shape, np.uint32)
    edge_value = edge_trace.item
    edge_set_value = edge_trace.itemset
    label_histogram = defaultdict(int)

    while queues:
        new_queues = []
        for label, (px, py), nbs in queues:
            for (ix, iy) in nbs:
                value = edge_value(ix, iy)
                if value == 0:
                    edge_set_value((ix, iy), label)
                    label_histogram[label] += 1                    
                    label_length[label] += 1.414214 if abs(ix - px) == 1 and                                                        abs(iy - py) == 1 else 1
                    new_queues.append((label, (ix, iy), neighbors(ix, iy)))
                elif value != label:
                    edges.add((min(label, value), max(label, value)))
        queues = new_queues

    # compute edge diameters
    diameters = 1
    # add edges to graph
    for l1, l2 in edges:
        u, v = label_node[l1], label_node[l2]
        if u == v:
            continue
        
        graph.add_edge(u, v, pixels=label_histogram[l1] + label_histogram[l2],
                       length=label_length[l1] + label_length[l2],
                       width=1,
                       width_var=1)
    return graph




graph = breadth_first_edge_detection(skeleton, img, graph)




NODESIZESCALING = 750
EDGETRANSPARENCYDIVIDER = 5
EDGETRANSPARENCY = False


def draw_graph(image, graph):
    tmp = draw_edges(image, graph)
    node_size = int(np.ceil((max(image.shape) / float(NODESIZESCALING))))
    return draw_nodes(tmp, graph, max(node_size, 1))


def draw_nodes(img, graph, radius=1):
    for x, y in graph.nodes_iter():
        cv2.rectangle(img, (y - radius, x - radius), (y + radius, x + radius),
                     (255, 0, 0), -1)
    return img


def draw_edges(img, graph, col=(255, 255, 255)):
    edg_img = np.copy(img)

    max_standard_deviation = 0   

    for (x1, y1), (x2, y2) in graph.edges_iter():
        start = (y1, x1)
        end = (y2, x2)
        diam = graph[(x1, y1)][(x2, y2)]['width']
        # variance value computed during graph detection
        width_var = graph[(x1, y1)][(x2, y2)]['width_var']
        # compute edges standard deviation by applying sqrt(var(edge))
        standard_dev = np.sqrt(width_var)
        if diam == -1: diam = 2
        diam = int(round(diam))
        #mymod
        diam = 1
        if diam > 255:
            print('Warning: edge diameter too large for display. Diameter has been reset.')
            diam = 255
        else:
            # simply draw a red line since we are not in the edge transparency mode
            cv2.line(edg_img, start, end, col, diam)

    edg_img = cv2.addWeighted(img, 0.5, edg_img, 0.5, 0)

    MAXIMUMSTANDARDDEVIATION = 0

    return edg_img




data = draw_graph(img, graph)




fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4.5), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

ax1.imshow(img, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('Original', fontsize=20)

ax2.imshow(skeleton, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Skeleton', fontsize=20)

ax3.imshow(data, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Found Network', fontsize=20)

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.98,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()




# filter kernel to perform the dilation
struct = [[ 0., 0., 1., 1., 0., 0.],
          [ 0., 1., 1., 1., 1., 0.],  
          [ 1., 1., 1., 1., 1., 1.], 
          [ 1., 1., 1., 1., 1., 1.], 
          [ 1., 1., 1., 1., 1., 1.], 
          [ 0., 1., 1., 1., 1., 0.],
          [ 0., 0., 1., 1., 0., 0.]]

# ----------------------------------------------------- Init ---
img = mpimg.imread('../input/images/1562.jpg')
img = np.divide(img,np.max(np.max(img))) # Normalize 


dilation = ndi.morphology.binary_dilation(img, structure=struct).astype(img.dtype)
img = dilation
img = np.divide(img,np.max(np.max(img))) # Normalize
skeleton = skeletonize(img)

# Network Extraction
graph = zhang_suen_node_detection(skeleton*255)
graph = breadth_first_edge_detection(skeleton, img, graph)
data = draw_graph(img, graph)




fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4.5), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

ax1.imshow(img, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('Original', fontsize=20)

ax2.imshow(skeleton, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Skeleton', fontsize=20)

ax3.imshow(data, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Found Network', fontsize=20)

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.98,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()

