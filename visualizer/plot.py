import matplotlib.pyplot as plt
import numpy as np


def draw_distance_geo_feat(geo_distance, feat_distance):
    geo_distance = geo_distance.flatten()
    feat_distance = feat_distance.flatten()
    plt.scatter(geo_distance, feat_distance)
    plt.show()
