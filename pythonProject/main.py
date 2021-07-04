# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import networkx as nx
import pylab
import numpy as np
import pandas as pd
import seaborn as sn
import math
from datetime import datetime
import calendar
import matplotlib.pyplot as plt
df = pd.read_csv("./a6c2de4a.csv")#读取共享单车数据
#tiqu 关键字段
#下边需要根据订单起点和终点经纬度确定起点和终点的唯一编码，这里是使用geopandas将订单OD点转为了点图层，然后提取O点和D点，以及计算OD之间的距离做为边的权重。
import pandas as pd
import geopandas as gpd
import os
from shapely.geometry import Point
df = df[['start_location_x', 'start_location_y', 'end_location_x', 'end_location_y']]
df['source'] = '(' + df['start_location_x'].map(str) + ',' + df['start_location_y'].map(str) + ')'
df['target'] = '(' + df['end_location_x'].map(str) + ',' + df['end_location_y'].map(str) + ')'
df['weight'] = ((df['start_location_x'] - df['end_location_x']).apply(lambda x: x ** 2) +
      (df['start_location_y'] - df['end_location_y']).apply(lambda x: x ** 2)).apply(lambda x: x**0.5)
# df['label'] = df['source']
# df['lng'] = df['start_location_x']
# df['lat'] = df['start_location_y']
# df_node = df[['label', 'lng', 'lat']]
df = df[['source', 'target', 'weight']]
df = df.head(6000)
# df_node = df_node.head(1000)
df.to_csv('out.csv',index=False)
# df_node.to_csv('out_node1.csv',index=False)


