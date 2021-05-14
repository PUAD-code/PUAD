# -*- coding: utf-8 -*-
import os

# base path of the whole experiment
DATA_ROOT = '/home/jialingxiang/NewDTWFrame/Data'  # type: str

# relative path under DATA_ROOT, store the cluster result
OUTPUT_PATH = 'result'  # type: str

# start and end timestamp of the KPIs for cluster
START_TS = 1521561900  # type: int
END_TS = 1524153000  # type:  int

# window size to smooth the KPIs
WINDOW_SIZE = 120  # type: int

# The distance measure to measure similarity between time series. Can be SBD, DTW, L1, Euclidean.
DIST = 'SBD'  # type: str

# allow shift when calculate SBD or not
NO_SHIFT = False  # type: bool

# Recalculate the similarity matrix for cluster dataset
RETRAIN = True  # type: bool

# Evaluate the cluster and classify result based on ground-truth
EVALUATE = False  # type: bool

# Ignore outlier when calculating evaluation metrics in evaluation step.
IGNORE = True  # type: bool

# parameter max_radius for determining the density radius in cluster
#MAX_RADIUS = 0.25  # type: float

# parameter inflection threshold for finding the flat part on k-dis curve.
INFLECT_THRESH = 0.005  # type: float

# sample rate for random sampling the input dataset to get the training dataset. Between (0,1]
SAMPLE_RATE = 0.8  # type: float

# start and end timestamp of KPIs for plotting the cluster results. Please do not include the first and last day on KPIs
# due to the cold-start.
DRAW_START = 1521580000  # type: int
DRAW_END = 1524134900  # type: int
