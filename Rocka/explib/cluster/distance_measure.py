# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
import time

from fastdtw import fastdtw

from scipy.spatial.distance import euclidean

__all__ = ['SBD', 'fast_dtw', 'l1_norm', 'Euclidean', 'SBD_no_shift']

# arr1 = [3,4,9,12,9,4,3,-1,8,9,6,0,0,0]
# arr1 = [-2,0.5,1.2,5,-0.5,1]
# #arr2 = [4,9,4,3,3]
# #arr2 = [3,4,9,4,3]
# arr2 = [3,3,4,9,4]
# arr2 = [0,0,0,3,4,9,12,9,4,3,-1,8,9,6]
# arr2 = [2,-2,0.5,1.2,5,-0.5]

# arr1 = np.arange(1,10)
# arr2 = arr1 + 1


# calculate ||x||
def l2_norm(arr):
    sum = 0
    for item in arr:
        sum += (item * item)
    return np.sqrt(sum)

# l2_norms = {}
# l2_norms[1] = l2_norm(arr1)
# l2_norms[2] = l2_norm(arr2)


# l1 norm distance measure
def l1_norm(arr1, arr2):
    return np.sum(np.fabs(arr1 - arr2))/len(arr1)


def Euclidean(arr1, arr2):
    return euclidean(arr1, arr2)


# cross-correlation measure and shape-based distance
def SBD(arr1, arr2):
    length = 2 ** (math.ceil(math.log(2 * len(arr1) - 1, 2)))
    m = len(arr1)
    # use fft and ifft to accelerate the cross-correlation calculation.
    value = np.fft.ifft(np.prod([np.fft.fft(arr1, length), np.conj(np.fft.fft(arr2, length))], axis=0))
    #print(value)
    r_max_val = np.real(np.max(value[:m]))
    l_max_val = np.real(np.max(value[-m+1:]))
    max_val = max(r_max_val, l_max_val)
    # slide arr2 through arr1
    if r_max_val >= l_max_val:              #positive lag, y shift towards right
        index = np.argmax(value[:m])
        shift = index
    elif r_max_val < l_max_val:             #negative lag, y shift towards left.
        index = np.argmax(value[-m+1:])
        shift = index + 1 - m
    else:
        raise ValueError(arr1 + "index error.")

    dist = 1 - max_val/(l2_norm(arr1) * l2_norm(arr2))

    if dist<0:
        dist=0

    if shift >= 0:
        y = np.append(np.zeros(shift), arr2[:m-shift])
    else:
        y = np.append(arr2[-shift:], np.zeros(-shift))
    return dist, y, shift


# # cross-correlation measure and shape-based distance
# def SBD(arr1, arr2):
#     length = 2 ** (math.ceil(math.log(2 * len(arr1) - 1, 2)))
#     m = len(arr1)
#     # use fft and ifft to accelerate the cross-correlation calculation.
#     value = np.fft.ifft(np.prod([np.fft.fft(arr1, length), np.conj(np.fft.fft(arr2, length))], axis=0))
#     #print(value)
#     max_val = np.real(np.max(value))
#     index = np.argmax(value)
#     #print(index)
#     dist = 1 - max_val/(l2_norm(arr1) * l2_norm(arr2))
#     # the index is no more than 2m-1. (slide arr2 through arr1)
#     if index < m:                       #positive lag, y shift towards right.
#         shift = index
#     elif length - index < m:            #negative lag, y shift towards left.
#         shift = index - length
#     else:
#         raise ValueError(arr1 + "index error.")
#
#     if shift >= 0:
#         y = np.append(np.zeros(shift), arr2[:m-shift])
#     else:
#         y = np.append(arr2[-shift:], np.zeros(-shift))
#     return dist, y, shift


def SBD_no_shift(arr1, arr2):
    length = 2 ** (math.ceil(math.log(2 * len(arr1) - 1, 2)))
    m = len(arr1)
    # use fft and ifft to accelerate the cross-correlation calculation.
    value = np.fft.ifft(np.prod([np.fft.fft(arr1, length), np.conj(np.fft.fft(arr2, length))], axis=0))
    # print(value)
    max_val = np.real(value[0])
    index = 0
    # print(index)
    dist = 1 - max_val/(l2_norm(arr1) * l2_norm(arr2))

    if dist<0:
        dist=0

    # the index is no more than 2m-1. (slide arr2 through arr1)
    if index < m:                       #positive lag, y shift towards right.
        shift = index
    elif length - index < m:          #negative lag, y shift towards left.
        shift = index - length
    else:
        raise ValueError(arr1 + "index error.")

    if shift >= 0:
        y = np.append(np.zeros(shift), arr2[:m-shift])
    else:
        y = np.append(arr2[-shift:], np.zeros(-shift))
    return dist, y, shift


def fast_dtw(arr1, arr2, dist_measure):
    distance, path = fastdtw(arr1, arr2, dist=dist_measure)
    distance = distance / (len(arr1) + len(arr2))
    return distance

# start = time.time()
# dist, y, shift = SBD(arr1,arr2)
# end = time.time()
# print(end - start)
#
# start = time.time()
# dist = fastdtw(arr1, arr2, dist=euclidean)
# end = time.time()
# print(end - start)
# print(dist, y, shift)
