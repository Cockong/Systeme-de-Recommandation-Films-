#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 17:49:28 2019

@author: rthiebaut
"""
import time

def a(**kwds):
    return 4

def b(i):
    return i


def temps(a,**kwds):
    start = time.perf_counter() # first timestamp
    print(a(**kwds))
    end = time.perf_counter() # second timestamp
    return end-start


print(temps(b,i=112))




"""
start = time.perf_counter() # first timestamp

# we place here the code we want to time


end = time.perf_counter() # second timestamp

elapsed = end - start
print("elapsed time = {:.12f} seconds".format(elapsed))

start = time.perf_counter() # first timestamp

# we place here the code we want to time

    
end = time.perf_counter() # second timestamp

elapsed = end - start
print("elapsed time = {:.12f} seconds".format(elapsed))
"""