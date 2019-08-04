#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 18:07:05 2019

@author: ankitgupta
"""

import matplotlib.pyplot as plt

def drawTriangle(fig, ax, points):
    '''
    fig, ax --> define the figure on which the plotting needs to be done
    points --> points on the triangle
    '''
    if fig is None:
        fig, ax = plt.subplots(1,1)
        
    