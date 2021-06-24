#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 18:28:50 2021

@author: daniel
"""
import matplotlib.pyplot as plt

def plot_fit(result_list, rbfit, rb_opts):
    plt.figure(figsize=(15, 6))
    for seed_num, data in enumerate(result_list):#range(1,len(result_list)):
        # Add another seed to the data
        rbfit.add_data([data])
    
    maxnum=len(rb_opts['rb_pattern'])
    if maxnum==4:
        cols=2
        rows=2
    elif maxnum > 3:
        cols=3
        rows=1+(maxnum-int(maxnum%3))/3
    else:
        cols=maxnum
        rows=1
    print(rows)
    print(cols)
    
    axis=[plt.subplot(rows, cols, i) for i in range(1,maxnum+1)]
        

    for i in range(maxnum):
        pattern_ind = i
    
        # Plot the essence by calling plot_rb_data
        rbfit.plot_rb_data(pattern_ind, ax=axis[i], add_label=True, show_plt=False)
    
        # Add title and label
        axis[i].set_title('%d Qubit RB - after seed %d'%(len(rb_opts['rb_pattern'][i]), seed_num), fontsize=18)
   