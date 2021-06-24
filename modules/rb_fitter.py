#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 13:34:18 2021

@author: daniel
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from matplotlib.offsetbox import AnchoredText

def exp_fit(m, alpha, A, B):
    return A*alpha**m+B

def get_survival_prob(result_list, nCliffs):
    shots=result_list[0].to_dict()['results'][0]['shots']
    n_qubits=result_list[0].to_dict()['results'][0]['header']['n_qubits']
    survival_prob=np.zeros((len(result_list),len(nCliffs)))
    for i, res in enumerate(result_list):
        for j, m in enumerate(res.get_counts()):
            survival_prob[i][j]= m['0'*n_qubits]/shots
    return survival_prob

def get_fit_param(xdata, ydata, sigma):   
    popt, pcov = curve_fit(exp_fit, xdata, ydata, p0=[0.98, 1, 0], sigma=sigma, absolute_sigma=False)
    return popt, pcov

def plot_fit(popt, pcov, result_list, nCliffs):
    alpha, A, B = popt
    plot_xs=np.linspace(1,200,1000)
    std=np.sqrt(pcov[0,0])
    textstr = '\n'.join((
    r'$\alpha=%.3f$' % (alpha, ),
    r'$\mathrm{std}=%.2E$' % (std, )))
    
    fig, ax = plt.subplots()
    ax.plot(nCliffs, np.transpose(get_survival_prob(result_list, nCliffs)),'kx')
    
    ax.plot(plot_xs, exp_fit(plot_xs, alpha, A , B))
    anchored_text = AnchoredText(textstr, loc=1)
    ax.add_artist(anchored_text)
    
def plot_fit2(popt, std, xdata, ydata, sigma_max):
    alpha, A, B = popt
    plot_xs=np.linspace(1,max(xdata),1000)
    textstr = '\n'.join((
    r'$\alpha=%.3f$' % (alpha, ),
    r'$\mathrm{std}=%.2E$' % (std, )))
    
    fig, ax = plt.subplots()
    ax.errorbar(xdata, ydata, yerr=sigma_max, marker='x')
    
    ax.plot(plot_xs, exp_fit(plot_xs, alpha, A , B))
    anchored_text = AnchoredText(textstr, loc=1)
    ax.add_artist(anchored_text)    
    

#%%
def get_conf_bound(m, r):
    var=m*r
    # var=(m*r)**2+(7./4.)*m*r**2
    return np.sqrt(var)
  
#%%  
        
    
    