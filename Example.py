# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 08:52:06 2020

@author: Daniel
"""
import os
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from matplotlib.patches import Rectangle
import re
import pickle
import signaltools as st
from astropy.timeseries import LombScargle
from scipy import signal


#%% load data
cwd = os.getcwd()
folder = os.path.join(cwd,"test_data")
file = "baldry"

raw_data = pd.read_csv(folder + '/' + file + '_data.csv')
#%%
baro_col = raw_data.columns[raw_data.columns.str.contains('pressure', case=False)][0]
dt_col = raw_data.columns[raw_data.columns.str.contains('datetime', case=False)][0]
raw_data[dt_col] = pd.to_datetime(raw_data[dt_col],dayfirst=True)

dnum_col = pd.to_numeric(raw_data[dt_col])
dnum_col = (dnum_col-dnum_col[0])
dnum_col = dnum_col/ 10**9 # from ns to seconds
dnum_col = dnum_col/(60*60*24) # to days

site = raw_data["BH3 [mAHD]"]
baro = raw_data[baro_col]

#%% least squares harmonic analysis
et_fqs = {'Q1': 0.893244, 'O1': 0.929536, 'M1': 0.966446, 'P1': 0.997262, 'S1': 1.0, 'K1': 1.002738, 'N2': 1.895982, 'M2': 1.932274, 'S2': 2.0, 'K2': 2.005476}
at_fqs = {'P1': 0.997262, 'S1': 1.0, 'K1': 1.002738, 'S2': 2.0, 'K2': 2.005476}

#%% Plotting
plt.style.use('ggplot')

# Barometric Pressure
fig, ax = plt.subplots(ncols= 1,figsize=(12.0,3.0), sharex = True) # ,sharey=True
ax = plt.plot(raw_data[dt_col],baro, color = "black")
plt.ylabel("Barometric Pressure [kPa]")

# Site
fig, ax = plt.subplots(ncols= 1,figsize=(12.0,3.0), sharex = True) # ,sharey=True
ax = plt.plot(raw_data[dt_col],site, color = "blue")
plt.ylabel("Pressure Head [m]")

#%% With td
idx = ~np.isnan(site.values)
baro_dt  = np.diff(dnum_col.iloc[idx])
dt_time  = raw_data.loc[idx,dt_col]
# Barometric Pressure
fig, ax = plt.subplots(nrows= 2,figsize=(8.0,4.0), sharex = True) # ,sharey=True
ax[0].plot(raw_data[dt_col],site, color = "blue")
ax[1].scatter(dt_time[1:],np.round(1.0/baro_dt,0), color = "red",s=4)
ax[1].set_yscale("log")
ax[1].set_ylabel("Sampling\nfrequency\n[$s * d^{-1}$]")
ax[0].set_ylabel("Barometric\nPressure\n[m]")
fig.autofmt_xdate()

#%% Method
import matplotlib.dates as mdates
years = mdates.YearLocator()
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

time = dnum_col.values
rel_time = time - time[0]

at_freqs = np.array(list(at_fqs.values()))
et_freqs = np.array(list(et_fqs.values()))

fqs_num = et_freqs
# detrending
detrend_day = 3

fig, ax = plt.subplots(nrows = 2,ncols= 2,figsize=(8.0,3.0),sharex="col") # ,sharey=True

for i,s in zip(range(2),[baro,site]):
    detrend_data = st.lin_window_ovrlp(time, s.values, detrend_day)
    # lomb scargle
    lsp_fqs, lsp_pow = LombScargle(rel_time, detrend_data).autopower(minimum_frequency=0.5, maximum_frequency=2.5)
    ax[1,0].plot(raw_data[dt_col],site, color = "blue")
    ax[1,0].set_ylabel("Pressure\nHead [m]")
    ax[1,0].set_xlabel("Time [date]")
    
    ax[0,0].plot(raw_data[dt_col],baro, color = "black")
    ax[0,0].set_ylabel("Barometric\nPressure [m]")
    # format the ticks
    ax[i,0].xaxis.set_major_locator(years)
    ax[i,0].xaxis.set_major_formatter(years_fmt)
    ax[i,0].xaxis.set_minor_locator(months)
    # format the coords message box
    ax[i,0].format_xdata = mdates.DateFormatter('%Y-%m-%d')
        
    ax[i,1].plot(lsp_fqs, np.sqrt(lsp_pow))
    ax[i,1].set_ylim(bottom=0)
    ax[i,1].set_ylabel('$\sqrt{Power}$')
    ax[1,1].set_xlabel("Frequency [cpd]")
    
    #ax[i,1].plot(fqs_num, result.amp_est, 'gx')

fig.autofmt_xdate()   
plt.tight_layout()

#%% date range
idx = ((raw_data[dt_col] >= pd.to_datetime('2006-09-01 00:00')) & (raw_data[dt_col] <= pd.to_datetime('2006-12-30 23:59')))
y_orig = baro[idx]
time_de = dnum_col.iloc[idx.values]
# create a big data gap for testing
y_gap = y_orig.copy()
y_gap[500:1000] = np.nan

y_demean = y_orig - np.mean(y_orig)
y_demean_gap = y_gap - np.mean(y_gap)

# Site
fig, ax = plt.subplots(ncols= 1,figsize=(12.0,3.0), sharex = True) # ,sharey=True
ax = plt.plot(time_de,y_orig, color = "blue")
plt.ylabel("Pressure Head [m]")

#%%
length = 3.2
stopper = 3
n_ovrlp = 5 # number of window overlaps (i.e. 12 equals a window step-size of 2h for 24 samples per day)

y_detrend = st.lin_window_ovrlp(time_de.values, y_gap.values, length, stopper,n_ovrlp)
#%%
#y1, y2 = linear_detrend(time_de.values, y_orig.values, length)
#% prepare the figure
fig = plt.figure(figsize=(10, 8))
gs0 = gridspec.GridSpec(3, 1, height_ratios=[1,1,1], hspace=0.2)

ax00 = fig.add_subplot(gs0[0])
ax00.plot(time_de, y_orig, label='Signal', lw=0.75,color = "black")
#ax00.set_ylabel()

ax1 = fig.add_subplot(gs0[1])
ax1.plot(time_de, y_demean_gap, label='normalized', lw=0.75,color="black")
#ax1.plot(data.index, y_demean - y1, label='win_detrend', lw=0.75)
ax1.plot(time_de, y_demean_gap - y_detrend, label='win_detrend_ovrlp_gap', lw=0.75, color="red")
ax1.legend(fontsize=10,loc='upper right')

ax2 = fig.add_subplot(gs0[2])
#ax2.plot(data.index, y1, label='win_detrend', lw=0.5)
ax2.plot(time_de, y_detrend, label='win_detrend_ovrlp_gap', lw=0.5,color="red")
ax2.legend(fontsize=10,loc='upper right')

ax2.set_xlabel('Time [days]')
ax2.set_ylabel('Detrended')

#%% save plot to file
fig.align_ylabels()
fig.tight_layout()
fig.savefig('detrend.png', dpi=150)
#%%
#ax[1].plot(fqs_num, result.amp_est, 'rx')
#ax[1].set_ylim(bottom=0)
#ax[1].set_ylabel('Amplitudes')
#ax[0].set_xlabel('Frequency [cpc]')