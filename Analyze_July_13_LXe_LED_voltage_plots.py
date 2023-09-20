#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 18:13:49 2023

@author: albertwang
"""
import sys
import numpy as np
# append necessary file paths, and change E -> D or vice versa
sys.path.append('/Users/albertwang/Desktop/nEXO')   
from MeasurementInfo import MeasurementInfo
from RunInfo import RunInfo
import heapq
from scipy import signal
from scipy.optimize import curve_fit
import AnalyzePDE
from AnalyzePDE import SPE_data
from AnalyzePDE import Alpha_data
import matplotlib.pyplot as plt
import matplotlib as mpl
import ProcessWaveforms_MultiGaussian
from ProcessWaveforms_MultiGaussian import WaveformProcessor as WaveformProcessor

#%%

def get_subtract_hist_mean(data1, data2, bias = '', numbins = 500, plot = False):
    if plot:
        plt.figure()
        (n, b, p) = plt.hist(data1, bins = numbins, density = False, label = 'LED-on', histtype='step')
        plt.axvline(x = np.mean(data1), color = 'blue')
        print('LED on hist: ' + str(np.mean(data1)))
        print('LED off hist: ' + str(np.mean(data2)))
        plt.axvline(x = np.mean(data2), color = 'orange')
        plt.hist(data2, bins = b, density = False, label = 'LED-off', histtype='step')
    counts1, bins1 = np.histogram(data1, bins = numbins, density = False)
    counts2, bins2 = np.histogram(data2, bins = bins1, density = False)
    centers = (bins1[1:] + bins1[:-1])/2
    subtracted_counts = counts1 - counts2
    # subtracted_counts[subtracted_counts < 0] = 0
    if plot:
        plt.step(centers, subtracted_counts, label = 'subtracted hist')
        plt.legend()
        
    norm_subtract_hist = subtracted_counts / np.sum(subtracted_counts)
    # weights = 1.0 / subtracted_counts / 
    mean_value = np.sum(centers * norm_subtract_hist)
    if plot:
        plt.axvline(x = mean_value, color = 'green')
        plt.title(f'The bias: {bias}, The mean value: {mean_value:0.3}')
    return mean_value
    # np.trapz(co)
    

#%% Test

test_high_bias = RunInfo(['/Users/albertwang/Desktop/nEXO/HDF Data/july_13th_LED_data/LED Operating V/405nm/Run_1689284451.hdf5'], do_filter = True, upper_limit = .5, baseline_correct = True, prominence = 0.008, plot_waveforms= False, is_led = True)
bias_test = test_high_bias.run_meta_data['/Users/albertwang/Desktop/nEXO/HDF Data/july_13th_LED_data/LED Operating V/405nm/Run_1689284451.hdf5']['RunNotes']
# test_high_bias.plot_hists('171', '.') #regular histogram
# test_high_bias.plot_peak_waveform_hist() #2D plot
# test_high_bias.plot_led_dark_hists('168', '.') #LED comparison plot
# print(len(test_high_bias.all_led_peak_data))
get_subtract_hist_mean(test_high_bias.all_led_peak_data, test_high_bias.all_dark_peak_data, bias = bias_test ,plot = True)

#%% 405nm Operating voltage analysis
files = ['Run_1689268742.hdf5','Run_1689272629.hdf5','Run_1689271255.hdf5','Run_1689272390.hdf5','Run_1689284451.hdf5','Run_1689269180.hdf5','Run_1689283942.hdf5','Run_1689285361.hdf5','Run_1689269572.hdf5','Run_1689270159.hdf5'] #2.5, 2.51V, 2.52V, 2.53, 2.54V2, 2.55V, 2.56V, 2.57 2.6V, 2.65V
proms = [0.008,0.008,0.008,0.008,0.008,0.01,0.01,0.01,0.015,0.15] #
upperlim = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.8] 
runs = []
led_voltages = []
 
for file in range(len(files)-1):
        run_spe = RunInfo(['/Users/albertwang/Desktop/nEXO/HDF Data/july_13th_LED_data/LED Operating V/405nm/' + files[file]], specifyAcquisition = False, do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file], is_led = True)
        led_voltage = run_spe.run_meta_data['/Users/albertwang/Desktop/nEXO/HDF Data/july_13th_LED_data/LED Operating V/405nm/' + files[file]]['RunNotes']
        led_voltages.append(float(led_voltage[:-1]))
        runs.append(run_spe)

#%% 
biases = [run.bias for run in runs]
a_LED = [np.mean(run.all_led_peak_data) for run in runs]
# print('Amp LED on: ' + str(a_LED[0]))
# print(len(runs[0].all_led_peak_data))
a_LED_err = [np.std(run.all_led_peak_data, ddof = 1) / np.sqrt(len(run.all_led_peak_data)) for run in runs]
a_dark = [np.mean(run.all_dark_peak_data) for run in runs]
# print('Amp LED off: ' + str(a_dark[0]))
a_dark_err = [np.std(run.all_dark_peak_data, ddof = 1) / np.sqrt(len(run.all_dark_peak_data)) for run in runs]
# do_subtract = True
# if do_subtract:
    # a_sub = [get_subtract_hist_mean(run.all_led_peak_data, run.all_dark_peak_data) for run in runs]
ratio = np.array([(float(len(run.all_led_peak_data)) - float(len(run.all_dark_peak_data))) / float(len(run.all_dark_peak_data)) for run in runs])
ratio_err = ratio * np.array([np.sqrt(1.0 / float(len(run.all_led_peak_data)) + 1.0 / float(len(run.all_dark_peak_data))) for run in runs])
# ratio = [run.led_ratio for run in runs]
# print(led_voltages)
# print(a_LED)
# print(a_dark)
# print(ratio)


#%%
# led_voltages = [float(volt[:-1]) for volt in led_voltages]
a_dark_err = np.array(a_dark_err)
dark_avg = np.mean(a_dark)
# dark_err = np.std(a_dark, ddof = 1) / np.sqrt(len(dark_avg))
dark_err = 1.0 / np.sqrt(np.sum(1.0 / (a_dark_err * a_dark_err)))


plt.figure()
x_values = led_voltages
y_values = a_LED
y2_values = a_dark

plt.fill_between(x_values, y1 = np.ones(len(x_values)) * dark_avg - dark_err, y2 = np.ones(len(x_values)) * dark_avg + dark_err, color = 'orange', alpha = 0.2)
plt.plot(x_values, np.ones(len(x_values)) * dark_avg, 'r-')
plt.errorbar(x_values, y_values, yerr = a_LED_err, fmt = '.', label = 'LED on')
plt.errorbar(x_values, y2_values, yerr = a_dark_err, fmt = '.', label = 'LED off')
# if do_subtract:
    # plt.plot(x_values, a_sub, '.', label = 'LED only')

plt.xlabel('LED voltage (V)')
plt.ylabel('Average Amplitude')
plt.legend()
plt.show()

plt.figure()
plt.errorbar(x_values, ratio, yerr = np.abs(ratio_err), fmt = '.')
plt.xlabel('LED Voltage (V)')
plt.ylabel('Ratio')
plt.show()

