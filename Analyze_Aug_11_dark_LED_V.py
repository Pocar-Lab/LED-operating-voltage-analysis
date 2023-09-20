#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 21:02:59 2023

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

#%% Test 405nm

# test_high_bias = RunInfo(['/Users/albertwang/Desktop/nEXO/HDF Data/aug_12th_data/LED Operating Voltage/405nm/Run_1691760273.hdf5'], do_filter = True, upper_limit = .5, baseline_correct = True, prominence = 0.008, plot_waveforms= False, is_led = True)
# bias_test = test_high_bias.run_meta_data['/Users/albertwang/Desktop/nEXO/HDF Data/july_13th_LED_data/LED Operating V/405nm/Run_1689284451.hdf5']['RunNotes']
# test_high_bias.plot_hists('171', '.') #regular histogram
# test_high_bias.plot_peak_waveform_hist() #2D plot
# test_high_bias.plot_led_dark_hists('168', '.') #LED comparison plot
# print(len(test_high_bias.all_led_peak_data))

#%% 405nm Operating voltage analysis Aug 11
files = ['Run_1691754895.hdf5','Run_1691755538.hdf5','Run_1691758152.hdf5','Run_1691760273.hdf5','Run_1691759868.hdf5','Run_1691759448.hdf5','Run_1691757749.hdf5','Run_1691758964.hdf5','Run_1691758558.hdf5','Run_1691757319.hdf5','Run_1691756743.hdf5','Run_1691756187.hdf5'] #2.4,2.43,2.47,2.5,2.51,2.52,2.53,2.55,2.57,2.6,2.63     
proms = [0.008 for i in range(len(files))] #
upperlim = [0.5 for i in range(len(files))] 
runs = []
led_voltages = []
 
for file in range(len(files)):
        run_spe = RunInfo(['/Users/albertwang/Desktop/nEXO/HDF Data/aug_12th_data/LED Operating Voltage/405nm/' + files[file]], specifyAcquisition = False, do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file], is_led = True)
        led_voltage = run_spe.run_meta_data['/Users/albertwang/Desktop/nEXO/HDF Data/aug_12th_data/LED Operating Voltage/405nm/' + files[file]]['RunName']
        led_voltages.append(float(led_voltage[:-8]))
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
do_subtract = False
# if do_subtract:
    # a_sub = [get_subtract_hist_mean(run.all_led_peak_data, run.all_dark_peak_data) for run in runs]
ratio = np.array([(float(len(run.all_led_peak_data)) - float(len(run.all_dark_peak_data))) / float(len(run.all_dark_peak_data)) for run in runs])
ratio_err = ratio * np.array([np.sqrt(1.0 / float(len(run.all_led_peak_data)) + 1.0 / float(len(run.all_dark_peak_data))) for run in runs])
# ratio = [run.led_ratio for run in runs]
# print(led_voltages)
# print(a_LED)
# print(a_dark)
# print(ratio)


#%% 405nm
# led_voltages = [float(volt[:-1]) for volt in led_voltages]
a_dark_err = np.array(a_dark_err)
dark_avg = np.mean(a_dark)
# dark_err = np.std(a_dark, ddof = 1) / np.sqrt(len(dark_avg))
dark_err = 1.0 / np.sqrt(np.sum(1.0 / (a_dark_err * a_dark_err)))

#%% July 11 GN 405nm
files3 = ['Run_1689169204.hdf5','Run_1689162429.hdf5','Run_1689170954.hdf5','Run_1689162836.hdf5','Run_1689171968.hdf5','Run_1689164095.hdf5','Run_1689170491.hdf5','Run_1689169630.hdf5','Run_1689171493.hdf5','Run_1689164978.hdf5'] #2.40V, 2.45V, 2.475V, 2.50V,2.525V 2.55V, 2.575V, 2.60V, 2.625,2.65V
proms3 = [0.007 for file in files3] #0.0045,0.005, 0.005,0.005,0.005,0.005 (prev prominence values)
upperlim3 = [2 for file in files3] #0.25, 0.2, 0.2, 0.2, 0.5, 0.4 (prev upper limits)
runs3 = []
led_voltages3 = []
 
for file in range(len(files3)):

    run_spe = RunInfo(['/Users/albertwang/Desktop/nEXO/HDF Data/july_11th_dark_data/LEDV/405nm/' + files3[file]], specifyAcquisition = False, do_filter = False, upper_limit = upperlim3[file], baseline_correct = True, prominence = proms3[file], is_led = True)
    led_voltage = run_spe.run_meta_data['/Users/albertwang/Desktop/nEXO/HDF Data/july_11th_dark_data/LEDV/405nm/' + files3[file]]['RunNotes']
    led_voltages3.append(float(led_voltage[:-1]))
    runs3.append(run_spe)

#%%
biases3 = [run.bias for run in runs3]
a_LED3 = [np.mean(run.all_led_peak_data) for run in runs3]
a_LED_err3 = [np.std(run.all_led_peak_data, ddof = 1) / np.sqrt(len(run.all_led_peak_data)) for run in runs3]
a_dark3 = [np.mean(run.all_dark_peak_data) for run in runs3]
a_dark_err3 = [np.std(run.all_dark_peak_data, ddof = 1) / np.sqrt(len(run.all_dark_peak_data)) for run in runs3]
            
ratio3 = np.array([(float(len(run.all_led_peak_data)) - float(len(run.all_dark_peak_data))) / float(len(run.all_dark_peak_data)) for run in runs3])
ratio_err3 = ratio3 * np.array([np.sqrt(1.0 / float(len(run.all_led_peak_data)) + 1.0 / float(len(run.all_dark_peak_data))) for run in runs3])
# ratio = [run.led_ratio for run in runs]



#%%
# led_voltages = [float(volt[:-1]) for volt in led_voltages]
a_dark_err3 = np.array(a_dark_err3)
dark_avg3 = np.mean(a_dark3)
# dark_err = np.std(a_dark, ddof = 1) / np.sqrt(len(dark_avg))
dark_err3 = 1.0 / np.sqrt(np.sum(1.0 / (a_dark_err3 * a_dark_err3)))

#%%

x_values = led_voltages
x_values3 = led_voltages3
y_values = a_LED
y_values3 = a_LED3
y2_values = a_dark
y2_values3 = a_dark3

plt.figure()
# plt.fill_between(x_values, y1 = np.ones(len(x_values)) * dark_avg - dark_err, y2 = np.ones(len(x_values)) * dark_avg + dark_err, color = 'orange', alpha = 0.2)
# plt.plot(x_values, np.ones(len(x_values)) * dark_avg, 'r-')
plt.errorbar(x_values, y_values, yerr = a_LED_err, fmt = '.', label = 'LED on (Vacuum)')
plt.errorbar(x_values3, y_values3, yerr = a_LED_err3, fmt = '.', label = 'LED on (GN))')

plt.xlabel('LED voltage (V)')
plt.ylabel('Average Amplitude')
plt.legend()

plt.figure()
plt.fill_between(x_values, y1 = np.ones(len(x_values)) * dark_avg - dark_err, y2 = np.ones(len(x_values)) * dark_avg + dark_err, color = 'blue', alpha = 0.2)
plt.plot(x_values, np.ones(len(x_values)) * dark_avg, 'r-')
plt.errorbar(x_values, y2_values, yerr = a_dark_err, fmt = '.', label = 'LED off (Vacuum)')

plt.fill_between(x_values3, y1 = np.ones(len(x_values3)) * dark_avg3 - dark_err3, y2 = np.ones(len(x_values3)) * dark_avg3 + dark_err3, color = 'orange', alpha = 0.2)
plt.plot(x_values3, np.ones(len(x_values3)) * dark_avg3, 'r-')
plt.errorbar(x_values3, y2_values3, yerr = a_dark_err3, fmt = '.', label = 'LED off (GN)')
# if do_subtract:
    # plt.plot(x_values, a_sub, '.', label = 'LED only')

plt.xlabel('LED voltage (V)')
plt.ylabel('Average Amplitude')
plt.legend()
plt.show()

# plt.figure()
# plt.errorbar(x_values, ratio, yerr = np.abs(ratio_err), fmt = '.')
# plt.xlabel('LED Voltage (V)')
# plt.ylabel('Ratio')
# plt.show()


#%% Test 310nm

# test_high_bias = RunInfo(['/Users/albertwang/Desktop/nEXO/HDF Data/aug_12th_data/LED Operating Voltage/310nm/Run_1691776647.hdf5'], do_filter = True, upper_limit = .2, baseline_correct = True, prominence = 0.01, plot_waveforms= False, is_led = True)
# bias_test = test_high_bias.run_meta_data['/Users/albertwang/Desktop/nEXO/HDF Data/july_13th_LED_data/LED Operating V/405nm/Run_1689284451.hdf5']['RunNotes']
# test_high_bias.plot_hists('171', '.') #regular histogram
# test_high_bias.plot_peak_waveform_hist() #2D plot
# test_high_bias.plot_led_dark_hists('168', '.') #LED comparison plot
# print(len(test_high_bias.all_led_peak_data))
# get_subtract_hist_mean(test_high_bias.all_led_peak_data, test_high_bias.all_dark_peak_data, bias = bias_test ,plot = True)

#%% 310nm Operating voltage analysis
files = ['Run_1691773528.hdf5','Run_1691774021.hdf5','Run_1691776245.hdf5','Run_1691776647.hdf5','Run_1691777137.hdf5','Run_1691778002.hdf5','Run_1691778443.hdf5','Run_1691778927.hdf5','Run_1691779521.hdf5','Run_1691779996.hdf5','Run_1691780668.hdf5','Run_1691781092.hdf5','Run_1691781548.hdf5','Run_1691781991.hdf5'] #3.55
proms = [0.008 for i in range(len(files))] #
upperlim = [0.06 for i in range(len(files))] 
runs = []
led_voltages = []
 
for file in range(len(files)):
        run_spe = RunInfo(['/Users/albertwang/Desktop/nEXO/HDF Data/aug_12th_data/LED Operating Voltage/310nm/' + files[file]], specifyAcquisition = False, do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file], is_led = True)
        led_voltage = run_spe.run_meta_data['/Users/albertwang/Desktop/nEXO/HDF Data/aug_12th_data/LED Operating Voltage/310nm/' + files[file]]['RunName']
        led_voltages.append(float(led_voltage[:-8]))
        runs.append(run_spe)
        
        
#%%
for i in range(len(runs)):
    plt.figure()
    runs[i].plot_led_dark_hists('170', '.')

        

#%% 
biases = [run.bias for run in runs]
a_LED = [np.mean(run.all_led_peak_data) for run in runs]
# print('Amp LED on: ' + str(a_LED[0]))
# print(len(runs[0].all_led_peak_data))
a_LED_err = [np.std(run.all_led_peak_data, ddof = 1) / np.sqrt(len(run.all_led_peak_data)) for run in runs]
a_dark = [np.mean(run.all_dark_peak_data) for run in runs]
# print('Amp LED off: ' + str(a_dark[0]))
a_dark_err = [np.std(run.all_dark_peak_data, ddof = 1) / np.sqrt(len(run.all_dark_peak_data)) for run in runs]
do_subtract = False
# if do_subtract:
    # a_sub = [get_subtract_hist_mean(run.all_led_peak_data, run.all_dark_peak_data) for run in runs]
ratio = np.array([(float(len(run.all_led_peak_data)) - float(len(run.all_dark_peak_data))) / float(len(run.all_dark_peak_data)) for run in runs])
ratio_err = ratio * np.array([np.sqrt(1.0 / float(len(run.all_led_peak_data)) + 1.0 / float(len(run.all_dark_peak_data))) for run in runs])
# ratio = [run.led_ratio for run in runs]
# print(led_voltages)
# print(a_LED)
# print(a_dark)
# print(ratio)


#%% 310nm
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