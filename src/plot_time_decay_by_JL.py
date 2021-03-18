#!/usr/bin/env python
import sys
from post_process_by_JL import read_em3d_raw, read_observation_line,\
   plot_decay_curve, read_tem_file, write_time_decay_files,\
   plot_multi_channel
import numpy as np
import matplotlib.pyplot as plt

# set up the data directories and output directory

model = "50"
asp_ratio = "Tx_"+model+"m"
data_directory = "./model5/"+asp_ratio+"/results_50msec_100S_set1/"

# directory for 3D primary field
data_back_directory = "./model5/"+asp_ratio+"/results_50msec_100S_set1_primary/"

# output directory
PDF_directory = "./"

print("--Plotting data files in the directory:", data_directory)


# 68 chs; unit: msec;  50ms basetime; (t=0 is the start of off time) c21: 0.0255; c41: 12.66; c44:48.115 (this is
# the last off-time channel,i.e.,  <48.5 ms, beyond 48.5, they are on-time channels with opposite
# polarity of current)

discrete_chs = np.array([-1.9500,-1.8500,-1.7500,-1.6500,-1.5500,-1.4495,-1.3500,
                  -1.2500,-1.1500,-1.0500,-0.9500,-0.8500,-0.7500,-0.6500,
                  -0.5500,-0.4500,-0.3500,-0.2500,-0.1500,-0.0500,0.0255,
                  0.0550, 0.0700, 0.0950, 0.1300, 0.1750, 0.2350, 0.3150, 0.4200,
                  0.5600, 0.7450, 0.9900, 1.3150, 1.7450, 2.3150, 3.0750, 4.0850,
                  5.4250, 7.2050, 9.5700, 12.6600, 22.7300, 39.3950, 48.1150, 48.5255,
                  48.5550,48.5700,48.5950,48.6300,48.6750,48.7350,48.8150,48.9200,
                  49.0600,49.2450,49.4900,49.8150,50.2450,50.8150,51.5750,52.5850,
                  53.9250,55.7050,58.0700,61.1600,71.2300,87.8950,97.1150])


maxFileName = "./Example_Maxwell_results/100S - Loop=-"+model+".tem"

# 68 chs; unit: msec; c1: 0.0255; c29: 49.95 (last on-time channel before ramp off); 
# c30:50.05; c44: 51.45 (the last channel during the ramp); ramp duration: c30-c44.
# (t=0 is the start of on time); only 68 - 44 = 24 channels of off-time; c65-c68 are zeros(ignored!)
'''
discrete_chs = np.array([0.0255,0.0550,0.0700,0.0950,0.1300,0.1750,0.2350,
                         0.3150,0.4200,0.5600,0.7450,0.9900,1.3150,1.7450,
                         2.3150,3.0750,4.0850,5.4250,7.2050,9.5700,12.6600,
                         22.7300,39.3950,48.6150,49.5500,49.6500,49.7500,49.8500,
                         49.9500,50.0505,50.1500,50.2500,50.3500,50.4500,50.5500,
                         50.6500,50.7500,50.8500,50.9500,51.0500,51.1500,51.2500,
                         51.3500,51.4500,51.5255,51.5550,51.5700,51.5950,51.6300,
                         51.6750,51.7350,51.8150,51.9200,52.0600,52.2450,52.4900,
                         52.8150,53.2450,53.8150,54.5750,55.5850,56.9250,58.7050,
                         61.0700,64.1600,74.2300,90.8950,99.6150])
'''
#maxFileName = "MAXWELL_PATH_HERE"
# zero time shift
#discrete_chs = discrete_chs - 51.5  ## msec

# 66 chs; unit: msec; 16.66 ms half cycle; t=0 is the beginning of off time. 
# c1-c5: before ramp off; c6-c20: during ramp off.
# c21(0.0255 ms)-c43(15.06 ms): off times (0 - 15.16 ms); c44-c66: next opposite-polarity on time
'''
discrete_chs = np.array([-1.9500,-1.8500,-1.7500,-1.6500,-1.5500,-1.4495,-1.3500,
                         -1.2500,-1.1500,-1.0500,-0.9500,-0.8500,-0.7500,-0.6500,
                         -0.5500,-0.4500,-0.3500,-0.2500,-0.1500,-0.0500, 0.0255,
                         0.0550,0.0700,0.0950,0.1300,0.1750,0.2350,0.3150,
                         0.4200,0.5600,0.7450,0.9900,1.3150,1.7450,2.3150,
                         3.0750,4.0850,5.4250,7.2050,9.5700,12.6600,14.6800,
                         15.0600,15.1855,15.2150,15.2300,15.2550,15.2900,15.3350,
                         15.3950,15.4750,15.5800,15.7200,15.9050,16.1500,16.4750,
                         16.9050,17.4750,18.2350,19.2450,20.5850,22.3650,24.7300,
                         27.8200,30.3450,31.2300])

'''
#maxFileName = "MAXWELL_PATH_HERE"

# 80 chs; unit: ms; 150 ms basetime; t=0 is the beginning of OFF-time. linear ramp t=-1.5 to 0 ms.
# c21-c50: first off-time channels; c77=161.16 ms;
'''
discrete_chs = np.array([-1.9500,-1.8500,-1.7500,-1.6500,-1.5500,-1.4495,-1.3500,
                         -1.2500,-1.1500,-1.0500,-0.9500,-0.8500,-0.7500,-0.6500,
                         -0.5500,-0.4500,-0.3500,-0.2500,-0.1500,-0.0500, 0.0255,
                         0.0550,0.0700,0.0950,0.1300,0.1750,0.2350,0.3150,
                         0.4200,0.5600,0.7450,0.9900,1.3150,1.7450,2.3150,
                         3.0750,4.0850,5.4250,7.2050,9.5700,12.6600,22.7300,
                         39.3950,56.0650,72.7300,89.3950,106.0650,122.7300,139.3950,
                         148.1150,148.5255,148.5550,148.5700,148.5950,148.6300,148.6750,
                         148.7350,148.8150,148.9200,149.0600,149.2450,149.4900,149.8150,
                         150.2450,150.8150,151.5750,152.5850,153.9250,155.7050,158.0700,
                         161.1600,171.2300,187.8950,204.5650,221.2300,237.8950,254.5650,
                         271.2300,287.8950,297.1150])
'''
#maxFileName = "MAXWELL_PATH_HERE"

# 102 chs; unit:msec; 500 ms of base time; t=0 is the beginning of OFF-time. Linear ramp off t=-1.5 to 0 ms.
# c21-c61(t=498.115 ms): first off-time channels; c62-c102: next on-time chs.
'''
discrete_chs = np.array([-1.9500,-1.8500,-1.7500,-1.6500,-1.5500,-1.4495,-1.3500,
                         -1.2500,-1.1500,-1.0500,-0.9500,-0.8500,-0.7500,-0.6500,
                         -0.5500,-0.4500,-0.3500,-0.2500,-0.1500,-0.0500,0.0255,
                         0.0550,0.0700,0.0950,0.1300,0.1750,0.2350,0.3150,
                         0.4200,0.5600,0.7450,0.9900,1.3150,1.7450,2.3150,
                         3.0750,4.0850,5.4250,7.2050,9.5700,12.6600,22.7300,
                         39.3950,56.0650,72.7300,89.3950,106.0650,122.7300,139.3950,
                         156.0650,172.7300,189.3950,206.0650,222.7300,239.3950,272.7300,
                         322.7300,372.7300,422.7300,472.7300,498.1150,498.5255,498.5550,
                         498.5700,498.5950,498.6300,498.6750,498.7350,498.8150,498.9200,
                         499.0600,499.2450,499.4900,499.8150,500.2450,500.8150,501.5750,
                         502.5850,503.9250,505.7050,508.0700,511.1600,521.2300,537.8950,
                         554.5650,571.2300,587.8950,604.5650,621.2300,637.8950,654.5650,
                         671.2300,687.8950,704.5650,721.2300,737.8950,771.2300,821.2300,871.2300,921.2300,971.2300,997.1150])
'''
#maxFileName = "MAXWELL_PATH_HERE"
# convert Maxwell channels to seconds
discrete_chs = discrete_chs * 1e-3


decay_or_profile = 1   # 1: decaying plots;  2: profile plots

if decay_or_profile == 1:
   decaying_plot = True
   profile_plot = False
elif decay_or_profile == 2:
   decaying_plot = False
   profile_plot = True
   
if decaying_plot is True:
   channel_start = 1
   channel_end = 68
   site = 201
   outputFileName_basic = "Basetime_50ms_plate_100S_"+asp_ratio+"_dBdt_"  # for time decay plot
   extra_mark = "_semiLogScale_1Period"
   #extra_mark = "_semiLogScale_QuarterPeriod"   # extra marking string to add for output file name;
   write_decay_file = False  # whether to write time decay file onto disk
else:
   channel_start = 21
   channel_end = 44
   # extra_mark_multi_list = ["_OFF1", "_OFF2", "_OFF3", "_OFF4"]
   # channel_start_list = [1, 9, 15, 21] # OFF1 takes the chs of 1-8 of total off-time chs, OFF2 takes 9-14
   #                                     # , etc
   # channel_end_list = [8, 14, 20, 24]
   extra_mark_multi_list = ["_OFF1", "_OFF2", "_OFF3", "_OFF4", "_OFF5"]
   channel_start_list = [1, 9, 15, 19, 21] # OFF1 takes the chs of 1-8 of total off-time chs, OFF2 takes 9-14
                                       # , etc
   channel_end_list = [8, 14, 18, 20, 22]
   outputFileName_multi = "Basetime_50ms_plate_100S_"+asp_ratio+"_dBdt_profile_"  # for multi-channel plot

channels_Maxwell = discrete_chs[channel_start - 1 : channel_end]
#print(channels_Maxwell)


normal_or_not = False   # whether normalize results, true or false
plate_cond = "plate=100 S"   # conductivity string marker
x_is_log = False
y_is_log = True
unit_magnetic = "nT"
baseTime = 50.e-3  # msec
rampTime = 1.5e-3  # msec, linear ramp off

# evenly distributed time instants(channels) in seconds; for 3D code
if y_is_log is True:
   # data amount for plotting
   nt_data = 500
else:
   nt_data = 1000
if decaying_plot is True:
   #channels_3D = np.linspace(-18.14e-3, 48.48e-3, nt_data, dtype=np.float)   # 16.66 ms, 1 period
   #channels_3D = np.linspace(-51.14e-3, 98.5e-3, nt_data, dtype=np.float)   # 50 ms, 3/4 period
   #channels_3D = np.linspace(-6.14e-3, 98.5e-3, nt_data, dtype=np.float)   # 50 ms, 1/2 period
   #channels_3D = np.linspace(-6.14e-3, 51.5e-3, nt_data, dtype=np.float)   # 50 ms, 1/4 period
   channels_3D = np.linspace(-6.14e-3, 148.4e-3, nt_data, dtype=np.float)   # 50 ms, 1 periods
   #channels_3D = np.linspace(-6.14e-3, 248.4e-3, nt_data, dtype=np.float)   # 50 ms, 1.5 periods
   #channels_3D = np.linspace(-6.14e-3, 348.4e-3, nt_data, dtype=np.float)   # 50 ms, 2 periods
   #channels_3D = np.linspace(-6.14e-3, 448.4e-3, nt_data, dtype=np.float)   # 50 ms, 2.5 periods
   #channels_3D = np.linspace(-6.14e-3, 548.4e-3, nt_data, dtype=np.float)   # 50 ms, 3 periods
   #channels_3D = np.linspace(-151.14e-3, 298.5e-3, nt_data, dtype=np.float)   # 150 ms, 3/4 period
   #channels_3D = np.linspace(-3e-3, 298.5e-3, nt_data, dtype=np.float)   # 150 ms, about 1/2 period
   #channels_3D = np.linspace(-6e-3, 161.5e-3, nt_data, dtype=np.float)   # 150 ms, selected times
   #channels_3D = np.linspace(-6e-3, 998.5e-3, nt_data, dtype=np.float)   # 500 ms, about 1/2 period
   #channels_3D = np.linspace(-6e-3, 505.5e-3, nt_data, dtype=np.float)   # 500 ms, about 1/4 period
if profile_plot is True:
   # when plotting multi-channel
   channels_3D = channels_Maxwell


# First we need to read in the field values
field_filename = data_directory + 'iTr=001_dBdt.dat'
print("--3D data file: ", field_filename)
print("--MAXWELL data file: ", maxFileName)

# This is the filename of the time-stepping schemes
timeStep_filename = data_directory + 'time_stepping_scheme.txt'

# The number of observation sites
obs_filename = data_directory + 'iTr=001_observation_points_coordinates.xyz'
stn = read_observation_line(obs_filename, whichColumn=2)
n_rec = stn.size

# The number of actual time iterations
time_iters_filename = data_directory + 'time_iterations.dat'
time_iters = np.loadtxt(time_iters_filename, dtype=np.float)
n_iters = time_iters.size


# Read in the 3D modeled response. field stores EM fields in the format:f(n_channel, n_location, n_component)
# 3D data unit: T/s
field_3D = read_em3d_raw(field_filename, n_rec, n_iters, channels_3D, timeStep_filename,
                         ZeroTimeShift=None, interp=False)
# read the primary 3D response
field_pri_filename = data_back_directory + 'iTr=001_dBdt.dat'
timeStep_filename = data_back_directory + 'time_stepping_scheme.txt'
time_iters_filename = data_back_directory + 'time_iterations.dat'
time_iters = np.loadtxt(time_iters_filename, dtype=np.float)
n_iters = time_iters.size

# when asking primary fields over more than 1 period
nch_total = channels_3D.size
nch_redueced_1T = np.zeros(nch_total) # shifted time channels

period = baseTime * 4.0
for k in range(nch_total):
   nch_redueced_1T[k] = (channels_3D[k]  + baseTime + rampTime)   # since t=-51.5 ms is the beginning of everything
   ratio = nch_redueced_1T[k] / period
   nch_redueced_1T[k] = nch_redueced_1T[k] - np.floor(ratio) * period

nch_redueced_1T = nch_redueced_1T - (baseTime + rampTime)
field_pri_3D = read_em3d_raw(field_pri_filename, n_rec, n_iters, nch_redueced_1T, timeStep_filename,
                              ZeroTimeShift=None, interp=False)

field_3D = field_3D - field_pri_3D

if unit_magnetic == "nT":
   field_3D = field_3D * 1e+9
if decaying_plot is True:
   if write_decay_file is True:
      print("--write data files in the directory:", PDF_directory)
      write_time_decay_files(channels_3D, stn, np.transpose(field_3D[:,:,0]), np.transpose(field_3D[:,:,1]),
                          np.transpose(field_3D[:,:,2]),path=PDF_directory,
                          title="LONG_V1x1_450_50_1000_16msec_3D_All_channels")


# read in Max solution
# max_responses(n_stations, n_channels)
[Max_stn_ori, Max_resp_x, Max_resp_y, Max_resp_z] = read_tem_file(maxFileName, discrete_chs,
                  head_line=12, gap=12, stn_symbol='STATION',comp_symbol='COMPONENT',
                  n_ch=discrete_chs.size, borehole=False, write_decay=False)

#Max_stn = Max_stn_ori - 200.0  # for model-3/4
Max_stn = Max_stn_ori - 300.0  # for model-5

# select partial data to plot
nch = channels_Maxwell.size
field_max = np.zeros((nch, Max_stn.size, 3), dtype=np.float)
field_max[:,:, 0] = np.transpose(Max_resp_x[:, channel_start - 1:channel_end])
field_max[:,:, 1] = np.transpose(Max_resp_y[:, channel_start - 1:channel_end])
field_max[:,:, 2] = np.transpose(Max_resp_z[:, channel_start - 1:channel_end])
# current was multiplied 1e+6 A before !!
field_max *= 1.e-6

if decaying_plot is True:
   # write decaying data
   if write_decay_file is True:
      write_time_decay_files(channels_Maxwell, Max_stn, np.transpose(field_max[:,:,0]), np.transpose(field_max[:,:,1]),
                          np.transpose(field_max[:,:,2]),path=PDF_directory,
                          title="LONG_V1x1_450_50_1000_16msec_Maxwell_All_channels")

ylabel = "dB/dt (" + unit_magnetic + "/s)"
if unit_magnetic == "T":
   field_max = field_max * 1e-9

   

if normal_or_not is True:
   normalized = "_normalized"
else:
   normalized = ""

if decaying_plot is True:
   if site < 10:
      site_str = r'{:d}'.format(site)
   elif site >= 10 and site <= 99:
      site_str = r'{:2d}'.format(site)
   elif site >= 100 and site <= 999:
      site_str = r'{:3d}'.format(site)

   site = site -1
   # get a location first
   site_position = stn[site]
   minimum_diff = abs(Max_stn[0] - site_position)
   site_Max = 0
   for k in range(1, len(Max_stn)):
      if abs(Max_stn[k] - site_position) <= minimum_diff:
         minimum_diff = abs(Max_stn[k] - site_position)
         site_Max = k
 

   # plot all 3 components at once
   for k in range(3):
      comp = k
      if comp == 0:
         compstr = ",x"
         site_mark = "_site"+site_str + "_x"
      elif comp == 1:
         compstr = ",y"
         site_mark = "_site"+site_str + "_y"
      elif comp ==2:
         compstr = ",z"
         site_mark = "_site"+site_str + "_z"
      basic = 'site'+site_str + compstr
      legend_labels = ['3D,'+basic, 'MAX,'+basic]

      fig_filename = PDF_directory + outputFileName_basic+ "decaying" +normalized + site_mark + extra_mark
      print("--Plotting decaying component: ", comp)
      plot_decay_curve(channels_3D, field_3D[:,site, comp,None], legend_labels, fig_filename, ylabel,
                       time2=channels_Maxwell, data2 = field_max[:,site_Max, comp,None],
                       normal=normal_or_not, xlim=None, ylim=None,
                       loc='best', colors=None, linestyle=None,
                       plot_error=False, plot_waveform=True,x_log=x_is_log, y_log=y_is_log,
                       extra_text=plate_cond)


# The title of each column in the plot.
title = ['In-line (x)', 'Cross-line (y)', 'Vertical (z)']
unit_magnetic = "nT"
# The label on the y-axis.
yaxis_label = "dB/dt (" + unit_magnetic + "/s)"

if profile_plot is True:
   for k in range(len(extra_mark_multi_list)):
      extra_mark_multi = extra_mark_multi_list[k]
      s1 = channel_start_list[k] - 1
      s2 = channel_end_list[k]
      multi_filename = PDF_directory + outputFileName_multi +normalized+extra_mark_multi
      print("--writing file: ",  multi_filename)
      plot_multi_channel(field_3D[s1:s2,:,:], stn, channels_3D[s1:s2], title, multi_filename, yaxis_label,
                         field_2=field_max[s1:s2,:,:], stn_2=Max_stn, data1="3D", data2="MAX",
                         n_plot=1, ncomp=3, normal=normal_or_not, extra_text=plate_cond)
