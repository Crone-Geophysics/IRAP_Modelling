import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time as tm
import csv

plt.rc('font', family='serif')
plt.rc('text', usetex=True)


def spline_interpolate(x_orig, y_orig, x_intp, y=False, dy=False, integ=False,
                       k=3, s=0):

    """This function takes two 1D arrays as x and y and uses spline
    interpolation to
    1. do the interpolation at points given by xIntp
    2. calculate the first-order and second-order derivatives of the original
    data at points given by xIntp

    Inputs:
    k: Degree of the smoothing spline. Must be 1 <= k <= 5. Default is k = 3, a cubic spline.
    s: positive smooth factor. for different smoothing effect. If 0 (default), spline will
        interpolate all data points.
    """
    from scipy.interpolate import UnivariateSpline

    if len(x_orig) <= k:
        raise Exception("spline_interpolate: # of data points should be greater than k (order of smoothness) !")
    if len(x_orig) != len(y_orig):
        raise Exception("spline_interpolate: # of x data points should be equal to # of y data points !")
    # Get the spline object/function from x_orig and y_orig
    spl = UnivariateSpline(x_orig, y_orig, k=k, s=s)
    # get the interpolated values at given evaluation positions
    y_intp = spl(x_intp)

    if(y):
        result = y_intp
    elif(dy):
        # Get the derivative of spl
        spl_deriv = spl.derivative()
        dy_intp = spl_deriv(x_intp)
        result = dy_intp
    elif(integ):
        # Get the 1d integration of spl
        integral = np.zeros(x_intp.shape)
        for i in range(x_intp.size):
            integral[i] = spl.integral(max(x_orig), x_intp[i])
            result = integral

    return result


def get_selected_time_channels(time, tch):

    """
    This function tries to get the indices of the time array where the elements
    are the closest to tch
    """
    time_diff = np.zeros(len(time))
    selected_ch = []
    for ich in range(len(tch)):
        for iline in range(len(time)):
            time_diff[iline] = abs(time[iline] - tch[ich])
        selected_ch.append(np.argmin(time_diff))

    return selected_ch


def get_waveform_vs_time(ctype, time, basetime=1.0, rampLen=1.5, nhalfperiod=1,
                         tao=1.0, amp=1.0):

    """
    get the 1-D waveform (current) function with time
    time: array of time instants (in milli-seconds, or msec of unit); 
        with t=0 at the beginning of off time
    ctype: type of periodical waveforms
      1: triangle
      2: trapezoidal
      3: exponential rise and linear ramp off
    
    tao: \tao parameter in an exponential turn on, unit: msec
    amp: actual current amplitude in Ampere, optional
    """
    nt = len(time)
    all_current = np.zeros(nt)
    # number of time values in each half period
    #nt_first = int(nt / nhalfperiod)
    time_half_period = 2.0 * basetime
    time_one_period = 4.0 * basetime
    print("In plotting current waveform, nt ==", nt)
    # the first period of waveform of current
    for it in range(nt):
        # shift the time series here for current calculation as if t=0 is the beginning of everything
        t = time[it] + basetime + rampLen
        # to check which period the actual time instant is in
        ratio = t / time_one_period
        # since waveform repeats over periods, transform the actual t into corresponding time within
        #   the first period
        t = t - np.floor(ratio) * time_one_period
        if t <= time_half_period:
            # for the first half period
            all_current[it]=get_waveform_vs_time_half_period(ctype, t, basetime, rampLen,tao)
        elif t > time_half_period and t <= time_one_period:
            t = t - time_half_period
            all_current[it]=get_waveform_vs_time_half_period(ctype, t, basetime, rampLen,tao)
            all_current[it] *= -1.0
        else:
            raise Exception("Time value is more than 4*basetime (one period) in waveform plotting!")
    
    all_current *= amp
    return all_current


def get_waveform_vs_time_half_period(ctype, time, basetime, rampLen,
                         tao):

    """
    get the 1-D waveform (current) function with time
    time: a time instant (in milli-seconds, or msec of unit); 
        with t=0 at the beginning of off time
    ctype: type of periodical waveforms
      1: triangle
      2: trapezoidal
      3: exponential rise and linear ramp off
    
    tao: \tao parameter in an exponential turn on, unit: msec
    """

    t = time
    current = 0.0
    if ctype == 3:
        slope = -1.0 / rampLen
        if t >= 0 and t <= basetime:
            current = (1.0 - np.exp(-t/tao)) / (1.0 - np.exp(-basetime/tao))
        elif t > basetime and t <= basetime + rampLen:
            current = 1.0 + slope * (t - basetime)
        elif t > basetime + rampLen and t <= 2.0 * basetime:
            current = 0.0
        else:
            raise Exception("Time value is out of range[0, 2*basetime]!")

    else:
        raise Exception("Waveform type error !")
        
    return current


def get_t_modeling(filename):

    """
    Reads in the time_stepping_scheme file and returns the index at which the
    off-time response starts as well as the time instants for used for the
    modeling.
    
    Unit in "time_stepping_scheme" file is: seconds
    On return:
    t_modelling: time series with t=0 as the beginning of off time
    istart: the index of the time series where t>0 begins
    """

    data = np.loadtxt(filename, comments='#', dtype=np.float)

    # choose the column of time series with t=0 as the beginning of the off
    #  time, instead of the beginning of the modelling process.
    t_modeling = data[:, 2]
    if t_modeling[0] < 0.:
        for k in range(t_modeling.size):
            if t_modeling[k] > 0.0:
                istart = k
                break

        #t_modeling = t_modeling[istart:]
    else:
        istart = 0

    return t_modeling, istart


def read_observation_line(filename, whichColumn=2):
    """
    This func reads the locations of measurements from file, assuming all locations
    are on a line at the surface.
    
    The text file to be read is assumed to have multiple columns (index, x, y,z, etc)
    It return a rank-one array of locations.
    On inputs:
    filename: the file name string
    whichColumn: optional choice of which column to be regarded as the 1-D location array
    """
    
    # load text file containing the info
    data = np.loadtxt(filename, dtype=np.float)
    nx,ny = data.shape
    if whichColumn < 1:
       raise Exception("Selected column outside data indexes! too small")
    if whichColumn > ny:
       raise Exception("Selected column outside data indexes! too big")

    locs = data[:, whichColumn-1]
    return locs


def read_em3d_raw(filename, n_rec, n_step, ch, t_filename, interp=False,
                  only_offtime=False, ZeroTimeShift=None, n_comp=3):
    
    """
    This function reads the field values from the output file of the EM3D code
    and returns the field values at time channels used by the real data as
    specified in ch

    Variables:
    n_rec: Num Stations, the number of recordings for each time step, can be the number
           of observation points for configurations like fixed-loop survey
           which has one transmitter corresponding to multiple receivers,
           or can be the number of transmitters for configurations like
           the Slingram-style survey which has multiple transmitters with
           each transmitter corresponding to possibly multiple receivers.
    n_step: Num Channels, the number of iteration steps used in the time-stepping.
    ch: the time instants (in seconds) at which the responses should be extracted.
    t_filename: the filename of the time-stepping schemes used for the
                modeling. We need to extract time information of the iteration
    interp: should we obtain the field values at the specified time channels by
            using spline interpolation or by finding the time step that is
            closest to the actual time channel. Interpolation may be more
            expensive but possibly more accurate when the time steps used for
            time-stepping is very coarse.
    only_offtime: For the case of full waveform modes. When this is True
              extract only the off-time responses
    ZeroTimeShift: time shift of the zero time. Zero time is by default the beginning
            of the off time; time instants can be shifted such that the zero time is
            the beginning of the on time.
    n_comp: the number of component in the recorded data. Commonly is just 3

    The format of the output file is assumed to be the following:

    For configurations with a single loop and multiple observation points
    f_x(iobs_1, time_1), f_y(iobs_1, time_1), f_z(iobs_1, time_1)
    f_x(iobs_2, time_1), f_y(iobs_2, time_1), f_z(iobs_2, time_1)
    ...
    f_x(iobs_n, time1), f_y(iobs_n, time_1), f_z(iobs_n, time1)
    .
    .
    .
    f_x(iobs_1, time_n), f_y(iobs_1, time_n), f_z(iobs_1, time_n)
    f_x(iobs_2, time_n), f_y(iobs_2, time_n), f_z(iobs_2, time_n)
    ...
    f_x(iobs_n, time1), f_y(iobs_n, time_n), f_z(iobs_n, time_n)

    """

    # Initialize proper arrays for the field values
    field = np.zeros((n_step, n_rec, n_comp), dtype=np.float)

    # Read in the original responses
    data = np.loadtxt(filename, dtype=np.float)
    print("amount of dBdt data values: ", np.size(data))
    print("shape of dBdt data array:", np.shape(data))
    # The number of channels based on ch
    nch = ch.size

    # Extract the information from data to field(n_step, n_rec, n_comp)
    istart = 0
    for istep in range(n_step):
        istop = istart + n_rec
        field[istep, :, :] = data[istart:istop, :]
        istart = istop

    # Get the time instants for each step in the time-stepping process
    # idx_offtime is the index at which the response changes from on-time to
    #     off-time. For step-off modeling, idx_offtime is 0
    # time units in "time_modeling" are: seconds (implied from data text file)
    time_modeling, idx_offtime = get_t_modeling(t_filename)
    if ZeroTimeShift is not None:
        time_modeling = time_modeling + ZeroTimeShift
    # extract ONLY the off-time responses
    if only_offtime is True:
        field = field[idx_offtime:, :, :]
        time_modeling = time_modeling[idx_offtime:]

    # Get the responses at certain time gates
    field_out = np.zeros((nch, n_rec, n_comp), dtype=np.float)
    for i in range(len(ch)):
        if ch[i] < np.nanmin(time_modeling) or ch[i] > np.nanmax(time_modeling):
            print("time channel requested: ", ch[i] * 1.e+3, " ms")
            print("Time range of the data: ", np.nanmin(time_modeling) * 1.e+3,
                  np.nanmax(time_modeling) * 1.e+3, " ms")
            raise Exception("Selected time channel outside the time range in the data!")
   
    if interp:
        for irec in range(n_rec):
            for icomp in range(n_comp):
                tmp = spline_interpolate(time_modeling, field[:, irec, icomp], ch, y=True)
                field_out[:, irec, icomp] = tmp
    else:
        selected_time_index = get_selected_time_channels(time_modeling, ch)
        for i in range(len(selected_time_index)):
            k = selected_time_index[i]
            if( abs(time_modeling[k] - ch[i]) > 0.1 * abs(ch[i]) ):
                print("target time channel: ", ch[i])
                print("selected closest time channel: ", time_modeling[k])
                #raise Exception("Selected 3-D time instant has over 10% difference from target time!")
                print("------Warning------")
                print(("Selected 3-D time instant has over 10% difference from target time!"))
                # switch to interpolation method (1-D interpolation over time)
                for irec in range(n_rec):
                    for icomp in range(n_comp):
                        tmp = spline_interpolate(time_modeling[k-2:k+2], field[k-2:k+2, irec, icomp],
                                                 ch[i], y=True)
                        field_out[i, irec, icomp] = tmp
            else:
                # put the following in iteration to allow ch.size be larger than time_modelling.size
                field_out[i,:,:] = field[k, :, :]

    return field_out


def plot_multi_channel(field, stn, ch, title, fig_filename, yaxis_label,
                       field_2=None, stn_2=None, data1=None, data2=None,
                       labels=[], n_plot=3, ncomp=3,normal=True,extra_text=None):

    """
    Plots multi-channel plots.
    By default, the function plots four plots vertically in a n_plot by 3 panels format.
    When field_2 and stn_2 are present, then this function plots two sets of responses
    ('field_2' stores another solution and should have the same shape of 'field')
    field: array(n_channels, n_stations, n_components)
    ch: input or selected discrete time channels (in seconds), size is n_channels
    stn: rank-1 array of offsets (stations), size is n_stations
    n_plot: number of panels; used to limit the number of channels plotted in each panel (subplot)
    ncomp: number of components of fields, default 3
    data1, data2: additional strings of legends to distinguish plot curves
    """

    # Colors that will be used to denote different channels
    color = ['#e50000', '#0343df', '#15b01a', '#00ffff', '#9a0eea', '#929591',
             '#75bbfd', '#380282', '#014d4e', '#fdaa48', '#556b2f', '#8e2323',
             '#ff69b4']

    if len(title) < ncomp:
        raise Exception("title string does not have enough names for all components.")

    # Needed for generating different legends for multiple stuff
    plot_list = []
    scatter_list = []
    
    # some basic parameters
    xaxis_label = "Station (m)"
    sym_size = 5  # scatter symbol size
    landscape = False 
    
    # data type/legends
    if data1 is None:
        sol1 = "data1"
    else:
        sol1 = data1
        
    if data2 is None:
        sol2 = "data2"
    else:
        sol2 = data2
    # number of channels
    nch = ch.size
    nch_list = []
    remainder = np.mod(nch, n_plot)
    if remainder != 0:
        nch_basic = np.floor(nch / n_plot)
    else:
        nch_basic = int(nch / n_plot)

    # Get a list to hold the number of channels for each panel
    for iplot in range(n_plot):
        if remainder != 0:
            if iplot+1 <= remainder:
                # the remainder of the division will be within (0, n_plot)
                nch_list.append( nch_basic + 1)
            else:
                nch_list.append( nch_basic )
        else:
            nch_list.append( nch_basic )
            
        #if(iplot != n_plot-1):
        #    nch_list.append(nch_basic)
        #else:
        #    nch_list.append(nch - nch_basic*(n_plot-1))


    #print("--debug--: stn is", stn)
    # Create fig and ax objects
    # good fig size:
    # for 20 * 3 (nplot * ncomp) plots, figsize=(15, 30)
    # for 2 * 3 plots, figsize=(15, 8); or (14,6)
    # for 2 * 2 plots, figsize=(10, 5.5)
    # for 1 * 3 plots, figsize=(10, 5.5)
    # x,y,z components arranged in portrait or landscape mode
    if landscape is True:
        fig, axs = plt.subplots(n_plot, ncomp, figsize=(14, 6), sharex=True)
    else:
        # ncomp by 1 layout (portrait mode)
        fig, axs = plt.subplots(ncomp,n_plot,  figsize=(8, 10), sharex=True)

    ch_end = 0
    for iplot in range(n_plot):
        # number of channels for this subplot
        ncurves = int(nch_list[iplot])
        if iplot ==0:
            # this is the global index of the last time channel for this panel
            ch_end = ch_end + ncurves - 1
        else:
            ch_end = ch_end + ncurves
        # this is the global index of the first time channel for this panel
        ch_start = ch_end - ncurves + 1
        for icomp in range(ncomp):
            if n_plot == 1:
                ax = axs[icomp]
            else:
                if landscape is True:
                    ax = axs[iplot, icomp]
                else:
                    ax = axs[icomp, iplot]
            whichColumn = icomp
            if ncomp ==2:
                # skip y-comp if only plot 2 columns
                if icomp == 1:
                    whichColumn = 2
            if normal is True:
            # if scale the same-panel responses by a maximum value
                value_max = np.abs( np.nanmax(field[ch_start:ch_end+1,:, whichColumn]) )
                if value_max <= 1.e-20:
                    value_max = 1.e-20
                if field_2 is not None:
                    value_max_2 = np.abs( np.nanmax(field_2[ch_start:ch_end+1,:, whichColumn]) )
                    if value_max_2 <= 1.e-20:
                        value_max_2 = 1.e-20
                #print("current max value: ", value_max_2, iplot+1, icomp+1)
            else:
                value_max = 1.0
                if field_2 is not None:
                    value_max_2 = 1.0
            local_idx = 0
            for idx_ch in range(ch_start, ch_end+1):
                # this is for adding legends of time channels 
                # note the unit convertion for time values
                label = r'{:6.3f}'.format(ch[idx_ch] * 1e+3) + ' ms'
                #label = r'{:6.4f}'.format(ch[idx_ch]) + ' sec'
                
                plot_line, = ax.plot(stn, field[idx_ch, :, whichColumn]/value_max,
                                     color=color[local_idx], linewidth=1,label=label)
                plot_list.append(plot_line)
                if field_2 is not None:
                    # no legend here
                    scatter_plot = ax.scatter(stn_2, field_2[idx_ch, :, whichColumn]/value_max_2,
                                         sym_size, color=color[local_idx], marker='d')
                    scatter_list.append(scatter_plot)
                local_idx += 1
            # tm.sleep(15)
            # ax.add_artist(legend1)
            ax.set_ylabel(yaxis_label, fontsize=16)
            if landscape is True:
                if(iplot == 0):
                    ax.set_title(title[icomp]+" (lines:"+sol1+"; dots:"+sol2+")", fontsize=14, fontweight='bold')
                if(iplot == n_plot -1):
                    ax.set_xlabel(xaxis_label, fontsize=16, fontweight='bold')
            else:
                if(icomp == 0):
                    ax.set_title(title[icomp]+" (lines:"+sol1+"; dots:"+sol2+")", fontsize=14, fontweight='bold')
                if(icomp ==0):
                    ax.text(0.06, 0.8, "X", fontsize=24, fontweight='bold',transform = ax.transAxes)
                    if extra_text is not None:
                        ax.text(0.6, 0.25, extra_text, fontsize=24, fontweight='bold',transform = ax.transAxes)
                        ax.text(0.6, 0.1, "TX = -50 m", fontsize=24, fontweight='bold',transform = ax.transAxes)
                elif(icomp ==1):
                    ax.text(0.06, 0.8, "Y", fontsize=24, fontweight='bold',transform = ax.transAxes)
                elif(icomp ==2):
                    ax.text(0.06, 0.8, "Z", fontsize=24, fontweight='bold',transform = ax.transAxes)
                if(icomp == ncomp -1):
                    ax.set_xlabel(xaxis_label, fontsize=16, fontweight='bold')

            # appearance of label texts over major ticks
            #ax.set_xlim(-300.0, 300.0)
            ax.xaxis.set_major_locator(ticker.AutoLocator()) # locations of major ticks
            #ax.xaxis.set_major_formatter()
            ax.tick_params('both', length=5, width=1, which='major', right=True, top=True)
            ax.tick_params('both', length=2, width=1, which='minor', right=True, top=True)
            #ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            #ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            ax.ticklabel_format(style='plain', axis='both')
            ax.ticklabel_format(style='plain', axis='x')
            ax.tick_params(labelbottom=True,labelsize=12)
            ax.minorticks_on()
            if normal is True:
            # add some annotation
                ax.text(-190.0, 0.8, "Lines-peak ="+r'{:4.1e}'.format(value_max), fontsize=10)
                ax.text(-190.0, 0.6, "Dots-peak ="+r'{:4.1e}'.format(value_max_2), fontsize=10)
                #print("ratio of 3D/Maxwell is:", value_max / value_max_2, "----",iplot+1, icomp+1)
            # only show part of x limit
            #ax.set_xlim(-200.0, 600.0)

        box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width*1.4, box.height])
        ax.set_position([box.x0, box.y0, box.width*1.2, box.height])
        legend1 = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                            fontsize=12, frameon=False, handlelength=3)
        ax.add_artist(legend1)

    plt.tight_layout()
    fig.savefig(fig_filename + '.pdf', format = 'pdf', dpi=96)


def plot_decay_curve(time1, data1, labels, fig_name, ylabel, time2=None, data2=None, normal=False,
                     xlim=None, ylim=None, loc='lower left', colors=None, linestyle=None,
                     plot_error=False, plot_waveform=False, x_log=True, y_log=True, extra_text=None):

    """
    This function plots one or multiple decay curves.
    
    time1 is a 1D numpy array consisting of time values (in seconds) 
    while data1 is a 2D numpy array consisting of nx by ny entries of field values,
    with nx as the same size of "time1" (i.e., number of time instants), and ny as
    the number of solutions (could be number of locations, components, or number of 
    solutions from other methods such as analytical ones)
    """

    # Create fig and ax objects
    if plot_error:
        #fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    else:
        #fig, axs = plt.subplots()
        #fig, axs = plt.subplots(figsize=(8, 4))
        fig, axs = plt.subplots(figsize=(10, 4))

    # Get nx and ny
    nx1, ny1 = data1.shape
    if nx1 != time1.size:
        raise Exception("{The size of time is :d while the size of data is :d}"
                        .format(time1.size, nx1))

    # Initialize colors and linestyle
    if colors is None:
        colors = ['r', 'b', 'g', 'm',  'c', 'k', 'gray', 'gray']

    if linestyle is None:
        linestyle = ['-', '--', '-', '-', '-', '-', '-.', '-.']

    marker_list = ['o', 'v', 's', 'd', '*','.', ',', '+', '^', '<', '>']
    marker_size = 8
    ndata = 1
    ny = ny1
    if data2 is not None:
        ndata = 2
        nx2, ny2 = data2.shape
        ny = ny1 + ny2
    if ny > len(colors):
        print("number of color list:", len(colors))
        print("ny:", ny)
        raise Exception("ny is larger than the length of the given color list")

    if len(labels) != ny:
        print("number of labels:", len(labels))
        print("ny:", ny)
        raise Exception("The number of labels given is not the same as ny")

    if(plot_error):
        ax = axs[0]
    else:
        ax = axs

    if normal is True and data2 is not None:
        max_data1 = np.nanmax(np.abs(data1))
        max_data2 = np.nanmax(np.abs(data2))
        # normalization factor for data2, which will be multiplied by data2
        normal_factor = max_data1 / max_data2
        data1 = data1 / max_data1
        data2 = data2 / max_data2
        print("data1 max is ", max_data1)
        print("data2 max is ", max_data2)
    ic = -1
    for m in range(ndata):
        if m != 1:
            time = time1
            data = data1
        else:
            time = time2
            data = data2
        nx, ny = data.shape
        # convert time to ms
        time = time * 1.e+3
        for i in range(ny):
            ic += 1   # color list index
            if y_log:
                time_pos = []
                time_neg = []
                data_pos = []
                data_neg = []
                # log-scale data; deal with negative values
                for k in range(nx):
                    if data[k, i] > 0.0:
                        time_pos.append(time[k])
                        data_pos.append(data[k, i])
                    else:
                        time_neg.append(time[k])
                        data_neg.append(data[k, i] * (-1.0))
                time_pos = np.array(time_pos)
                time_neg = np.array(time_neg)
                data_pos = np.array(data_pos)
                data_neg = np.array(data_neg)
                label_on = False
                if time_pos.size > 0: 
                    label_on = True       
                    ax.scatter(time_pos, data_pos, marker_size, colors[ic], marker_list[ic], label=labels[ic])
                if time_neg.size > 0:
                    # no label defined here so that later legend won't pick up these negative-value plots if 
                    # label is already shown for postive-value plots;
                    # this is equivalent to using '  label="_nolegend_"   '
                    if label_on is not True:
                        my_label = labels[ic]
                    else:
                        my_label = "_nolegend_"
                    ax.scatter(time_neg, data_neg, marker_size*1.2, c=None, marker=marker_list[ic], label= my_label,
                        linewidth = marker_size * 0.05, edgecolors=colors[ic], facecolors='none')
            else:
                ax.plot(time, data[:, i], linewidth=1.5, label=labels[ic],
                    linestyle=linestyle[ic], color=colors[ic])
	
    # Set the limits
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.set_xlim(np.nanmin(time1*1.e+3), np.nanmax(time1*1.e+3))

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if normal is True and ndata > 1:
        le_title = "3D peak/Max peak: "+r'{:6.3e}'.format(normal_factor)
    else:
        le_title = None
    [xmin_ax,xmax_ax] = ax.get_xlim()
    #ax.vlines(x=50.0, ymin=ymin, ymax=ymax, colors='b')
    if extra_text is not None:
        # plot an extra text 
        ax.text(0.6, 0.15, extra_text, fontsize=20, fontweight='bold',transform = ax.transAxes)
        #ax.text(0.1, 0.05, r"$\sigma_b = 10^{-9}$ S/m", fontsize=20, fontweight='bold',transform = ax.transAxes)
    if x_log:
        ax.set_xscale('log')

    if y_log:
        ax.set_yscale('log')
    else:
        # plot a horizontal zero line
        #print("I've got here ?")
        timex = np.linspace(xmin_ax, xmax_ax, 1000, dtype=np.float)
        datax = np.zeros((timex.size, 1))
        ax.plot(timex, datax, linewidth=0.6, linestyle="--", color="gray")
    # Set up the figure
    #ax.yaxis.set_major_locator(ticker.AutoLocator()) # locations of major ticks
    #ax.yaxis.set_minor_locator(ticker.AutoLocator())
    #locmaj = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,1.0, ))
    ## The following two lines re-defines the location and number of major and
    ##  minor ticks of an axis, in case some minor ticks in log-scale plots
    ##  are missing (which is very bad as we cannot see the nice minor ticks that
    ##  indicate we are showing log scale data). When they are used, it's important
    ##  to comment out 'ax.minorticks_on()'.
    # params:
    # numdecs: number of decades to be shown for data
    # numticks: number of ticks
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numdecs=16, numticks=16))
    #ax.yaxis.set_major_locator(ticker.FixedLocator([1.e-8,1.e-6,1.e-4,1.e-2,1.e+0,1.e+2]))
    #ax.yaxis.set_major_formatter(ticker.FixedFormatter([1.e-8,1.e-6,1.e-4,1.e-2,1.e+0,1.e+2]))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, numdecs=16, subs = np.arange(1.0, 10.0) * 0.1, numticks=16))
    #ax.minorticks_on()
    ax.tick_params(axis='both', length=5, width=1, which='major')
    ax.tick_params(axis='both', length=2, width=1, which='minor')
    ax.set_xlabel('Time (ms)', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(axis="both",linewidth=0.5, linestyle=":")
    ax.legend(loc=loc, fontsize=14, frameon=True, title=le_title)
    
    if plot_waveform is True:
        # use twinx to create y2 axis
        axw = ax.twinx()
        # time units are all msec for waveform data
        wtime = time1*1e+3 # from s to ms
        wtime_1 = np.nanmin(wtime)
        wtime_2 = np.nanmax(wtime)
        num_halfperiod = 6
        if num_halfperiod <= 3:
            num_points = 1000
        else:
            num_points = 3000
        wtime = np.linspace(wtime_1, wtime_2, num_points, dtype=np.float)
        wave_data = get_waveform_vs_time(ctype=3, time=wtime, basetime=50,
                                         rampLen=1.5, nhalfperiod=num_halfperiod, tao=1.0, amp=30.0)
        axw.plot(wtime, wave_data, linewidth=0.5, label="waveform",
                    linestyle='-', color="green")
        axw.set_ylim(-40, 40)
        axw.minorticks_on()
        axw.set_ylabel("Current (A)", fontsize=16)
        
    if plot_error:
        ax = axs[1]
        # First get the relative errors
        if ny == 1:
            raise Exception("data array has only one column and no \
            relative error will be plotted")

        for i in range(1, ny):
            if i == ny-2:
               diff = np.abs(data1[:, i]) - np.abs(data1[:, ny-1])
               e = np.abs(diff / data1[:, ny-1]) * 100.
               err_label = labels[i] + " and " + labels[ny-1] 
               ax.plot(time1, e, linewidth=2, label=err_label, linestyle=linestyle[i],
                    color=colors[i])

        # Set up the figure
        ax.tick_params('both', length=5, width=1, which='major')
        ax.tick_params('both', length=3, width=1, which='minor')
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel('Relative error (\%)', fontsize=16)
        if x_log:
            ax.set_xscale('log')

        ax.tick_params(labelsize=14)
        ax.legend(loc="best", fontsize=14, frameon=False)

    plt.tight_layout()
    fig.savefig(fig_name + '.pdf', format = 'pdf', dpi=96)


def read_tem_file(fileIn, channels, head_line=3, gap=6, stn_symbol='STN',
                  comp_symbol='C', n_ch=0, borehole=False, write_decay=False):

    """
    This function reads in the data from a TEM file. Data unit depends on that of the data file.
    On inputs:
    stn_symbol: the string of "STATION", or "STN", corresponding to the station column
    comp_symbol: the string of "COMPONENT", corresponding to the indicating column of different x,y,z components
    gap: SEEMS to be the same as head_line (number of head lines)
    n_ch: number of channels
    channels: list of all time channels, in seconds, appeared in the .tem file.
    """

    import os
    import csv
    import shutil
    from subprocess import call

    # component strings identification
    if(borehole):
        str1 = 'A'
        str2 = 'U'
        str3 = 'V'
    else:
        str1 = 'X'
        str2 = 'Y'
        str3 = 'Z'

    # First we need to open the file and read in the line data
    f = open(fileIn, 'r')
    # read all records at once and get them to a long list
    lines = f.readlines()
    # how many extra lines below the title string containing "STATION, COMPONENT, CH1, ..."
    extra_lines_below = 1
    # Find out the line of string containing Stn(STN), Component (C), the first Channel (CH1)
    #   and also their indexes in the column position
    heading = lines[head_line - 1 - extra_lines_below].split()   # spliting, default delimiter is whitespace
    stn_idx = heading.index(stn_symbol)
    comp_idx = heading.index(comp_symbol)
    ch1_idx = heading.index("CH1")
    if(borehole):
        azimuth_idx = heading.index("AZIMUTH")
        dip_idx = heading.index("INCLINATION")
    # print('stn_idx, comp_idx, ch1_idx: ', stn_idx, comp_idx, ch1_idx)
    if('NCH' in heading):
        nch_idx = heading.index('NCH')
        # Figure out n_ch
        n_ch = int(lines[gap].split()[nch_idx])
    else:
        if(n_ch == 0):
            raise Exception('When there is not NCH information in the TEM \
            file, it should be provided when calling the function')

    # Define some of the variables
    nrecords = len(lines) - gap
    stations = np.zeros((nrecords))
    field_line = np.zeros((nrecords, n_ch))
    comp = []
    azimuth_line = np.zeros((nrecords))
    dip_line = np.zeros((nrecords))
    # Now read in the line data
    for iline in range(gap, len(lines)):
        line = lines[iline].split()
        thisRow = iline - gap
        stations[thisRow] = float(line[stn_idx])
        comp.append(line[comp_idx])
        field_line[thisRow, :] = line[ch1_idx:ch1_idx+n_ch]
        if(borehole):
            azimuth_line[thisRow] = line[azimuth_idx]
            dip_line[thisRow] = line[dip_idx]

    f.close()
    stn = np.unique(stations)
    n_stn = stn.size

    field_multi_x = np.zeros((n_stn, n_ch))
    field_multi_y = np.zeros((n_stn, n_ch))
    field_multi_z = np.zeros((n_stn, n_ch))
    if(borehole):
        azimuth = np.zeros(n_stn)
        dip = np.zeros(n_stn)

    for istn in range(n_stn):
        nXcomp = 0
        nYcomp = 0
        nZcomp = 0
        fieldx = np.zeros(n_ch)
        fieldy = np.zeros(n_ch)
        fieldz = np.zeros(n_ch)
        for iline in range(gap, len(lines)):
            thisRow = iline - gap
            if( np.abs(stations[thisRow] - stn[istn]) <= 0.1 ):
                if(borehole):
                    azimuth[istn] = azimuth_line[thisRow]
                    dip[istn] = dip_line[thisRow]
                if(comp[thisRow] == str1):
                    nXcomp = nXcomp + 1
                    fieldx = fieldx + field_line[thisRow, :]
                elif(comp[thisRow] == str2):
                    nYcomp = nYcomp + 1
                    fieldy = fieldy + field_line[thisRow, :]
                elif(comp[thisRow] == str3):
                    nZcomp = nZcomp + 1
                    fieldz = fieldz + field_line[thisRow, :]
        # print(nXcomp, nYcomp, nZcomp)
        if(nXcomp != 0):
            fieldx = fieldx / nXcomp
        else:
            raise Exception("nXcomp is zero: ", nXcomp)

        if(nYcomp != 0):
            fieldy = fieldy / nYcomp
        else:
            raise Exception("nYcomp is zero: ", nYcomp)

        if(nZcomp != 0):
            fieldz = fieldz / nZcomp
        else:
            raise Exception("nZcomp is zero: ", nZcomp)

        field_multi_x[istn, :] = fieldx
        field_multi_y[istn, :] = fieldy
        field_multi_z[istn, :] = fieldz

    if write_decay is True:
        write_time_decay_files(channels, field_multi_x, field_multi_y, field_multi_z)

    '''
    if(write_decay):
        path = 'tem_data'
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)

        call('mv ./tem_obs* ./tem_data', shell=True)
    '''

    if(not borehole):
        return stn, field_multi_x, field_multi_y, field_multi_z
    else:
        return stn, tCh, field_multi_x, field_multi_y, field_multi_z, \
            azimuth, dip


def write_time_decay_files(channels, stns, fieldx, fieldy, fieldz, path=None, title=None):
    # fieldx: f(n_stations, n_channels)
    # path: path to write files
    #
    #filex = 'tem_time_decay_' + '{:03d}'.format(istn + 1) + '_x.dat'
    if path is None:
        path = ""
    if title is None:
        title = ""
    else:
        title = title+"_"
    filex = path + title+'tem_time_decay_x.dat'
    filey = path + title+'tem_time_decay_y.dat'
    filez = path + title+'tem_time_decay_z.dat'
    fx = open(filex, 'w')
    fy = open(filey, 'w')
    fz = open(filez, 'w')
    writerx = csv.writer(fx)
    writery = csv.writer(fy)
    writerz = csv.writer(fz)
    channels = channels * 1e+3

    # writerx.writerow(["Time (s)  STN1, STN2, etc"])
    #writery.writerow(["Time (s)  Stations (m)"])
    #writerz.writerow(["Time (s)  Stations (m)"])
    [nstn, n_ch] = fieldx.shape
    #print("fieldz[:,0].size  ", np.array(fieldz[0:7,0]) )
    format0 = "%5s %14s" +"%16s"*nstn +"\n"
    stations_name = []
    for i in range(nstn):
        stations_name.append("STATION" + '{:d}'.format(i+1))
        
    fx.write("Data type: dB/dt; UNIT: nT/s\n")
    fx.write(("%s %d\n") % ("Number of stations: ", nstn))
    fx.write(("%s" +"%10.2f"*nstn +"\n") % ("Stations (metre): ", *stns))

    fy.write("Data type: dB/dt; UNIT: nT/s\n")
    fy.write(("%s %d\n") % ("Number of stations: ", nstn))
    fy.write(("%s" +"%10.2f"*nstn +"\n") % ("Stations (metre): ", *stns))

    fz.write("Data type: dB/dt; UNIT: nT/s\n")
    fz.write(("%s %d\n") % ("Number of stations: ", nstn))
    fz.write(("%s" +"%10.2f"*nstn +"\n") % ("Stations (metre): ", *stns))

    formats = "%5d %10.2f\n"
    for i in range(nstn):
        fx.write((formats) % (i+1, stns[i]))
        fy.write((formats) % (i+1, stns[i]))
        fz.write((formats) % (i+1, stns[i]))
        
    fx.write("dB/dt responses:\n")
    fy.write("dB/dt responses:\n") 
    fz.write("dB/dt responses:\n") 

    fx.write((format0) % ("NO.","Time (ms)",  *stations_name ))
    fy.write((format0) % ("NO.","Time (ms)",  *stations_name ))
    fz.write((format0) % ("NO.","Time (ms)",  *stations_name ))    
        
    format1 = "%5d %14.4f" +"%16.5E"*nstn+"\n"
    for i in range(n_ch):
        fx.write((format1) % (i+1, channels[i], *fieldx[:, i]))
        fy.write((format1) % (i+1, channels[i], *fieldy[:, i]))
        fz.write((format1) % (i+1, channels[i], *fieldz[:, i]))
    fx.close()
    fy.close()
    fz.close()

