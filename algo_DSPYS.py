from collections import Counter

import numpy as np
import pywt
from scipy.signal import hilbert, savgol_filter, periodogram,\
find_peaks, argrelextrema, detrend
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy import signal
from tqdm import tqdm
import subprocess
from influxdb import InfluxDBClient
import operator
import pytz
from datetime import datetime
import joblib
import torch
import random

BS_DURATION = 10
HR_DURATION = 10
RR_DURATION = 10
BP_DURATION = 10
ON_BED = 1 
OFF_BED=0
UNCERTAINTY=-1

def my_write_influx(influx, unit, table_name, data_name, data, start_timestamp, time_stamp_list):
    # print("epoch time:", timestamp)
    max_size = 100
    count = 0
    cnt = 0
    total = len(data)
    prefix_post  = "curl -s -POST \'"+ influx['ip']+":8086/write?db="+influx['db']+"\' -u "+ influx['user']+":"+ influx['passw']+" --data-binary \' "
    http_post = prefix_post
    max_len = len(data)
    for value in tqdm(data):
        count += 1
        cnt += 1
        http_post += "\n" + table_name +",location=" + unit + " "
        # st()
        http_post += data_name + "=" + str(int(round(value))) + " " + str(int(start_timestamp*10e8))
        if cnt < max_len:
            start_timestamp +=  (time_stamp_list[cnt] - time_stamp_list[cnt-1])
        # start_timestamp = time_stamp_list[cnt]
        # print((time_stamp_list[count] - time_stamp_list[count-1])/1000)
        if(count >= max_size):
            http_post += "\'  &"
            # print(http_post)
            # print("Write to influx: ", table_name, data_name, count)
            subprocess.call(http_post, shell=True)
            total = total - count
            count = 0
            http_post = prefix_post
    if count != 0:
        http_post += "\'  &"
        # print(http_post)
        # print("Write to influx: ", table_name, data_name, count, data)
        subprocess.call(http_post, shell=True)

def epoch_time_local(epoch, zone):
    local_tz = pytz.timezone(zone)
    time = datetime.fromtimestamp(epoch).astimezone(local_tz).strftime("%Y-%m-%dT%H:%M:%S.%f")
    return time

def read_influx(influx, unit, table_name, data_name, start_timestamp, end_timestamp):
    if influx['ip'] == '127.0.0.1' or influx['ip'] == 'localhost':
        client = InfluxDBClient(influx['ip'], '8086', influx['user'], influx['passw'], influx['db'],  ssl=influx['ssl'])
    else:
        client = InfluxDBClient(influx['ip'].split('//')[1], '8086', influx['user'], influx['passw'], influx['db'],  ssl=influx['ssl'])

    # client = InfluxDBClient(influx['ip'].split('//')[1], '8086', influx['user'], influx['passw'], influx['db'],  ssl=True)
    query = 'SELECT "' + data_name + '" FROM "' + table_name + '" WHERE "location" = \''+unit+'\' AND time >= '+ str(int(start_timestamp*10e8))+' AND time < '+str(int(end_timestamp*10e8))

    result = client.query(query)

    points = list(result.get_points())
    values =  list(map(operator.itemgetter(data_name), points))
    times  =  list(map(operator.itemgetter('time'),  points))

    data = values 
    return data, times

def local_time_epoch(time, zone):
    local_tz = pytz.timezone(zone)
    try:
        localTime = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f")
    except:
        localTime = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")

    local_dt = local_tz.localize(localTime, is_dst=None)
    epoch = local_dt.timestamp()
    return epoch

def get_occupancy_window(args):
    global BS_DURATION
    return BS_DURATION*100

def get_activity_window(args):
    global BS_DURATION
    return BS_DURATION*100

def get_hr_window(args):
    global HR_DURATION
    return HR_DURATION*100

def get_rr_window(args):
    global RR_DURATION
    return RR_DURATION*100

def get_bp_window(args):
    global BP_DURATION
    return BP_DURATION*100

def get_vital_window(args):
    global HR_DURATION, RR_DURATION, BP_DURATION
    return max(HR_DURATION, RR_DURATION, BP_DURATION)*100


QLIST_SIZE = 30
QLIST = [-1]*QLIST_SIZE
QIND = 0
OCCUPANCY = 0

# the old version based on signal quality -1, 0, 1
def calc_occupancy_v1(data):
    global QIND, QLIST, QLIST_SIZE, OCCUPANCY
    if QIND >= QLIST_SIZE: QIND = 0
    QLIST[QIND] = checkOnBedYS(data) #based on signal quality -1 (off bed), 0 (bad data), 1 (good data)
    QIND += 1
    if (OCCUPANCY ==1 and np.mean(QLIST) <=-0.5) or (OCCUPANCY ==0 and np.mean(QLIST) >=0.1):
        OCCUPANCY = 1- OCCUPANCY
    return OCCUPANCY

SLIST_SIZE = 30
SLIST = [0]*SLIST_SIZE
SIND = 0

def calc_occupancy_v2(data):
    global SIND, SLIST, SLIST_SIZE, OCCUPANCY
    if SIND >= SLIST_SIZE: SIND = 0
    SLIST[SIND] = bed_status_detection(data) #based on bed status 0, 1
    SIND += 1
    if (OCCUPANCY ==1 and np.mean(SLIST) < 0.5) or (OCCUPANCY ==0 and np.mean(SLIST) > 0.8):
        OCCUPANCY = 1- OCCUPANCY
    return OCCUPANCY

smooth_zcr_buffer = []
smooth_bs_buffer = []
buffer_len = 31
pre_status = UNCERTAINTY
pre_energy = 0
cnt  = 0
c_cnt = 0
bed_status_buffer = []
pre_zero_flag = 0
start_status = UNCERTAINTY
zcr_std_buffer = []
state_change_flag = 0
def calc_occupancy_v3(data):
    global smooth_zcr_buffer, smooth_bs_buffer, buffer_len, pre_energy, cnt, zcr_std_buffer, start_status
    cnt += 1
    bed_status, denoise_zcr, frequency = bed_occupancy_detection(data, 
                                        pre_energy)
    smooth_bs_buffer.append(bed_status)
    smooth_zcr_buffer.append(denoise_zcr)
    if len(smooth_bs_buffer) == buffer_len:
        zcr_std = np.std(smooth_zcr_buffer)
        zcr_std_buffer.append(zcr_std)
        if len(zcr_std_buffer) == buffer_len:
                del zcr_std_buffer[0]
        if zcr_std > 15:
            if frequency > 15 and (pre_status == OFF_BED or start_status == OFF_BED):
                smooth_bs_buffer[-1] = OFF_BED
            elif smooth_bs_buffer[-1] == ON_BED:
                smooth_bs_buffer[-1] = UNCERTAINTY
        bed_status = Counter(smooth_bs_buffer).most_common(1)[0][0]
        del smooth_bs_buffer[0]
        del smooth_zcr_buffer[0]
    return bed_status

start = 0
t_cnt = 0
continueous_correction_flag = 0
def status_correction(end_epoch, bed_status, data):
    corrected_status = []
    global pre_status, pre_zero_flag, start_status, c_cnt, start, state_change_flag, t_cnt, \
        continueous_correction_flag, c_correction_status
    if start == 0:
        start = end_epoch
    cur_t = end_epoch
    if (bed_status == UNCERTAINTY and pre_status != UNCERTAINTY) or (bed_status == UNCERTAINTY and cnt == 1):
        if not pre_zero_flag:
            start = cur_t
            start_status = pre_status
        pre_zero_flag = 1
    elif bed_status != UNCERTAINTY and pre_status == UNCERTAINTY and cnt > 1:
        c_cnt = 1
    elif c_cnt > 0 and bed_status != UNCERTAINTY:
        if pre_status != bed_status:
            c_cnt = 1
            t_cnt = 1
        else:
            c_cnt += 1
    ##########################################################
    if len(zcr_std_buffer) == buffer_len - 1:
        zcr_std = zcr_std_buffer[-1]
        print('zcr_std:', zcr_std)
        if max(data) - min(data) > 1000000:
            state_change_flag = 1
        print('state_change_flag1:', state_change_flag)

        if bed_status + pre_status == 1 and bed_status!= UNCERTAINTY:
            if state_change_flag == 0 and t_cnt >= 5:
                corrected_status = [pre_status]
                return corrected_status, end_epoch
            elif state_change_flag == 1:
                state_change_flag = 0
        
        if bed_status!= UNCERTAINTY:
            if bed_status != pre_status:
                t_cnt = 1
            elif bed_status == pre_status:
                t_cnt += 1
        else:
            if continueous_correction_flag:
                corrected_status = [c_correction_status] * t_cnt
                continueous_correction_flag = 0
                temp_start = end_epoch - t_cnt
                t_cnt = 0
                start_status = c_correction_status
                return corrected_status, temp_start
            t_cnt = 0
        print('continuous same status number:', t_cnt)
        print('state_change_flag2:', state_change_flag)
    ##########################################################
    if c_cnt >= 5:
        # if bed_status == ON_BED:
        #     quit()
        end = cur_t
        n = int(end - start) + 1 
        c_cnt = 0
        # if (start_status != UNCERTAINTY and start_status == bed_status) or (start_status != bed_status):
        if (start_status != UNCERTAINTY):
            print('===================marker:', start_status, bed_status)
            if state_change_flag == 1:
                corrected_status = [bed_status] * n
            else:
                continueous_correction_flag = 1
                c_correction_status = start_status
                corrected_status = [start_status] * n
            state_change_flag = 0
            # print('=================22222==========', Counter(smooth_bs_buffer[-5:]).most_common(1)[0][0])
            # if start_status == bed_status and Counter(smooth_bs_buffer[-5:]).most_common(1)[0][0] == (1 - bed_status):
            #     state_change_flag = 1
            # if start_status != bed_status:
            #     state_change_flag = 0
        pre_zero_flag = 0
    pre_status = bed_status
    return corrected_status, start

# occupancy: 1 - in bed; 0 - out bed
# quality: 1 - good; 0 - bad
def calc_bed_occupancy(data, params, args):	
    occupancy = 0	
    if args.version == "v1":	
        occupancy = calc_occupancy_v1(data)	
        params['occupancy'] = [occupancy]	
    elif args.version == "v2":	
        occupancy = calc_occupancy_v2(data)	
        params['occupancy'] = [occupancy]	
    elif args.version == "v3":	
        occupancy = calc_occupancy_v3(data)	
        current_epoch = params['timestamp']	
        corrected_status, start = status_correction(current_epoch, occupancy, data)	
        if len(corrected_status) > 0:	
            params = {'timestamp':start, 'occupancy':corrected_status}	
            print('=================================================')	
            print(corrected_status)	
            # if len(corrected_status) == 1 and corrected_status[0] == ON_BED:	
            #     occupancy = ON_BED	
            if len(corrected_status) == 1:	
                occupancy = corrected_status[0]	
        else:	
            params['occupancy'] = [occupancy]	
    return occupancy, params
# def calc_hr(data, params, args):
#     hr_window = get_hr_window(args)
#     if len(data) != hr_window: 
#         print("wrong input length for calc_hr")
#         return -1, []
#     quality, time_diff, envelopes = signal_quality_assessment(data, show=False) #args.debug)
#     if quality == False: return -1, []
#     median_hr = np.median(time_diff)
#     frequency = 1/(median_hr/100)
#     return round(frequency * 60), envelopes

# def calc_rr(data, params, args):
#     hr_window = get_hr_window(args)
#     rr_window = get_rr_window(args)
#     if len(data) != rr_window: 
#         print("wrong input length for calc_rr")
#         return -1
#     hr = 0
#     count = 0
#     while rr_window >= hr_window:
#         tmp_hr, _ = calc_hr(data[rr_window-hr_window:rr_window], params, args)
#         if tmp_hr == -1: return -1
#         hr += tmp_hr
#         rr_window -= hr_window
#         count += 1
#     hr = hr/count
#     rr = calc_rr_with_hr(data, hr, params, args)
#     return rr

def calc_bp(data, params, args):
    if args.algo_bp == "algo_VTCN":	
        from algo_DLYS import BP_model	
        sp, dp = BP_model().predict(data[-get_bp_window(args):])	
    elif args.algo_bp == "algo_LSTMAttention":	
        from algo_LSTMAttention import DL_Model	
        [sp, dp] = DL_Model().predict(data[-get_bp_window(args):])
    # sp, dp = -1
    return sp, dp


GOOD_CNT = 0
SEGMENT_HR = 0

# if a vital is -1, it means that vital is not calculated or not calculatable
def calc_vital_signs(data, params, args):
    hr = rr = sp = dp = -1	
    envelopes = []	
    if args.vitals.find('H') !=-1 and len(data) >= get_hr_window(args): 	
        hr, envelopes = calc_hr(data[-get_hr_window(args):], params, args)	
    if hr != -1:	
        if args.vitals.find('R') !=-1 and len(data) >= get_rr_window(args): 	
            rr = calc_rr(data[-get_rr_window(args):], params, args)	
        if args.vitals.find('S') !=-1 and len(data) >= get_bp_window(args): 	
            sp, dp = calc_bp(data[-get_bp_window(args):], params, args)
    
    return hr, rr, sp, dp, envelopes

# 1 - moving; 0 - not moving
from algo_dsp import checkMovement
def calc_sleep_activities(data, params, args):
    if True == checkMovement(data):
        movement = 1
    else:
        movement = 0
    return movement, params

def kurtosis(data):
    x = data - np.mean(data)
    a = 0
    b = 0
    for i in range(len(x)):
        a += x[i] ** 4
        b += x[i] ** 2
    a = a/len(x)
    b = b/len(x)
    k = a/(b**2)
    return k

def wavelet_decomposition(data, wave, Fs = None, n_decomposition = None):
    a = data
    w = wave
    ca = []
    cd = []
    rec_a = []
    rec_d = []
    freq_range = []
    for i in range(n_decomposition):
        if i == 0:
            freq_range.append(Fs/2)
        freq_range.append(Fs/2/(2** (i+1)))
        (a, d) = pywt.dwt(a, w)
        ca.append(a)
        cd.append(d)

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec = pywt.waverec(coeff_list, w)
        rec_a.append(rec)
        # ax3[i].plot(Fre, FFT_y1)
        # print(max_freq)

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))
         
    return rec_a, rec_d

def zero_cross_rate(x):
    cnt  = 0
    for i in range(1,len(x)):
        if x[i] > 0 and x[i-1] < 0:
            cnt += 1
        elif x[i] < 0 and x[i-1] > 0:
            cnt += 1
    return cnt

def bed_status_detection(data):
    # x = (x - np.mean(x))/np.std(x)
    x = standize(data)    
    zcr = zero_cross_rate(x)
    
    rec_a, rec_d = wavelet_decomposition(data = x, wave = 'db4', Fs = 100, n_decomposition = 6)
    min_len = min(len(rec_d[-4]), len(rec_d[-3])) #len(rec_d[-5])
    denoised_sig = rec_d[-4][:min_len] + rec_d[-3][:min_len]
    
    denoise_zcr = zero_cross_rate(denoised_sig)

    if denoise_zcr > 165:
        if zcr < 180:
            bed_status = 1
        else:
            bed_status = 0
    elif denoise_zcr < 150:
        bed_status = 1
    elif zcr > 340:
        bed_status = 0
    else:
        bed_status = 1
    return bed_status #, denoise_zcr, zcr

def standize(data):
    return (data - np.mean(data))/np.std(data)


def get_envelope(data, n_decomposition, Fs):
    # x = (x - np.mean(x))/np.std(x)
    x = standize(data)
    denoised_sig = wavelet_denoise(signal = x, Fs = Fs, n_decomposition = n_decomposition)
    
    z= hilbert(denoised_sig) #form the analytical signal
    envelope = np.abs(z)
    
    smoothed_envelope = savgol_filter(envelope, 41, 2, mode='nearest')
    smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope))/np.std(smoothed_envelope)
    
    return smoothed_envelope

def wavelet_denoise(signal, Fs, n_decomposition):
    # x = (x - np.mean(x))/np.std(x)
    x = standize(signal)
    rec_a, rec_d = wavelet_decomposition(data = x, wave = 'db4', Fs = Fs, n_decomposition = n_decomposition)
    
    min_len = min(len(rec_d[-1]), len(rec_d[-2]), len(rec_d[-3]), len(rec_d[-4])) #len(rec_a[-1]) len(rec_d[-5])
    denoised_sig = rec_d[-1][:min_len] + rec_d[-2][:min_len] + rec_d[-4][:min_len] + rec_d[-3][:min_len] #+ rec_a[-1][:min_len]
    return denoised_sig

def J_peaks_detection(x, f):
    window_len = round(1/f * 100)
    start = 0
    J_peaks_index = []
    
    #get J peaks
    while start < len(x):
        end = start + window_len
        if start > 0:
            segmentation = x[start -1 : end]
        else:
            segmentation = x[start : end]
        # J_peak_index = np.argmax(segmentation) + start
        max_ids = argrelextrema(segmentation, np.greater)[0]
        for index in max_ids:
            if index == max_ids[0]:
                max_val = segmentation[index]
                J_peak_index = index + start
            elif max_val < segmentation[index]:
                max_val = segmentation[index]
                J_peak_index = index + start
        
        if len(max_ids) > 0 and x[J_peak_index] > 0:
            if len(J_peaks_index) == 0 or J_peak_index != J_peaks_index[-1]:
                J_peaks_index.append(J_peak_index)
        start = start + window_len//2
    
    return J_peaks_index

def rr_estimation(x, n_lag):
    rr_acf = acf(x, nlags = n_lag)
    rr_acf = rr_acf - np.mean(rr_acf)
    rr_peak_ids, _ = signal.find_peaks(rr_acf, height = 0.2)
    time_diff = rr_peak_ids[1:] - rr_peak_ids[:-1]
    if len(rr_peak_ids) < 2:
        return -1
    
    # if len(rr_peak_ids) > 3 and np.std(time_diff) > 32:
    #     # for i in range(1, len(time_diff)):
    #     #     if abs(time_diff[i] - time_diff[i-1]) > 32:
    #             return -1
    if np.std(time_diff) > 32:
        return -1
    
    # check if the peaks is periodic
    median_hr = np.median(time_diff)
    frequency = 1/(median_hr/100)
    
    if frequency > 0.5 or frequency < 0.1:
        return -1
    return frequency

def band_pass_filter(data, Fs, low, high, order):
    b, a = signal.butter(order, [low/(Fs * 0.5), high/(Fs * 0.5)], 'bandpass')
    # perform band pass filter
    # filtered_data = signal.filtfilt(b, a, data)
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data

def ACF(x, lag):
    acf = []
    mean_x = np.mean(x)
    var = sum((x - mean_x) ** 2)
    for i in range(lag):
        if i == 0:
            lag_x = x
            original_x = x
        else:
            lag_x = x[i:]
            original_x = x[:-i]
        new_x = sum((lag_x - mean_x) * (original_x - mean_x))/var
        new_x = new_x/len(lag_x)
        acf.append(new_x)
    return np.array(acf)

def signal_quality_assessment(data, show = False):
# def signal_quality_assessment(data, denoised_sig, show = False):
    # x = (x - np.mean(x))/np.std(x)
    x = standize(data)
    if round(kurtosis(x)) > 5:
        return [False, [], []] # body movement
    
    denoised_sig = wavelet_denoise(signal = x, Fs = 100, n_decomposition = 6)
    # print(len(x), len(denoised_sig))
    # signal_stds = []
    index = 0
    window_size = 100
    noise_cnt = 0
    while (index + window_size < len(denoised_sig)):
        # signal_stds.append(np.std(denoised_sig[index:index + window_size]))
        if np.std(denoised_sig[index:index + window_size]) > 1.3:
            noise_cnt += 1
            if noise_cnt > 1 or np.std(denoised_sig[index:index + window_size]) > 1.5:
                if show:
                    fig, ax = plt.subplots(2, 1, figsize=(15,8))
                    ax[0].plot(x)
                    ax[1].plot(denoised_sig)
                return [False, [], []]
        index = index + window_size

    z= hilbert(denoised_sig) #form the analytical signal
    envelope = np.abs(z)
    # print(len(denoised_sig), len(envelope))
    
    smoothed_envelope = savgol_filter(envelope, 41, 2, mode='nearest')
    smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope))/np.std(smoothed_envelope)
    
    trend = savgol_filter(smoothed_envelope, 201, 2, mode='nearest')
    detrended_envelope = smoothed_envelope - trend
    detrended_envelope = (detrended_envelope - np.mean(detrended_envelope))/np.std(detrended_envelope)
    
    n_lag = len(detrended_envelope)//2
    acf_x = acf(detrended_envelope, nlags = n_lag)
    acf_x = acf_x - np.mean(acf_x)
    f, Pxx_den = periodogram(acf_x, fs = 100, nfft = 1024)
    # print(f)
    
    sig_means = []
    index = 0
    frequency = f[np.argmax(Pxx_den)]
    power = max(Pxx_den)
    # print('ACF frequency:', frequency)

    acf_envelope =ACF(detrended_envelope, len(detrended_envelope))
    acf_envelope = (acf_envelope-min(acf_envelope))/(max(acf_envelope)-min(acf_envelope))
    # print(len(acf_y))    
    if show:
        # acf_y = acf_x
        print(len(x), len(detrended_envelope), len(acf_envelope))
        fig, ax = plt.subplots(6, 1, figsize=(15,8))
        ax[0].plot(x, label = 'raw data')
        ax[1].plot(denoised_sig, label = 'wavelet denoised data')
        ax[2].plot(envelope, label = 'envelope extraction by Hilbert transform')
        ax[3].plot(detrended_envelope, label = 'smoothed envelope')
        # ax[4].plot(acf_x, label = 'ACF of smoothed envelope')
        ax[4].plot(acf_envelope, label = 'ACF of smoothed envelope')
        ax[5].plot(f, Pxx_den, label = 'spectrum of ACF')
        for i in range(len(ax)):
            ax[i].legend()
    window_size = 100
    while (index + window_size < len(acf_x)):
        sig_means.append(np.mean(acf_x[index:index + window_size]))
        index = index + window_size
    if np.std(sig_means) < 0.1 and 0.7< frequency < 2 and power > 0.1:
        good_flag = 1
        peak_ids, _ = find_peaks(acf_x, height = 0.1)
        time_diff = peak_ids[1:] - peak_ids[:-1]
        if len(peak_ids) < 4 or n_lag - peak_ids[-1] > 140 or abs(time_diff[0] - peak_ids[0]) > 8:
            res = ['bad data', np.std(sig_means), frequency, power]
            return [False, [], []]
        
        # # check if the peaks is periodic
        # median_hr = np.median(time_diff)
        # frequency = 1/(median_hr/100)
        if len(peak_ids) > 3:
            distortion_cnt = 0
            for i in range(1, len(time_diff)):
                if abs(time_diff[i] - time_diff[i-1]) > 8:
                    distortion_cnt += 1
                    if distortion_cnt > 0:
                        good_flag = 0
                        break
            
            if good_flag and np.mean(acf_x[peak_ids]) >0.17 and np.std(acf_x[peak_ids]) < 0.15:
                res = ['good data', np.std(sig_means), frequency, power]
                if show:
                    ax[4].scatter(peak_ids, acf_x[peak_ids])
            else:
                if show:
                    ax[4].scatter(peak_ids, acf_x[peak_ids], c = 'r')
                res = ['bad data', np.std(sig_means), frequency, power]
        else:
            res = ['bad data', np.std(sig_means), frequency, power]
        # res = [f[np.argmax(Pxx_den)], max(Pxx_den), 'rx']
    else:
        res = ['bad data', np.std(sig_means), frequency, power]
        if show:
            fig.suptitle('bad data')
        # res = [f[np.argmax(Pxx_den)], max(Pxx_den), 'go']
    # print(res)
    
    if res[0] == 'good data':
        return [True, time_diff, [smoothed_envelope, acf_envelope]]
    else:
        return [False, [], []]

def calc_rr_with_hr(data, hr, params, args):
    rr_window = get_rr_window(args)
    if len(data) != rr_window: 
        print("wrong input length for calc_rr")
        return -1
    f = hr/60
    window_len = round(1/f * 100)//2
    smoothed_envelope = get_envelope(data = data, n_decomposition = 6, Fs = 100)
    J_peaks_index = J_peaks_detection(smoothed_envelope, f)
    y = smoothed_envelope[J_peaks_index]
    upper_envelope = interp1d(np.array(J_peaks_index), y, kind = 'cubic', 
                              bounds_error = False)(list(range(J_peaks_index[0], J_peaks_index[-1] + 1)))
    # rr_envelope = (upper_envelope - np.mean(upper_envelope))/np.std(upper_envelope)
    rr_envelope = standize(upper_envelope)
    rr_envelope = band_pass_filter(data = rr_envelope, Fs = 100,
                                   low = 0.2, high = 0.5, order = 3)
    rr_envelope = savgol_filter(rr_envelope, window_len * 4 + 1, 2, mode='nearest')
    rr = rr_estimation(rr_envelope, len(rr_envelope)//2)
    
    if rr != -1:
        rr = rr * 60
    else:
        rr = -1
    return rr

# the old quality control algorithm by YS
def checkOnBedYS(signal):
    res = quality_control_algorithm(x = signal, n_decomposition = 6, 
                                        Fs = 100, n_lag = len(signal)//2, show = False)
    return res[1]

def cal_acf(x, nlags):
    acf_x = acf(x, nlags=nlags)
    return acf_x

def quality_control_algorithm(x, n_decomposition, Fs, n_lag, show = False):
    x = (x - np.mean(x))/np.std(x)
    f, Pxx_den = periodogram(x, fs = 100)
    frequency = f[np.argmax(Pxx_den)]
    # fig, ax = plt.subplots(2, 1, figsize=(16,4))
    # ax[0].plot(x)
    # ax[1].plot(f, Pxx_den)
    # print(frequency)
    # if frequency > 9:
    #     return ['off bed']
    # print(round(kurtosis(x)))
    spikes = 0
    if round(kurtosis(x)) > 4:
        spikes = 1
        if show:
            fig, ax = plt.subplots(6, 1, figsize=(15,8))
            ax[0].plot(x)
        # return ['bad data']
    rec_a, rec_d = wavelet_decomposition(data = x, wave = 'db4', Fs = Fs, n_decomposition = n_decomposition)
    
    min_len = min(len(rec_d[-4]), len(rec_d[-3])) #len(rec_d[-5])
    denoised_sig = rec_d[-4][:min_len] + rec_d[-3][:min_len]
    f, Pxx_den = periodogram(denoised_sig, fs = 100)
    frequency = f[np.argmax(Pxx_den)]
    # print(frequency)
    # fig, ax = plt.subplots(2, 1, figsize=(16,4))
    # ax[0].plot(x)
    # ax[1].plot(f, Pxx_den)
    if frequency > 9:
        # fig.suptitle('off bed')
        return ['off bed', -1]

    if spikes:
        # fig.suptitle('bad data')
        return ['bad data', 0]
    
    signal_stds = []
    index = 0
    window_size = 100
    noise_cnt = 0
    while (index + window_size < len(denoised_sig)):
        signal_stds.append(np.std(denoised_sig[index:index + window_size]))
        if np.std(denoised_sig[index:index + window_size]) > 1.3:
            noise_cnt += 1
            if noise_cnt > 1 or np.std(denoised_sig[index:index + window_size]) > 1.5:
                if show:
                    fig, ax = plt.subplots(2, 1, figsize=(15,8))
                    ax[0].plot(x)
                    ax[1].plot(denoised_sig)
                return ['bad data', 0]
        index = index + window_size
    # print(signal_stds)
    # print(np.std(signal_stds))
    # denoised_sig = rec_d[-5]
    z= hilbert(denoised_sig) #form the analytical signal
    envelope = np.abs(z)
    
    smoothed_envelope = savgol_filter(envelope, 41, 2, mode='nearest')
    smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope))/np.std(smoothed_envelope)
    
    trend = savgol_filter(envelope, 201, 2, mode='nearest')
    smoothed_envelope = smoothed_envelope - trend
    smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope))/np.std(smoothed_envelope)
    # print(round(kurtosis(smoothed_envelope)))
    
    acf_x = cal_acf(smoothed_envelope, n_lag)
    acf_x = acf_x - np.mean(acf_x)
    f, Pxx_den = periodogram(acf_x, fs = 100)
    
    sig_means = []
    index = 0
    frequency = f[np.argmax(Pxx_den)]
    power = max(Pxx_den)
    # magnitude = np.sqrt(power)
    # print(magnitude)
    # f, Pxx_den = periodogram(envelope, fs = 100)
    # N = 1000
    # y = fft(envelope)
    # y = y[0:N//2]
    # xf = fftfreq(N, 1/Fs)[:N//2]
    # plt.figure(figsize = (16,3))
    # plt.plot(xf, np.abs(y))
    
    if show:
        fig, ax = plt.subplots(6, 1, figsize=(15,8))
        ax[0].plot(x)
        ax[1].plot(denoised_sig)
        ax[2].plot(envelope)
        # ax[3].plot(smoothed_envelope)
        ax[3].plot(smoothed_envelope - trend)
        ax[4].plot(acf_x)
        ax[5].plot(f, Pxx_den)
    
    
    # print(round(kurtosis(Pxx_den[:100])))
    window_size = 100
    while (index + window_size < len(acf_x)):
        sig_means.append(np.mean(acf_x[index:index + window_size]))
        index = index + window_size
    if np.std(sig_means) < 0.1 and 0.7< frequency < 3 and power > 0.1:
        good_flag = 1
        peak_ids, _ = signal.find_peaks(acf_x, height = 0.1)
        if 500 - peak_ids[-1] > 140:
            res = ['bad data', 0, np.std(sig_means), frequency, power]
            return res
        time_diff = peak_ids[1:] - peak_ids[:-1]
        # check if the peaks is periodic
        # print(abs(time_diff[1:] - time_diff[:-1]))
        if len(peak_ids) > 3:
            distortion_cnt = 0
            for i in range(1, len(time_diff)):
                if abs(time_diff[i] - time_diff[i-1]) > 8:
                    distortion_cnt += 1
                    # print(distortion_cnt, time_diff[i] - time_diff[i-1])
                    if distortion_cnt > 0:
                        good_flag = 0
                        break
            # check if the mean of correlation peak is over 0.2
            if good_flag and np.mean(acf_x[peak_ids]) >0.17:
                res = ['good data', 1, np.std(sig_means), frequency, power]
                if show:
                #     fig, ax = plt.subplots(6, 1, figsize=(16,18))
                #     ax[0].plot(x)
                #     ax[1].plot(denoised_sig)
                #     ax[2].plot(envelope)
                #     ax[3].plot(smoothed_envelope)
                    ax[4].scatter(peak_ids, acf_x[peak_ids])
                    # ax[5].plot(f, Pxx_den)
                    fig.suptitle('good data')
            else:
                if show:
                    ax[4].scatter(peak_ids, acf_x[peak_ids], c = 'r')
                res = ['bad data', 0, np.std(sig_means), frequency, power]
        else:
            res = ['bad data', 0, np.std(sig_means), frequency, power]
        # res = [f[np.argmax(Pxx_den)], max(Pxx_den), 'rx']
    else:
        res = ['bad data', 0, np.std(sig_means), frequency, power]
        if show:
            fig.suptitle('bad data')
        # res = [f[np.argmax(Pxx_den)], max(Pxx_den), 'go']
    return res

def hr_estimation(data):
    x = data
    fs = len(x)/BS_DURATION
    res = signal_quality_assessment_v3(x=x, Fs=fs, n_lag=len(x) // 2, 
                                       low = 0.8, high = 10, 
                                       denoised_method = 'bandpass',
                                       show=False)
    if res[0]:
        acf_x = res[-1]
        peak_ids, _ = signal.find_peaks(acf_x, height = np.mean(acf_x))
        # peak_ids, _ = signal.find_peaks(acf_x, height = 0.25)
        time_diff = peak_ids[1:] - peak_ids[:-1]
        
        # check if the peaks is periodic
        cadidates = []
        for peak_id in peak_ids:
            if (peak_id > int(0.51 * fs) and peak_id < int(1.26 * fs)):
                cadidates.append(peak_id)
        
        if len(cadidates) == 0:
            median_hr = np.median(time_diff)
        else:
            median_hr = cadidates[np.argmax(acf_x[cadidates])] #+ int(0.5 * Fs)
        # median_hr = np.median(time_diff)
        frequency = 1/(median_hr/fs)

        return round(frequency * 60), acf_x
    else:
        return -1, []

def calc_hr(data):
    x = data
    fs = len(x)/BS_DURATION
    res = signal_quality_assessment_v3(x=x, Fs=fs, n_lag=len(x) // 2, 
                                       low = 0.7, high = 35, 
                                       denoised_method = 'bandpass',
                                       show=False)
    if res[0]:
        acf_x = res[-1]
        peak_ids, _ = signal.find_peaks(acf_x) #height = np.mean(acf_x))
        # peak_ids, _ = signal.find_peaks(acf_x, height = 0.25)
        time_diff = peak_ids[1:] - peak_ids[:-1]
        
        # check if the peaks is periodic
        cadidates = []
        for peak_id in peak_ids:
            if (peak_id > int(0.25 * fs) and peak_id < int(0.80 * fs)):
                cadidates.append(peak_id)
        
        if len(cadidates) == 0:
            median_hr = np.median(time_diff)
        else:
            # median_hr = cadidates[np.argmax(acf_x[cadidates])] #+ int(0.5 * Fs)
            median_hr = cadidates[0]
        # median_hr = np.median(time_diff)
        frequency = 1/(median_hr/fs) 

        hr = round(frequency * 60, 2) + random.randint(-2,2) + random.uniform(0.0,1.0)

        return hr, acf_x
    else:
        return -1, []

def rr_calculation(data):
    x = data
    x = (x - np.mean(x))/np.std(x)
    cum_x = np.cumsum(x)
    cum_detrended = detrend(cum_x)
    rec_a, rec_d = wavelet_decomposition(data = np.copy(cum_detrended), wave = 'db12', 
                                    Fs = 100, n_decomposition = 9)
    
    for j in range(1,4):
        cut_len = (len(rec_d[-j]) - len(x))//2
        if j == 1:
            filtered_x = rec_d[-j][cut_len :- cut_len]
        else:
            filtered_x += rec_d[-j][cut_len :- cut_len]
    N = 201
    sg_filtered_x = savgol_filter(cum_detrended, N, 3, mode='nearest')
    sg_filtered_x = savgol_filter(sg_filtered_x, 101, 3, mode='nearest')
    MA_x = np.convolve(cum_detrended, np.ones(N)/N, mode='same')
    MA_x = savgol_filter(MA_x, N, 3, mode='nearest')
    
    ##########################################
    acf1 = acf(MA_x, nlags = len(MA_x))
    acf1 = (acf1 - min(acf1))/(max(acf1) - min(acf1))
    acf1 = acf1/sum(acf1)
    acf2 = acf(filtered_x, nlags = len(filtered_x))
    acf2 = (acf2 - min(acf2))/(max(acf2) - min(acf2))
    acf2 = acf2/sum(acf2)
    acf3 = acf(sg_filtered_x, nlags = len(sg_filtered_x))
    acf3 = (acf3 - min(acf3))/(max(acf3) - min(acf3))
    acf3 = acf3/sum(acf3)
    min_len = min(len(acf1), len(acf2), len(acf3))
    acf_x = acf1[:min_len] * acf2[:min_len] * acf3[:min_len]
    rr_peak_ids, _ = find_peaks(acf_x[:len(acf_x)], height = np.mean(acf_x[:len(acf_x)]))
    candidates_rr_ids = []
    for rr_peak_id in rr_peak_ids:
        if rr_peak_id > 239 and rr_peak_id < 750:
            candidates_rr_ids.append(rr_peak_id)
    
    if len(candidates_rr_ids) > 1:
        intv = candidates_rr_ids[np.argmax(acf_x[candidates_rr_ids])]
        
    elif len(candidates_rr_ids) == 1:
        intv = candidates_rr_ids[0]
    else:
        return -1
    rr = 1/(intv/100) * 60
    return rr

def calc_rr(data):
    x = data
    x = (x - np.mean(x))/np.std(x)
    cum_x = np.cumsum(x)
    cum_detrended = detrend(cum_x)
    rec_a, rec_d = wavelet_decomposition(data = np.copy(cum_detrended), wave = 'db12', 
                                    Fs = 100, n_decomposition = 9)
    
    for j in range(1,2):
        cut_len = (len(rec_d[-j]) - len(x))//2
        if j == 1:
            filtered_x = rec_d[-j][cut_len :- cut_len]
        else:
            filtered_x += rec_d[-j][cut_len :- cut_len]

    # filtered_x = band_pass_filter(data=np.copy(cum_detrended),Fs=100, low=0.7,high=35,order=5)
    N = 51
    sg_filtered_x = savgol_filter(cum_detrended, N, 3, mode='nearest')
    sg_filtered_x = savgol_filter(sg_filtered_x, 101, 3, mode='nearest')
    # MA_x = expectation(cum_detrended, 201)
    MA_x = np.convolve(cum_detrended, np.ones(N)/N, mode='same')
    MA_x = savgol_filter(MA_x, N, 3, mode='nearest')
    
    ##########################################
    acf1 = acf(MA_x, nlags = len(MA_x))
    acf1 = (acf1 - min(acf1))/(max(acf1) - min(acf1))
    acf1 = acf1/sum(acf1)
    acf2 = acf(filtered_x, nlags = len(filtered_x))
    acf2 = (acf2 - min(acf2))/(max(acf2) - min(acf2))
    acf2 = acf2/sum(acf2)
    acf3 = acf(sg_filtered_x, nlags = len(sg_filtered_x))
    acf3 = (acf3 - min(acf3))/(max(acf3) - min(acf3))
    acf3 = acf3/sum(acf3)
    min_len = min(len(acf1), len(acf2), len(acf3))
    acf_x = acf1[:min_len] * acf2[:min_len] * acf3[:min_len]
    rr_peak_ids, _ = find_peaks(acf_x[:len(acf_x)],height = np.mean(acf_x[:len(acf_x)])) #height = np.mean(acf_x[:len(acf_x)])
    candidates_rr_ids = []
    for rr_peak_id in rr_peak_ids:
        if rr_peak_id > 119 and rr_peak_id < 400:
            candidates_rr_ids.append(rr_peak_id)
    
    if len(candidates_rr_ids) > 1:
        # intv = candidates_rr_ids[np.argmax(acf_x[candidates_rr_ids])]
        intv = candidates_rr_ids[0]
    elif len(candidates_rr_ids) == 1:
        intv = candidates_rr_ids[0]
    else:
        # return -1
        f, Pxx_den = periodogram(acf_x, fs = 100, nfft = len(acf_x))
        frequency = f[np.argmax(Pxx_den)]
        intv = (1/frequency) * 100
    rr = 1/(intv/100) * 60
    return rr, acf_x

#####################################
def feature_extraction(temp_data, Fs):
    f, Pxx_den = periodogram(np.copy(temp_data), fs=Fs)
    frequency = f[np.argmax(Pxx_den)]
    
    temp_max_min = max(temp_data) - min(temp_data)
    
    denoised_x = band_pass_filter(data = temp_data, Fs = Fs, 
                     low = 0.8, high = 12, order = 5)
    f, Pxx_den = periodogram(denoised_x, fs=Fs)
    denoised_frequency = f[np.argmax(Pxx_den)]
    
    denoise_zcr = zero_cross_rate(denoised_x)
    
    quality = signal_quality_assessment_v3(x=temp_data, Fs=Fs, 
                                           n_lag=len(temp_data) // 2, 
                                           low = 0.8, high = 12, show=False)
    acf_mean_var = quality[1]
    acf_f = quality[2]
    acf_spec_power = quality[3]
    
    peak_ids, _ = signal.find_peaks(quality[-1], 
                                    height = np.mean(quality[-1]))
    acf_prob = max(quality[-1][peak_ids])
    return frequency, temp_max_min, denoised_frequency, denoise_zcr,\
        acf_mean_var, acf_f, acf_spec_power, acf_prob
# loaded_rf = joblib.load("../random_forest.joblib")
#####################################
########################### load deep learning model
# bs_model = torch.load('../BS_on_0.9567_off_0.9903.pth', map_location=torch.device('cpu'))
bs_model = torch.load('BS_acc_0.9401_f1_0.9363.pth', map_location = 'cpu')
device =  "cpu"
import torch.nn.functional as F
from BS_model import TemporalConvNet, classifier, Spec_Net, fusion_classifier
model = TemporalConvNet(num_inputs = 1, first_output_num_channels = 16, num_blocks = 5, kernel_size=3, dropout=0.1).to(device)
if len(bs_model) == 3:
    model.load_state_dict(bs_model['t_model'])
    spec_model = Spec_Net(n_input_channels = 1, n_output_channels = 64, num_blocks = 3, kernel_size=3, stride=2).to(device)
    spec_model.load_state_dict(bs_model['f_model'])
    clf = fusion_classifier(final_out_channels = 256, features_len = 32, n_target = 2).to(device)
else:
    model.load_state_dict(bs_model['model'])
    clf = classifier(final_out_channels = 256, features_len = 32, n_target = 2).to(device)
clf.load_state_dict(bs_model['classifier'])
###########################

def bed_occupancy_detection(data, pre_energy):
    x = np.copy(data)
    if len(x) < 1000:
        bed_status = UNCERTAINTY
        return bed_status
    fs = len(x)/BS_DURATION
    if fs!= 100:
        fs = 100
    denoised_x = wavelet_reconstruction(np.copy(x), fs=fs, 
                                        low = 0.8, high = 12)
    denoise_zcr = zero_cross_rate(denoised_x)

    f, Pxx_den = periodogram(np.copy(x), fs=fs)
    frequency = f[np.argmax(Pxx_den)]

    # quality = signal_quality_assessment_v3(x=x, Fs=fs, n_lag=len(x) // 2, 
    #                                        low = 0.8, high = 12, show=False)
        
    # temp_features = feature_extraction(x, fs)
    # temp_features = np.expand_dims(temp_features, 0)
    
    # rf_probs = loaded_rf.predict_proba(temp_features)[0]
    # bed_status = np.argmax(rf_probs)
    # bed_status_prob = rf_probs[bed_status]

    ###########deep learning model
    x = (x - np.mean(x))/np.std(x)
    if len(bs_model) == 3:
        f, x_f = periodogram(x, fs=fs)
        x_f = np.sqrt(x_f)
        x_f = np.expand_dims(x_f, 0)
        x_f = np.expand_dims(x_f, 0)
        x_f = torch.tensor(x_f.astype(np.float32)).to(device)
    x = band_pass_filter(x, fs, 2, 10, 5)
    temp = np.expand_dims(x, 0)
    temp = np.expand_dims(temp, 0)
    temp = torch.tensor(temp.astype(np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        features = model(temp)
        if len(bs_model) == 3:
            f_features = spec_model(x_f)
            preds, tf_features = clf(features, f_features)
        else:
            preds = clf(features)
        prob = F.softmax(preds, dim=1)
        prob = prob.cpu().numpy()[0]
        # np.set_printoptions(formatter={'float_kind':"{:.2f}".format})
        # print('probability:', prob)
    bed_status = np.argmax(prob)
    bed_status_prob = prob[bed_status]
    ############################
    
    if bed_status_prob < 0.65:
        bed_status = UNCERTAINTY
    # print('rf probs:', rf_probs)
        
    
    if kurtosis(x) > 10:# or (bed_status == OFF_BED and max(data) - min(data) > 20000):
        # or (bed_status == ON_BED and max(data) - min(data) > 50000):
        print('max - min:', max(data) - min(data))
        bed_status = UNCERTAINTY
    return bed_status, denoise_zcr, frequency

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def freq_com_select(Fs, low, high):
    n = 0
    valid_freq = Fs/2
    temp_f = valid_freq
    min_diff_high = abs(temp_f - high)
    min_diff_low = abs(temp_f - low)
    
    while(temp_f > low):
        temp_f = temp_f / 2
        n += 1
        diff_high = abs(temp_f - high)
        diff_low = abs(temp_f - low)
        if diff_high < min_diff_high:
            max_n = n
            min_diff_high = diff_high
        if diff_low < min_diff_low:
            min_n = n
            min_diff_low = diff_low
    return n, max_n, min_n

def wavelet_reconstruction(x, fs, low, high):
    n_decomposition, max_n, min_n = freq_com_select(Fs = fs, low = low, high = high)
    rec_a, rec_d = wavelet_decomposition(data = x, wave = 'db12', Fs = fs, n_decomposition = n_decomposition)
    min_len = len(rec_d[max_n])
    for n in range(max_n, min_n):
        if n == max_n:
            denoised_sig = rec_d[n][:min_len]
        else:
            denoised_sig += rec_d[n][:min_len]
    cut_len = (len(denoised_sig) - len(x))//2
    denoised_sig = denoised_sig[cut_len:-cut_len]
    return denoised_sig

def signal_quality_assessment_v3(x, Fs, n_lag, low, high,
                                 denoised_method, show = False):
    # x = (x - np.mean(x))/np.std(x)
    if denoised_method == 'DWT':
        denoised_sig = wavelet_reconstruction(x = x, fs = Fs, 
                                              low = low, high = high)

    elif denoised_method == 'bandpass':
        denoised_sig = band_pass_filter(data=x, Fs=Fs, 
                                        low=low, high= high, order=5)
    index = 0
    window_size = int(Fs)
    z= hilbert(denoised_sig) 
    # z = x 
    envelope = np.abs(z)
    
    sg_win_len = round(0.11 * Fs)
    if sg_win_len%2 == 0:
        sg_win_len -= 1
    smoothed_envelope = savgol_filter(envelope, sg_win_len, 3, mode='nearest')
    smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope))/np.std(smoothed_envelope)
    
    sg_win_len = round(1.01 * Fs)
    if sg_win_len%2 == 0:
        sg_win_len -= 1
    trend = savgol_filter(smoothed_envelope, sg_win_len, 7, mode='nearest')
    smoothed_envelope = smoothed_envelope - trend
    smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope))/np.std(smoothed_envelope)
    acf_x = acf(smoothed_envelope, nlags=n_lag)
    acf_x = acf_x / acf_x[0]
    
    nfft = next_power_of_2(x = len(x) * 2)
    f, Pxx_den = periodogram(acf_x, fs = Fs, nfft = nfft)

    sig_means = []
    index = 0
    frequency = f[np.argmax(Pxx_den)]
    power = max(Pxx_den)
    if show:
        fig, ax = plt.subplots(6, 1, figsize=(16, 18))
        ax[0].plot(x, label='raw data')
        ax[1].plot(denoised_sig, label='denoised data')
        ax[2].plot(envelope, label='envelope extraction by Hilbert transform')
        ax[3].plot(smoothed_envelope, label='smoothed envelope')
        ax[4].plot(acf_x, label='ACF of smoothed envelope')
        ax[5].plot(f, Pxx_den, label='spectrum of ACF')
        for i in range(len(ax)):
            ax[i].legend()

    while (index + window_size < len(acf_x)):
        sig_means.append(np.mean(acf_x[index:index + window_size]))
        index = index + window_size
    if np.std(sig_means) < 0.1 and 2 < frequency < 3.5 and power > 0.23:
        res = [True, np.std(sig_means), frequency, power, acf_x]
        if show:
            fig.suptitle('good data')
    else:
        res = [False, np.std(sig_means), frequency, power, acf_x]
        if show:
            fig.suptitle('bad data')
    return res


def wavelet_denoise_v3(signal, Fs, n_decomposition):
    # x = (x - np.mean(x))/np.std(x)
    x = standize(signal)
    rec_a, rec_d = wavelet_decomposition(data=x, wave='db12', Fs=Fs, n_decomposition=n_decomposition)

    min_len = min(len(rec_d[-1]), len(rec_d[-2]), len(rec_d[-3]), len(rec_d[-4]))  # len(rec_a[-1]) len(rec_d[-5])
    denoised_sig = rec_d[-1][:min_len] + rec_d[-2][:min_len] + rec_d[-4][:min_len] + rec_d[-3][
                                                                                     :min_len]  # + rec_a[-1][:min_len]
    return denoised_sig
