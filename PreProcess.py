from numpy import poly1d,polyfit,array
from scipy.signal import butter,lfilter

def baseline_correction(acceleration,time,order):
    constants = polyfit(time,acceleration,order)
    f = poly1d(constants)
    predicted = f(time)
    return array(acceleration) - predicted

def butter_bandpass_filter(data,lowcut, highcut, dt, order=4):
    nyq = 0.5 / dt
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, dt, order=4):
    nyq = 0.5 / dt
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y