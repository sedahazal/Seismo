from numpy import fft,array

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def FourierAmplitude(data,dt):
    n = nextpow2(len(data))
    fftx = abs(fft.fft(data,n))
    fa = (fftx[0:int(n/2+1)])*dt
    delf = 1/(n*dt)
    f = array(range(1,int(n/2+2)))*delf
    return fa,f

