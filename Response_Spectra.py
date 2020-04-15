from numpy import array,logspace,longfloat,append,cos,sin,exp,pi
import matplotlib.pyplot as plt
def ResponseSpectra(data,dt,T=logspace(-2,0.6)):
    n = len(T)
    data_n = len(data)
    PSA = PSV = PSD = array([],dtype=longfloat)
    ksi = 0.05
    for i in range(0,n):
        u = vel = array([0],dtype=longfloat)
        w = ((2*pi)/T[i])
        wd = (w * (1 - ksi ** 2) ** (1 / 2))
        E = cos(wd*dt)*exp(-ksi*w*dt)
        F = sin(wd*dt)*exp(-ksi*w*dt)
        A11 = E + (ksi*w/wd)*F
        A12 = F/wd
        A21 = -(w**2)*A12
        A22 = E - (ksi*w/wd)*F
        B11 = (A11 - 1) / w**2
        B12 = (A12 - 2 * ksi * w * B11 - dt) / ((w**2) * dt)
        B21 = -A12
        B22 = B11 / dt

        for j in range(1,data_n):
            uj = ((A11*u[j-1]+A12*vel[j-1]+B11*data[j-1]+B12*(data[j]-data[j-1])))
            u=append(u,uj)
            velj = (A21*u[j-1]+A22*vel[j-1]+B21*data[j-1]+B22*(data[j]-data[j-1]))
            vel=append(vel,velj)

        Sd = max(abs(array(u)))
        psa = (w**2)*Sd
        psv = Sd*w
        psd = Sd
        PSA= append(PSA,psa)
        PSV = append(PSV,psv)
        PSD = append(PSD,psd)

    PSV*=981
    PSD*=981

    return PSA,PSV,PSD

