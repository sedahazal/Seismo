from Time_Series import vel_and_disp
from numpy import arange,where,mean,pi
from scipy.integrate import cumtrapz
from Response_Spectra import ResponseSpectra
from PreProcess import butter_lowpass_filter
from FAS import FourierAmplitude
import matplotlib.pyplot as plt
class GMP:
    def __init__(self,acceleration,dt):
        self.accelerations = acceleration
        self.dt = dt
        self.times = arange(0,dt*len(acceleration),dt)
        self.velocities, self.displacements = vel_and_disp(acceleration,dt)
        self.g = 9.81

    def max_acceleration(self):
        pga = max(abs(self.accelerations))
        pga_time = self.times[where(self.accelerations == pga )[0][0]]

        return round(pga,5),round(pga_time,5)

    def max_velocity(self):
        pgv = max(abs(self.velocities))
        pgv_time = self.times[where(self.velocities == pgv )[0][0]]

        return round(pgv,5),round(pgv_time,5)

    def max_displacement(self):
        pgd = max(abs(self.displacements))
        pgd_time = self.times[where(self.displacements == pgd )[0][0]]

        return round(pgd,5),round(pgd_time,5)

    def vmax_amax(self):
        pga,_ = self.max_acceleration()
        pgv,_ = self.max_velocity()

        return pgv/(pga*self.g*100)

    def a_rms(self):
        Td = self.times[-1]

        return (cumtrapz(self.accelerations**2,dx=self.dt)/Td)[-1]**0.5

    def v_rms(self):
        return mean(self.velocities[1:]**2)**0.5

    def d_rms(self):
        return mean(self.displacements[1:] ** 2) ** 0.5

    def arias_intensity(self):
        return (cumtrapz((self.accelerations*self.g) ** 2, dx=self.dt)*(0.5*pi/self.g))[-1]

    def characteristic_intensity(self):
        return (self.a_rms()**1.5)*(self.times[-1]**0.5)

    def specific_energy_density(self):
        return cumtrapz(self.velocities**2,dx=self.dt)[-1]

    def housner_intensity(self):
        T = arange(0.1,2.6,0.1)
        psa,psv,psd = ResponseSpectra(self.accelerations,self.dt,T)

        return cumtrapz(psv,dx=0.01)[-1]

    def sustained_max_acceleration(self):
        SMA = 0
        for i in range(2,len(self.accelerations)):
            acc =  sorted(abs(self.accelerations),reverse=True)[i]
            index = where(self.accelerations == acc)[0][0]
            try:
                left = self.accelerations[(index-20):index]
            except:
                left = self.accelerations[:index]

            try:
                right = self.accelerations[index+1:(index+21)]
            except:
                right = self.accelerations[index+1:]
            if len(where(acc > left)[0]) == 20 and len(where(acc > right)[0]) == 20:
                SMA = acc
                break
        return SMA

    def sustained_max_velocity(self):
        SMV = 0
        for i in range(2, len(self.velocities)):
            velocity = sorted(abs(self.velocities), reverse=True)[i]
            index = where(self.velocities == velocity)[0][0]
            try:
                left = self.velocities[(index - 20):index]
            except:
                left = self.velocities[:index]

            try:
                right = self.velocities[index + 1:(index + 21)]
            except:
                right = self.velocities[index + 1:]
            if len(where(velocity > left)[0]) == 20 and len(where(velocity > right)[0]) == 20:
                SMV = velocity
                break
        return SMV

    def effective_design_acceleration(self):
        filtered_data = butter_lowpass_filter(self.accelerations,9,self.dt,order=1)
        return max(abs(filtered_data))

    def A95(self):
        #Tekrar bak
        Ia = self.arias_intensity()
        for n in range(int(len(self.accelerations)*0.9),len(self.accelerations)):
            Ia_new = (cumtrapz((self.accelerations[:n]*self.g) ** 2, dx=self.dt)*(0.5*pi/self.g))[-1]
            rate = 100*Ia_new/Ia
            if rate >94.99:
                A95 = self.accelerations[n]
                break

        return A95

    def predomimant_period(self):
        T = arange(0.02,4.02,0.02)
        psa,psv,psd = ResponseSpectra(self.accelerations,self.dt,T)

        return T[where(psa==max(psa))[0][0]]

    def mean_period(self):
        #Tekrar bak
        fa,f = FourierAmplitude(self.accelerations,self.dt)
        f_list = f[(f>=0.25) & (f<=20)]
        A = 0
        B = 0
        for i in f_list:
            Ci = fa[where(f == i)[0][0]]
            A += (Ci**2)/i
            B += Ci**2

        return A/B