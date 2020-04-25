from numpy import arange,where,mean,isclose
from Spectra import *
from PreProcess import *

class GMP:
    def __init__(self,acceleration,velocity,displacement,dt,response_spectra,period):
        self.accelerations = acceleration
        self.dt = dt
        self.times = arange(0,dt*len(acceleration),dt)
        self.velocities = velocity
        self.displacements = displacement
        self.g = 9.81
        self.response_spectra = response_spectra
        self.period = period

    def max_acceleration(self):
        pga = max(abs(self.accelerations))
        pga_time = self.times[where(abs(self.accelerations) == pga )[0][0]]

        return round(pga,5),round(pga_time,5)

    def max_velocity(self):
        pgv = max(abs(self.velocities))
        pgv_time = self.times[where(abs(self.velocities) == pgv )[0][0]]

        return round(pgv,5),round(pgv_time,5)

    def max_displacement(self):
        pgd = max(abs(self.displacements))
        pgd_time = self.times[where(abs(self.displacements) == pgd )[0][0]]

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
        return (cumtrapz((self.accelerations*self.g) ** 2, dx=self.dt)*(0.5*pi/self.g))

    def characteristic_intensity(self):
        return (self.a_rms()**1.5)*(self.times[-1]**0.5)

    def specific_energy_density(self):
        return cumtrapz(self.velocities**2,dx=self.dt)

    def housner_intensity(self):
        if self.period[-1] >= 2.5:
            dx = self.period[2] - self.period[1]
            index = isclose(self.period,2.5).nonzero()[0][0]+1
            psv = self.response_spectra["Velocity"][:index]
        else:
            response_spectra = ResponseSpectra(self.accelerations,self.dt,arange(0.1,2.51,0.1))
            psv = response_spectra["Velocity"]
            dx = 0.1
        return cumtrapz(psv,dx=dx)[-1]

    def sustained_max_acceleration(self):
        SMA = 0
        if all(sorted(abs(self.velocities)) != self.velocities):
            for i in range(2,len(self.accelerations)):
                acc =  sorted(abs(self.accelerations),reverse=True)[i]
                index = where(abs(self.accelerations) == acc)[0][0]
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
        if all(sorted(abs(self.velocities)) != self.velocities):
            for i in range(2, len(self.velocities)):
                velocity = sorted(abs(self.velocities), reverse=True)[i]
                index = where(abs(self.velocities) == velocity)[0][0]
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
        filtered_data = Filtering(self.accelerations,self.dt,"Butterworth","low",9,15,1)
        return max(abs(filtered_data))

    def cumulative_absolute_velocity(self):
        return cumtrapz(abs(self.accelerations)*self.g, dx=self.dt)[-1]*100

    def acceleration_spectrum_intensity(self):
        if self.period[-1]<0.5:
            dx = 0.1
            T = arange(0.1, 0.51, dx)
            response_spectrum = ResponseSpectra(self.accelerations, self.dt, T)
            psa = response_spectrum["Acceleration"]

        else:
            dx = self.period[2] - self.period[1]
            index = isclose(self.period, 0.5).nonzero()[0][0] + 1
            psa = self.response_spectra["Acceleration"][:index]

        return cumtrapz(psa, dx=dx)[-1]

    def velocity_spectrum_intensity(self):
        if self.period[-1] < 2.5:
            dx = 0.1
            T = arange(0.1, 2.51, dx)
            response_spectrum = ResponseSpectra(self.accelerations, self.dt, T)
            psv = response_spectrum["Velocity"]

        else:
            dx = self.period[2] - self.period[1]
            index = isclose(self.period, 2.5).nonzero()[0][0] + 1
            psv = self.response_spectra["Velocity"][:index]

        return cumtrapz(psv, dx=dx)[-1]

    def A95(self):
        def f(accelerations,a95,dt,Ia):
            diff = accelerations**2 - a95**2
            positives = [0 if i<0 else i for i in diff]
            Ia_new = cumtrapz(positives,dx=dt)[-1]
            return Ia_new/Ia - 0.05

        def bisection(f, accelerations, dt, Ia):
            x1 = 0
            x2 = max(abs(accelerations))
            c = (x1 + x2) / 2
            while abs(f(accelerations, c, dt, Ia)) > 0.001:

                if f(accelerations, c, dt, Ia) > 0:
                    x1 = c
                else:
                    x2 = c
                c = (x1 + x2) / 2
            return c

        a95 = bisection(f,self.accelerations,self.dt,cumtrapz(self.accelerations**2,dx=self.dt)[-1])
        return a95

    def predomimant_period(self):
        psa = self.response_spectra["Acceleration"]

        return self.period[where(psa==max(psa))[0][0]]

    def mean_period(self):
        f,fa,pa = FourierAmplitude(self.accelerations,self.dt)
        f_list = f[(f>=0.25) & (f<=20)]
        A = 0
        B = 0
        for i in f_list:
            Ci = fa[where(f == i)[0][0]]
            A += (Ci**2)/i
            B += Ci**2

        return A/B

    def uniform_duration(self,A0 = "Default"):
        if A0 == "Default":
            A0 = max(abs(self.accelerations))*0.05

        higher = len(where(self.accelerations**2 > A0**2)[0]) - 1
        return higher*self.dt

    def bracketed_duration(self,A0 = "Default"):
        if A0 == "Default":
            A0 = max(abs(self.accelerations))*0.05

        t1 = self.times[where(self.accelerations**2 >= A0**2)[0][0]]
        t2 = self.times[where(self.accelerations**2 >= A0**2)[0][-1]]
        return t2 - t1 + self.dt

    def significant_duration(self,p1 = 5, p2 = 95):
        Ia = self.arias_intensity()
        normalized_Ia = 100*Ia/max(Ia)
        t1 = self.times[where(normalized_Ia <= p1)[0][-1]]
        t2 = self.times[where(normalized_Ia >= p2)[0][0]]

        return t2 - t1 - self.dt

    def effective_duration(self,Ia1, Ia2):
        Ia = self.arias_intensity()
        t1 = self.times[where(Ia <= Ia1)[0][-1]]
        t2 = self.times[where(Ia >= Ia2)[0][0]]

        return t2 - t1 - self.dt