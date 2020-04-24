from numpy import loadtxt,arange
from GroundMotionParameters import GMP
from PreProcess import *
from Spectra import FourierAmplitude as FA
from Time_Series import time_series
from Graphs import graph_creater
from Cython.AmplitudeSpectra import FourierAmplitude
class SeismoAnalysis:
    def __init__(self,acceleration,dt,user_name,width,height,output_path):
        self.acceleration = acceleration
        self.processed_acceleration = self.acceleration
        self.dt = dt
        self.time = arange(0,dt*len(accelerations),dt)
        self.user_name = user_name
        self.width = width
        self.height = height
        self.output_path = output_path

    def PreProcess(self,filter_type,filter_configuration,lowcut,highcut,order,bc_order,baseline_correction=False, filtering=False):

        if baseline_correction:
             self.processed_acceleration = BaselineCorrection(self.acceleration,self.time,bc_order)

        if filtering:
            self.processed_acceleration = Filtering(self.processed_acceleration,self.dt,filter_type,filter_configuration,lowcut,highcut,order)

    def TimeHistory(self):
        self.velocity,self.displacement = time_series(self.processed_acceleration,self.dt)
        graph_creater(self.output_path,"{}_acceleration_time.html".format(self.user_name),self.time, self.processed_acceleration, "Zaman(s)", "İvme(g)", self.width, self.height)
        graph_creater(self.output_path,"{}_velocity_time.html".format(self.user_name),self.time, self.velocity, "Zaman(s)", "Hız(cm/s)", self.width, self.height)
        graph_creater(self.output_path,"{}_displacement_time.html".format(self.user_name),self.time, self.displacement, "Zaman(s)", "Yer Değiştirme(cm)",self.width, self.height)

    def ResponseSpectrum(self,damping,period):
        response_spectrum = ResponseSpectra(self.processed_acceleration,self.dt,period,damping)
        self.spectral_displacement = response_spectrum["Displacement"]
        self.spectral_acceleration = response_spectrum["Acceleration"]
        self.spectral_velocity = response_spectrum["Velocity"]
        self.pseudo_acceleration = response_spectrum["Pseudo-Acceleration"]
        self.pseudo_velocity = response_spectrum["Pseudo-Velocity"]

        graph_creater(self.output_path,"{}_SpectralAcceleration_period.html".format(self.user_name), period, self.spectral_acceleration, "Periyot(s)", "Spektral İvme(g)", self.width, self.height)
        graph_creater(self.output_path,"{}_SpectralVelocity_period.html".format(self.user_name), period, self.spectral_velocity, "Periyot(s)", "Spektral Hız(cm/s)", self.width, self.height)
        graph_creater(self.output_path,"{}_Displacement_period.html".format(self.user_name), period, self.spectral_displacement, "Periyot(s)", "Yer Değiştirme(cm)", self.width, self.height)
        graph_creater(self.output_path,"{}_PseudoAcceleration_period.html".format(self.user_name), period, self.pseudo_acceleration, "Periyot(s)", "Pseudo İvme(g)", self.width, self.height)
        graph_creater(self.output_path,"{}_PseudoVelocity_period.html".format(self.user_name), period, self.pseudo_velocity, "Periyot(s)", "Pseudo Hız(cm)", self.width, self.height)

    def AmplitudeSpectra(self):
        self.frequency,self.fourier_amplitude, self.power_amplitude = FourierAmplitude(self.processed_acceleration,self.dt)
        graph_creater(self.output_path,"{}_FourierSpectra.html".format(self.user_name), self.frequency, self.fourier_amplitude, "Frekans(Hz)", "Fourier Büyüklüğü", self.width, self.height,"log")
        graph_creater(self.output_path,"{}_PowerSpectra.html".format(self.user_name), self.frequency, self.power_amplitude, "Frekans(Hz)", "Spektral Yoğunluk", self.width, self.height,"log")

    def ground_motion_parameters(self):
        gmp = GMP(self.processed_acceleration,self.velocity,self.displacement,self.dt)
        self.pga,self.pga_time = gmp.max_acceleration()
        self.pgv,self.pgv_time = gmp.max_velocity()
        self.pgd,self.pgd_time = gmp.max_displacement()
        self.vmax_amax = gmp.vmax_amax()
        self.a_rms = gmp.a_rms()
        self.v_rms = gmp.v_rms()
        self.d_rms = gmp.d_rms()
        self.arias_intensity_list = gmp.arias_intensity()
        self.arias_intensity = self.arias_intensity_list[-1]
        self.characteristic_intensity = gmp.characteristic_intensity()
        self.specific_energy_density_list = gmp.specific_energy_density()
        self.spesific_energy_density = self.specific_energy_density_list[-1]
        self.housner_intensity = gmp.housner_intensity()
        self.sustained_max_acceleration = gmp.sustained_max_acceleration()
        self.sustained_max_velocity = gmp.sustained_max_velocity()
        self.effective_design_acceleration = gmp.effective_design_acceleration()
        self.cumulative_absolute_velocity = gmp.cumulative_absolute_velocity()
        self.acceleration_spectrum_intensity = gmp.acceleration_spectrum_intensity()
        self.velocity_spectrum_intensity = gmp.velocity_spectrum_intensity()
        self.A95 = gmp.A95()
        self.predominant_period = gmp.predomimant_period()
        self.mean_period = gmp.mean_period()
        self.uniform_duration = gmp.uniform_duration()
        self.bracketed_duration = gmp.bracketed_duration()
        self.significant_duration = gmp.significant_duration()
        self.effective_duration = gmp.effective_duration(0.48*self.arias_intensity,0.96*self.arias_intensity)



        graph_creater(self.output_path,"{}_AriasIntensity.html".format(self.user_name), self.time, 100*self.arias_intensity_list/self.arias_intensity, "Zaman(s)", "Arias Yoğunluğu(%)", self.width, self.height)
        graph_creater(self.output_path,"{}_EnergyFlux.html".format(self.user_name), self.time, 100*self.specific_energy_density_list/self.spesific_energy_density, "Zaman(s)", "Enerji Akısı(cm2/s)", self.width, self.height)


from time import perf_counter

def read_file(file_name,column):
    a1,a2,a3,a4,a5,a6,a7,a8 = loadtxt(file_name,skiprows=0,unpack=True)
    return [a1,a2,a3,a4,a5,a6,a7,a8][column-1]

vibration_file = "FieldResults/open trench/oc4.txt"
accelerations = read_file(vibration_file,1)
dt = 0.002
times = arange(0,dt*len(accelerations),dt)
t1 = perf_counter()
"""S = SeismoAnalysis(accelerations,dt,"nubufi",1100,660,"FieldResults")
S.TimeHistory()
S.ResponseSpectrum(0.05,arange(1,10.1,0.1))
S.AmplitudeSpectra()
S.ground_motion_parameters()"""
f,fa,pa = FA(accelerations,dt)
t2 = perf_counter()

print(t2 - t1)

t3 = perf_counter()
f1,fa1,pa1 = FourierAmplitude(accelerations,dt)
t4 = perf_counter()

print(t4-t3)
