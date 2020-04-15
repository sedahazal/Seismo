from numpy import loadtxt,arange
from GroundMotionParameters import GMP
from PreProcess import baseline_correction
def read_file(file_name,column):
    a1,a2,a3,a4,a5,a6,a7,a8 = loadtxt(file_name,skiprows=0,unpack=True)
    return [a1,a2,a3,a4,a5,a6,a7,a8][column-1]

vibration_file = "FieldResults/open trench/oc4.txt"
accelerations = read_file(vibration_file,1)
times = arange(0,0.002*len(accelerations),0.002)

gmp = GMP(accelerations,0.002)
print(gmp.mean_period())