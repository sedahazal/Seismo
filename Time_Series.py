from scipy.integrate import cumtrapz
from numpy import insert

def vel_and_disp(acceleration,dt):
    vel = cumtrapz(acceleration*981,dx=dt)
    vel = insert(vel,0,0)
    disp = cumtrapz(vel,dx=dt)
    disp = insert(disp, 0, 0)
    return vel,disp



