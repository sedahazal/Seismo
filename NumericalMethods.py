def bisection(x1,x2, f,accelerations,dt,Ia):
    c = (x1 + x2)/2
    while abs(f(accelerations,c,dt,Ia)) > 0.001:
        if f(accelerations,c,dt,Ia) > 0:
            x1 = c
        else:
            x2 = c
        c = (x1 + x2) / 2
    return c