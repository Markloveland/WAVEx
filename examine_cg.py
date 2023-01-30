import numpy as np
import CFx.wave

npoints = 150
x = np.linspace(0,3959,npoints)
y = np.zeros(x.shape)
sigma_vec = 2*np.pi/10*np.ones(x.shape)
theta_vec = np.zeros(x.shape)
depth = 20 -x/200
u = np.zeros(x.shape)
v = np.zeros(x.shape)
dHdx=(-1/200)*np.ones(x.shape)
dHdy = np.zeros(x.shape)
dudx=np.zeros(x.shape)
dudy=np.zeros(x.shape)
dvdy=np.zeros(x.shape)
dvdx = np.zeros(x.shape)

c,cph,k = CFx.wave.compute_wave_speeds_pointwise(x,y,sigma_vec,theta_vec,depth,u,v,dHdx=dHdx,dHdy=dHdy,dudx=dudx,dudy=dudy,dvdx=dvdx,dvdy=dvdy,g=9.81)

np.savetxt('cph_A11.txt',cph,delimiter=',')
