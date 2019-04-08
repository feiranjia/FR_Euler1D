import matplotlib.pyplot as plt
plt.style.use('sjc')
fig=plt.figure()
ax=fig.gca()
import numpy as np
wid=2
mid=-2
npts=201

x_arr=np.linspace(mid-wid,mid+wid,npts)

sigma_arr=0.5*(1-np.sin(np.pi*(x_arr-mid)/(2.0*wid)))
ax.plot(x_arr,sigma_arr,'o-',label="SIN")

sigma_arr=0.5*(-np.tanh((x_arr-mid)*wid)+1)
ax.plot(x_arr,sigma_arr,'s-',label="TANH")

ax.legend()
plt.show()

