import numpy as np
import matplotlib.pyplot as plt
import pylab as py
from matplotlib import rcParams
#import Image




array=np.zeros((1600,2))
array=np.loadtxt("./cau_1.dat") 



###=============================================================================
plt.clf()
#plt.plot(array[1200:1600,0],array[1200:1600,1], "r-")
plt.plot(array[800:1000,0],array[800:1000,1], "b.")
#plt.xlabel(r"$(t-t0)/tE$")
#plt.ylabel(r"$Magnification$")
fig3=plt.gcf()
fig3.savefig("./lightcurve.png")
print(">>>>>>>>>>>>>>>>>>>>>>>> The model light curve was made <<<<<<<<<<<<<<<")


'''


###=============================================================================
plt.clf()
plt.plot(array[:,3],array[:,5],'ro', ms=0.3)
plt.xlabel(r"$(t-t0)/tE$")
plt.ylabel(r"$polarization[\%]$")
fig3=plt.gcf()
fig3.savefig("./files/polcurve.png")
print(">>>>>>>>>>>>>>>>>>>>>>>> The model light curve was made <<<<<<<<<<<<<<<")


###=============================================================================


array1=np.zeros((31608,9))
array1=np.loadtxt("./files/pol_intrinsicb.txt") 

plt.clf()
plt.plot(array1[:,3],array1[:,4],'ro', ms=0.1)
plt.xlabel(r"$R(R_sun)$")
plt.ylabel(r"$polarization[\%]$")
fig3=plt.gcf()
fig3.savefig("./files/polin.png")
print(">>>>>>>>>>>>>>>>>>>>>>>> The model light curve was made <<<<<<<<<<<<<<<")

'''


