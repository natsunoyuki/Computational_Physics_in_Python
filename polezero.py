#for reference this line is exactly 80 characters long#########################
from __future__ import division
from scipy import *
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import signal
import time

#This script loads a seismometer response file with the following format:
"""
POLES <N_POLES>
REAL_P1 IMAG_P1
...
ZEROS <N_ZEROS>
REAL_Z1  IMAG_Z1
CONSTANT <CONSTANT>
"""
#and calculates the corresponding complex seismometer response

def readpzfile(pzfilename):
    f=open(pzfilename,'r')
    X=f.readlines()
    f.close()
    NPOLES=int(X[0][6])
    POLES=zeros(NPOLES,'complex')
    for i in range(NPOLES):
        buf=X[i+1]
        buf=buf.strip().split()
        POLES[i]=float(buf[0])+float(buf[1])*1j
    NZEROS=int(X[NPOLES+1][6])
    ZEROS=zeros(NZEROS,'complex')
    for i in range(NZEROS):
        buf=X[i+NPOLES+2]
        buf=buf.strip().split()
        if buf[0]=='CONSTANT':
            break
        else:
            ZEROS[i]=float(buf[0])+float(buf[1])*1j
    CONSTANT=float(X[-1].strip().split()[1])
    return POLES, ZEROS, CONSTANT

#pzfilename='/users/user/kirishima_invert/g3t100'
#pzfilename='/users/user/kirishima_invert/tri120p'
#pzfilename='/users/user/kirishima_invert/tri40'
#pzfilename='/users/user/kirishima_invert/usgsdst'
pzfilename='/home/user/kirishima_invert/tri120p'
#pzfilename='/home/user/kirishima_invert/g3t100'
#pzfilename='/home/user/kirishima_invert/none'
POLES,ZEROS,CONSTANT=readpzfile(pzfilename)

#apparently in some cases the poles and zeros are all in Hz,
#so to convert to Rad/s multiply by 2*pi
#POLES=POLES*2*pi
#ZEROS=ZEROS*2*pi

print "Poles:", POLES
print "Zeros:", ZEROS
print "Constant:", CONSTANT

#calculate the gain and phase by substituting for s the value
#s = i omega = i 2 pi f and then calculate the modulus of H:
ds=0.01
s=arange(0,100,ds)*1j #this needs to be imaginary!
#s has units of Rad/s
H=ones(len(s))

#multiple the zeros
for i in range(len(ZEROS)):
    H=H*(s-ZEROS[i])

#multiple the poles:
for i in range(len(POLES)):
    H=H/(s-POLES[i])

#obtain the final seismometer response:
H=H*CONSTANT

#obtain the phase and amplitude information from the
#complex transfer function

pha=angle(H)
amp=abs(H)
mag=20*log10(amp)

plt.subplot(2,1,1)
plt.semilogx(imag(s),mag)
plt.ylabel('Amplitude')
plt.grid('on')
plt.subplot(2,1,2)
plt.semilogx(imag(s),pha/pi*180)
plt.ylabel('Phase')
plt.grid('on')
plt.xlabel('Frequency')
plt.show()
