"""
Mixed: Campo total con los modos large-scale y small-scale

Lorenz 1996 adaptation from Grooms' idea,  Grooms NPG 2015

"""
import os
import numpy as np
import numpy.random as rnd
from scipy.linalg import sqrtm
import sys;  sys.path.insert(0, '../')
from mdl.l96 import M as L96 
import scipy.interpolate as interpol
from scipy.fftpack import fft, ifft
from scipy import fftpack

class M(L96):
    def __init__(self,*args,F=8,h=0.5,nx=1312,nl=41, **kwargs):
        super().__init__(*args, **kwargs) # inicializo con los mismos atributos

        if (nl/2 == nl//2):
            sys.exit('nl must be even')
        self.F=F
        self.h=h
        self.nx=nx # total
        self.nl=nl # number of large scale variables
        self.ns=self.nx//self.nl # small scale variable per large scale
        # variables comunes (x)
        self.indm1=list(range(-1,self.nx-1))
        self.indp1=list(np.arange(1,self.nx+1)%self.nx)
        self.indp2=list(np.arange(2,self.nx+2)%self.nx)
        # large scale variables (xl)
        self.indlm2=list(range(-2,self.nl-2))
        self.indlm1=list(range(-1,self.nl-1))
        self.indlp1=list(np.arange(1,self.nl+1)%self.nl)
        
        self.modes=list(np.concatenate([np.arange(0,self.nl//2+1),np.arange(-self.nl//2+1,0)])) # select modes of the large scale
        #assumes nl is even
        
    def small2large2(self,x):
        "NO Promedio en un entorno para obtener variable large-scale "
        return x.reshape(self.ns, self.nl) @ np.ones(self.nl)
    def large2small2(self,y):
        "NO FUNCIONA AUN interpolo usando splines cubicos "
        x=np.arange(self.nl)
        spl=interpol.splrep(x,y,k=3)
        x2 = np.linspace(0,self.nl,self.nx,endpoint=False)
        y2 = interpol.splev(x2, spl) # Falta pegar extremos para periodicidad
        return y2
    
    def small2large(self,x):
        " transforma fast fourier y recorta small-scale"
        xhat=fft(x)
        return np.real(ifft(xhat[self.modes]))
    
    def large2small(self,xl):
        " transformo, expando y antitransformo al espacio fisico"
        xhatl=fft(xl)
        xhat=np.zeros(self.nx, dtype=complex)
        xhat[self.modes]=xhatl 
        return np.real(ifft(xhat))
    
    def _mdl_ls(self,x):
        " L96sum model equations"
        xl = self.small2large(x)
        dxl = xl[self.indlm1] * (xl[self.indlp1]-xl[self.indlm2]) #- xl + self.F 
        dx = self.large2small(dxl) / self.ns #esto no se condice con Grooms
        dx = dx + ( self.h * x[self.indp1] * (x[self.indm1] - x[self.indp2]) 
                    + self.F  - x )
        return dx * self.dt
    
    # rk4 y pinteg son tomados de l96
    #  initilization se usa el mismo que en el l96 con ny=0

if __name__=="__main__":
    import matplotlib.pyplot as plt

# para correr solo con l96 large-scale (comentar _mdl_ls) 
#    Mdl=M(h=0.0,expdir='tmp/',nx=40,nl=40)

    Mdl=M(h=0.5,expdir='tmp/',nl=41,nx=2624,F=8)
    # only large scale initial condition
    #xl =1.0*rnd.normal(0, 1, Mdl.nl)
    #x = Mdl.large2small(xl)

    # all modes
    x  = 1.0*rnd.normal(0, 1, Mdl.nx) # randon initial condition
    nt=1000
    x_t = np.zeros((Mdl.nx,nt))
    x_t[:,0]=x
    for it in range(1,nt):
        x_t[:,it] = Mdl.integ(x_t[:,it-1])
        #print(np.max(x_t[:,it]),np.min(x_t[:,it]))
        
    plt.imshow(x_t[:,500:],aspect='auto')
    #plt.show()
    plt.savefig('tmp/l96sum_f8.png')
