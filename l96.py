#!/usr/bin/python
"""
Lorenz 1996 dynamical system Lorenz-96
   One and two scale
   Integrate model equations with Runge-Kuta 4th order
   Works with ensemble if input is an ensemble

L96
\[ d_t X_k = - X_{k-1} (X_{k-2} - X_{k+1} ) - X_k + F \]
L96 two scale
\[ d_t X_k = - X_{k-1}  (X_{k-2} - X_{k+1} ) - X_k + F - h*c/b * sum Y_j\]
\[ d_t Y_j = - c b Y_{j+1}  (Y_{j+2} - X_{j-1} ) - c Y_j + h*c/b * X_int(j-1)/J\]

See reference for further details.

 Callable functions: 
   integ and initialization

 Author Manuel Pulido
 Reference:

Pulido M., G. Scheffler, J. Ruiz, M. Lucini and P. Tandeo, 2016: Estimation of the functional form of subgrid-scale schemes using ensemble-based data assimilation: a simple model experiment. Q. J.  Roy. Meteorol. Soc.,  142, 2974-2984.

http://doi.org/10.1002/qj.2879

Feel free to cite it if the code was helpful
"""
import os
import numpy as np
import numpy.random as rnd
class M:
    def __init__(self,
                 F=8,nx=40, #model parameters
                 dtcy=0.05,kt=5, #integration parameters
                 nem = 100,    #number of ensemble members
                 # ny=0 1scl ny>0 L96-2scl
                 ny = 0, c=10, b=10, h=1, # L96 two scale
                 expdir='./', lcliminens=0):
        " define Model 96 parameters "
        
        self.F=F             # forcing
        self.nx=nx           # number of variables
        self.ny=ny           # ny=0 1scl ny>0 number of small-scale variables for L96-2scl
        
        self.kt=kt           # number of integrations in a cycle
        self.dt=dtcy/self.kt # integration time step
        self.dtcy=dtcy       # cycle length
        
        self.nem=nem         # number of ensemble members (for initialization)
        
        self.expdir=expdir   # directory for the data
        self.lcliminens=lcliminens # choosing the initial ensemble
                                   # 0 from climotology, 1 around truth, 2 from file

        #indices for model scheme
        self.indm2=list(range(-2,nx-2))
        self.indm1=list(range(-1,nx-1))
        self.indp1=list(np.arange(1,nx+1)%nx)

        if ny == 0:
            self.__mdl=self.__mdl_ls # 1-scale L96
        else: # two scale
             # Default ny=32 nx=8, F=18
            self.__mdl=self.__mdl_2scl
            self.indm1y=list(range(-1,ny-1))
            self.indp1y=list(np.arange(1,ny+1)%ny)
            self.indp2y=list(np.arange(2,ny+2)%ny)
            self.indx2y=list(np.array(np.arange(ny)/(ny//nx),dtype=int))
            
            self.Fin = F
            
            self.c=c             # c,b,h small-scale parameters
            self.a=c*b           # 
            self.hint=h*c/b
        
#----------------------------------------------------------

    def integ(self,xold):
        " Integrate L96 eq using RK4. xold[nx] xold[nx,nem]"

        x=np.copy(xold)
        for it in range(self.kt):
            x = self.__rk4 (x)

        return x
    __call__=integ
    
    #----------------------------------------------------------
    
    def __mdl_2scl(self,x):
        " Two scale L96 model equations"
        xls=x[:self.nx,...]
        xss=x[self.nx:,...]

        idx=(self.nx,self.ny//self.nx)
        if (xss.ndim == 2): idx=idx+(-1,)  #did not find a better way          
        sumxss=(xss[:,...].reshape(idx)).sum(1)
        
        self.F=self.Fin - self.hint  *  sumxss        
        dxls = self.__mdl_ls(xls)
        
        self.Fls = self.hint * xls
        dxss = self.__mdl_ss(xss)
        
        return np.concatenate((dxls,dxss))

    #----------------------------------------------------------
    
    def __mdl_ls(self,x):
        " L96 model equations"
        dx = (x[self.indp1] - x[self.indm2]) * x[self.indm1] - x + self.F
        return dx * self.dt 
    
    #----------------------------------------------------------
    
    def __mdl_ss(self,y):
        " L96 small-scale model equations"
        dy = (-self.a * (y[self.indp2y] - y[self.indm1y]) * y[self.indp1y] -
              self.c * y + self.Fls[self.indx2y])
        return dy * self.dt
    
    #----------------------------------------------------------
    
    def __rk4(self,xold):
        " Runge-Kutta 4th order"

        dx1 = self.__mdl( xold )
        dx2 = self.__mdl( xold + 0.5 * dx1 )
        dx3 = self.__mdl( xold + 0.5 * dx2 )
        dx4 = self.__mdl( xold + dx3 )

        x = xold +  ( dx1 + 2.0 * (dx2 + dx3) + dx4 ) / 6.0

        return x 
    
#----------------------------------------------------------------        
    def initialization(self):

        " Set initial ensamble and true initial condition "

        nx=self.nx if self.ny == 0 else self.nx+self.ny
        nem=self.nem
        
        dat_fname = self.expdir+'l96_climat.npz'
        
        if os.path.isfile(dat_fname):
            print ( 'Read sample file from: ',dat_fname )
            data = np.load(dat_fname)
            x_t  = data['x']
            nt   = len(x_t[0,:])
        else:
            # spinup of the truth      
            nt = 4*7200 # 20years
            x  = 1.0*rnd.normal(0, 1, nx) # randon initial condition
            if self.ny > 0: #small-scale IC
                xss = 0.1 *rnd.normal(0, 1, self.ny)
                x=np.concatenate((x,xss))
                
            kt  = self.kt
            self.kt=kt*nt
            x  = self.integ(x) # spinup truth
            # Integration of a climatology (uncorrelated states)
            nt  = 1000 # number of realizations 
            self.kt = 20*kt # uncorrelated states
            x_t = np.zeros((nx,nt))
            x_t[:,0]=x
            for it in range(1,nt):
                x_t[:,it] = self.integ(x_t[:,it-1]) # trajectory (climatology)\
            self.kt = kt
            np.savez(dat_fname,x=x_t)# save climatology

        # reset random seed
        rnd.seed(15) #to choose (from the sample) always the same xt (after clim)
        if (self.lcliminens==1):
            # climatology
            print ( 'Take initial ensemble from climatology' )
            idx = np.random.choice(nt,nem+1,replace=True)            
            X0 = x_t[:,idx[:nem]]
            xt0 = x_t[:,idx[nem]]
 
        elif(self.lcliminens==0): #based on the truthxt[0]-xt_t[0,0]xt[0]-xt_t[0,0]
            print ( 'Take initial ensemble randomly around true state' )
            idx = np.random.choice(nt,1,replace=True)            
            xt0 = x_t[:,idx[0]] # or a random realization of the climatology
            wrk = rnd.normal(0, 1, [nx,self.nem]) 
            X0 = np.reshape(xt0,(nx,1)) +0.5*wrk #should be climatological covariance
            
        elif(self.lcliminens==2): #read from file (case study experiment)
            dat_fname = self.expdir+'l96-X0.npz' # should go as input parameter (or use a std name)        
            if os.path.isfile(dat_fname):
                print ( 'Read initial sample file from: ',dat_fname )
                data = np.load(dat_fname)
                X0 = data['X0']
                xt0 = data['xt0']
            else:
                print ( 'Initial sample file ',dat_fname,' does not exist')
        
        return X0,xt0

if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    #L96= M(F=10) # 1scl
    
    L96= M(F=16,ny=256,nx=8)# 2scl
    X0,xt0=L96.initialization()
    nt=100

    for it in range(1,nt):
        X0 = L96(X0) 
    
    x_t = np.zeros((xt0.shape[0],nt))
    x_t[:,0]=xt0
    for it in range(1,nt):
        x_t[:,it] = L96(x_t[:,it-1]) 

    plt.figure(figsize=(6,4))
    plt.plot(x_t[0,:])
    plt.plot(x_t[1,:])
    plt.plot(x_t[2,:])
    #plt.savefig('./fig.png')
    plt.show()
