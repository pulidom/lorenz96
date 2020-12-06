# Lorenz 1996 dynamical system 

- One and two scale Lorenz-96 model

- Integrate model equations with Runge-Kuta 4th order

- Works with ensemble if the input is an ensemble

### Equations

L96 system:

$$d_t X_k = - X_{k-1} (X_{k-2} - X_{k+1} ) - X_k + F$$

L96 two scale system:

$$d_t X_k = - X_{k-1}  (X_{k-2} - X_{k+1} ) - X_k + F - h*c/b * sum Y_j$$

$$d_t Y_j = - c b Y_{j+1}  (Y_{j+2} - X_{j-1} ) - c Y_j + h*c/b * X_int(j-1)/J$$

See reference for further details on L96 systems.

### Callable functions: 
   integ and initialization

 
### Reference:

Pulido M., G. Scheffler, J. Ruiz, M. Lucini and P. Tandeo, 2016: Estimation of the functional form of subgrid-scale schemes using ensemble-based data assimilation: a simple model experiment. Q. J.  Roy. Meteorol. Soc.,  142, 2974-2984.

http://doi.org/10.1002/qj.2879

Feel free to cite it if the code was helpful

*Author:* Manuel Pulido
