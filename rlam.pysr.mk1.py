import pysr as srp
import pandas as pd
import numpy as np

data_file = 'output.dat'

rho0, rho1, tau0, tau1, jx0, jx1, jy0, jy1, jz0, jz1, sodensx0, sodensx1, sodensy0, sodensy1, sodensz0, sodensz1, spindenx0, spindenx1, \
spindeny0, spindeny1, spindenz0, spindenz1, drhosx0, drhosx1, drhosy, drhosy10, drhosz, drhosz10, rlam0, rlam1 = np.loadtxt( data_file , skiprows=1 , unpack=True)

rho_tot = rho0 + rho1
tau_tot = tau0 + tau1

jx_tot = jx0 + jx1
jy_tot = jy0 + jy1
jz_tot = jz0 + jz1

j2 = jx_tot**2 + jy_tot**2 + jz_tot**2

sodensx_tot = sodensx0 + sodensx1
sodensy_tot = sodensy0 + sodensy1
sodensz_tot = sodensz0 + sodensz1

sodens2 = sodensx_tot**2 + sodensy_tot**2 + sodensz_tot**2

spindenx_tot = spindenx0 + spindenx1
spindeny_tot = spindeny0 + spindeny1
spindenz_tot = spindenz0 + spindenz1

spinden2 = spindenx_tot**2 + spindeny_tot**2 + spindenz_tot**2

drhosx_tot = drhosx0 + drhosx1
drhosy_tot = drhosy + drhosy10
drhosz_tot = drhosz + drhosz10

drhos2 = drhosx_tot**2 + drhosy_tot**2 + drhosz_tot**2

fl = np.column_stack ( ( rho_tot , tau_tot , j2 , sodens2 , spinden2 , drhos2 ) ) # list of features.

rlam_tot = rlam0 + rlam1

opl = [ "*" , '+' , '-' , '/' ] # basic arithmetic operators.

uopl= [ "square" , "cbrt" , 'neg' , 'cube','sqrt','inv','exp','log' ] # single input math operations.

reg_iter = int ( 1e4 )

model = srp.PySRRegressor ( binary_operators = opl , unary_operators = uopl , niterations = reg_iter ,batching = True )

model.fit ( fl , rlam_tot )
