#-------------------------------------------------------------------------------
'                             IMPORTED LIBRARIES                               '
#-------------------------------------------------------------------------------

import pysr as srp
import pandas as pd
import numpy as np

#-------------------------------------------------------------------------------
'                             SCRIPT PARAMETERS                               '
#-------------------------------------------------------------------------------
'''
PySR instance parameters:

niterations
Number of iterations of the algorithm to run. The best equations are printed and migrate between populations at the end of each iteration.
Default: 100

populations
Number of populations running.
Default: 31

population_size
Number of individuals in each population.
Default: 27

ncycles_per_iteration
Number of total mutations to run, per 10 samples of the population, per iteration.
Default: 380

batching : a bool to reduce data size, necessary for large datasets.

'''


input_file = 'rlambda_48X48Y48Z.txt'
output_file = 'dlocal_48X48Y48Z.txt'

opl = [ "*" , '+' , '-' , '/' ] # basic arithmetic operators.

uopl= [ "square" , "cbrt" , 'neg' , 'cube','sqrt','inv','exp','log' ] # single input math operations.

variable_names = [ 'p' , 't' , 'Dp' ]

niterations = int ( 1e2 ) 
populations = 31
population_size = 27
ncycles_per_iteration = 380
batching = True

#-------------------------------------------------------------------------------
'                                LOADING DATA                                  '
#-------------------------------------------------------------------------------

inp_df = pd.read_csv(input_file,skiprows=0,sep=' ')
with open(input_file, 'r') as file:
    first_line = file.readline()
inp_df.columns = first_line.split()

out_df = pd.read_csv(output_file,skiprows=0,sep=' ')
with open(output_file, 'r') as file:
    first_line = file.readline()
    first_line.replace('# ','')
out_df.columns = first_line.split()

#-------------------------------------------------------------------------------
'                              PREPARING DATA                                  '
#-------------------------------------------------------------------------------

rho0 = inp_df['rho0'].to_numpy()

tau0 = inp_df['tau0'].to_numpy()

drhosx0 = inp_df['drhosx0'].to_numpy()
drhosy0 = inp_df['drhosy'].to_numpy()
drhosz0 = inp_df['drhosz'].to_numpy()

drhos2 = drhosx0**2 + drhosy0**2 + drhosz0**2

fl = np.column_stack ( ( rho0 , tau0 , drhos2 ) ) # list of features.

dlocal0 = out_df['Spin_Up_0'].to_numpy()

#-------------------------------------------------------------------------------
'                            REGRESSOR INSTANCE                                '
#-------------------------------------------------------------------------------

print('Running PySR...')

model = srp.PySRRegressor ( binary_operators = opl , unary_operators = uopl ,
                            niterations = niterations ,batching = batching ,
                            populations = populations ,
                            population_size = population_size ,
                            ncycles_per_iteration = ncycles_per_iteration )

model.fit ( fl , dlocal0 , variable_names = variable_names )