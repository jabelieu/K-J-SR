#-------------------------------------------------------------------------------
'                             IMPORTED LIBRARIES                               '
#-------------------------------------------------------------------------------

import pysr as srp
import pandas as pd
import numpy as np
import datetime

#-------------------------------------------------------------------------------
'                             SCRIPT PARAMETERS                               '
#-------------------------------------------------------------------------------
'''
PySR instance parameters:

niterations
Number of iterations of the algorithm to run. The best equations are printed and
migrate between populations at the end of each iteration.
Default: 100

populations
Number of populations running.
Default: 31

population_size
Number of individuals in each population.
Default: 27

ncycles_per_iteration
Number of total mutations to run, per 10 samples of the population, per
iteration.
Default: 380

batching : a bool to reduce data size, necessary for large datasets.

'''

read_file = 'data_file_names.txt'

with open ( read_file , 'r' ) as f :

    first_line = f.readline().split()

    input_file = first_line[1]
    output_file = first_line[3]


now = datetime.datetime.now()
formatted_now = now.strftime("%d_%m_%Y_%H_%M_%S")

opl = [ "*" , '+' , '-' , '/' ] # basic arithmetic operators.

uopl= [ "square" , "cbrt" , 'neg' , 'cube',
        'sqrt','inv','exp','log' ] # single input math operations.

variable_names = [ 'p' , 't' , 'Dp' ]

niterations = int ( 1e2 ) 
populations = 31
population_size = 27
ncycles_per_iteration = 380
batching = True
run_id = formatted_now

print('Loading data from files:')
print(f'{input_file} and {output_file}')
print(f'Results will be saved to outputs/{run_id}')

#-------------------------------------------------------------------------------
'                                LOADING DATA                                  '
#-------------------------------------------------------------------------------

inp_df = pd.read_csv(input_file,skiprows=0,sep=' ')
with open(input_file, 'r') as file:
    first_line = file.readline()
    first_line.replace('# ','')
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
                            ncycles_per_iteration = ncycles_per_iteration , 
                            run_id = run_id )

model.fit ( fl , dlocal0 , variable_names = variable_names )