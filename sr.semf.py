#******************************************************************************
#
# PROGRAM NAME : sr.semf.py -> "SR SEMF"
#
# AUTHOR : Josh Belieu <fletch>
#
# DATE CREATED : 18.12.24
#
# PURPOSE : Recreate Semi Empirical Mass Formula from synthetc data using 
#           symbolic regression.
#
#******************************************************************************

#==============================================================================
"                               Begin Program                                 "
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
"                             Imported Libraries                              "
#------------------------------------------------------------------------------

import pysr as srp
import pandas as pd
import numpy as np

#------------------------------------------------------------------------------
"                               Loading Data                                  "
#------------------------------------------------------------------------------

df = pd.read_csv ( 'synth.semf.dat.csv' )

al = df [ 'A' ] # nucleon number.
zl = df [ 'Z' ] # proton number.
bel = df [ 'BE/A' ] # binding energy per nucleon
                                 # of stable nuclei.

fl = np.column_stack ( ( al , zl ) ) # list of features.

#------------------------------------------------------------------------------
"                           Operator Definitions                              "
#------------------------------------------------------------------------------

opl = [ "+" , "-" , "*" , "/" ] # basic arithmetic operators.

uopl= [ "square" , 
       "cbrt" , 
       "two_third(x)=x >= 0 ? x^(2//3) : typeof(x)(NaN)" ] # single input math operations.

esml = { "one_third" : lambda x : x ** ( 1/3 ) ,
         "two_third" : lambda x : x ** ( 2/3 ) } # extra sympy mappings for 
                                                 # built-in functionality.

fln = [ 'A' , 'Z' ] # feature list names.

#
## here we establish the instance of the symbolic regressor and tell it what
## operators we would like it to consider. unary_operators are operators that
## only take a single input. Custom operators can go there.
#

model = srp.PySRRegressor ( binary_operators = opl ,
                            unary_operators = uopl ,
                            extra_sympy_mappings = esml )

#
## here we ask the regressor to fit the generated data to our generating
## function.
#

model.fit ( fl , bel , variable_names = fln )

print ( model.sympy() )

#------------------------------------------------------------------------------
"                               End Program                                   "
#==============================================================================