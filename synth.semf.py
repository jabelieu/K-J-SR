#******************************************************************************
#
# PROGRAM NAME : synth.semf -> Synthetic SEMF
#
# AUTHOR : Josh Belieu <fletch>
#
# DATE CREATED : 15.12.24
#
# PURPOSE : Generate data pertaining to the predicted binding energy per
#           nucleon via the SEMF. The binding energies of "stable" nuclei (the
#           highest BE for given A) are saved and eventually passed to the
#           regressor.
#
#******************************************************************************

#==============================================================================
"                               Begin Program                                 "
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
"                             Imported Libraries                              "
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
"                             Program Parameters                              "
#------------------------------------------------------------------------------

min_a = 1 # lower limit of number of nucleons. strange data for min_a < 3.
max_a = 300 # upper limit of number of nucleons (inclusive).
a_step = 1 # step size between adjacent A.

pad = 8 # pad setting for fstring output. this controls the number of white
        # spaces between the printed output.

#%%%%%%%%%
# 0 = off
# 1 = on 
# %%%%%%%%

input_file_flag = 1 # do you want to use an input file for
                    # the settings below?

if input_file_flag == 0 :

    plot_flag            = 1 # flag for generating plot. 0 = no plot,
                             # 1 = generate plot.
    plot_save_flag       = 1 # flag for saving plot. 0 = don't save,
                             # 1 = save plot.
    data_table_save_flag = 1 # would you like to save a table of data?
                             # 0 = no, 1 = yes. Saves as csv using pandas.
    print_flag           = 0 # would you like a table of data printed
                             # to terminal? 0 = no, 1 = yes.

else :

    input_file = 'semf.settings.inp'

    flags = np.loadtxt( input_file , dtype = int )

    plot_flag            = flags [ 0 ] # flag for generating plot. 0 = no plot,
                                       # 1 = generate plot.
    plot_save_flag       = flags [ 1 ] # flag for saving plot. 0 = don't save,
                                       # 1 = save plot.
    data_table_save_flag = flags [ 2 ] # would you like to save a table of data?
                                       # 0 = no, 1 = yes.
    print_flag           = flags [ 3 ] # would you like a table of data printed
                                       # to terminal? 0 = no, 1 = yes.

plot_file_name = 'be_per_a.png' # name of file for plot.
data_file_name = 'synth.semf.dat' # name of data file to save synthetic data.

#------------------------------------------------------------------------------
"                              SEMF Coefficients                              "
#------------------------------------------------------------------------------

a_v = 15.67 # volume coeff.

a_s = 17.23 # surface coeff.

a_c = 0.75 # coulomb coeff.

a_a = 23. # asymmetry coeff.

#------------------------------------------------------------------------------
"                                semf Function                                "
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
# PUPOSE : Given the number of nucleons return the most stable binding energy
#          divided by the given bnumber of nucleons. 
# 
# "most stable" is determined by the largest binding energy for the given
# number of nucleons and all possible numbers of protons, z, adhering to a=n+z
# where n is the number of neutrons. Natuarlly, a > n and z but not n+z.
#
# INPUT : 
# -----
#
# a : Integer. The number of nucleons.
#
# OUTPUTS : 
# -------
#
# bl/a : Array. An array of binding energies per nucleon.
#
# zl : Array. A list of all possible values of z given the value of a.
#------------------------------------------------------------------------------

def semf ( a ) :

    #
    ## generate a list from 0 to a in integer steps. this ensures z =< a.
    #

    zl = np.arange ( 0, a + 1 , 1 )

    #
    ## list of binding energies.
    #

    bl = []

    #
    ## calculate binding energies for given a and all generated z.
    #

    for z in zl :

        #
        ## for the last term in semf, we have special
        ## values for each set of a and z.
        #

        if a % 2 != 0 : # a is odd.

            a_p = 0.

        elif a % 2 == 0 and z % 2 == 0 : # a and z are even.

            a_p = 12.

        else : # a is even and z is odd.

            a_p = -12.

        #
        ## SEMF
        #

        b = a_v * a - a_s * a ** ( 2 / 3 ) - a_c * z * ( z - 1 ) / a ** ( 1 / 3 ) - \
            a_a * ( a - 2 * z ) ** 2 / a + a_p / a ** ( 1 / 2 )
        
        #
        ## append calculated binding energy to concomitant list.
        #

        bl.append ( b )
    
    return bl / a , zl

#------------------------------------------------------------------------------
"                              Main Body of Code                              "
#------------------------------------------------------------------------------

al = np.arange ( min_a , max_a + 1 , a_step ) # list of all nucleon
                                              # number to model.

fbl = [] # Final Binding energy List.
fzl = [] # Final Z List. a list containing the total number of protons.

#
## for each A in the al list, generate the most stable binding energy and
## report the energy and associated proton number.
#

for a in al :

    b_per_a , lzl = semf ( a )

    b_max = np.max ( b_per_a )

    fbl.append ( b_max )

    fzl.append ( lzl [ list(b_per_a).index ( np.max ( b_per_a ) ) ] )

#
## print a table of A, Z, and BE/A.
#

if print_flag == 1 :

    pio = f"|{"A":^{pad}}|{"Z":^{pad}}|{"BE/A":^{pad}}|"

    print ( "=" * len ( pio ) )
    print ( pio )
    print ( "|" + "-" * pad + "|" + "-" * pad + "|" + "-" * pad + "|" )

    for i in range ( len ( fbl ) ) :

        print ( f"|{str(al[i]).zfill(3):^{pad}}|{str(fzl[i]).zfill(3):^{pad}}|{fbl[i]:^{pad}.2f}|" )

    print ( "=" * len ( pio ) )

#
## write a csv containing A, Z, BE/A.
#

if data_table_save_flag == 1 :

    data = { 'A' : al , 'Z' : fzl , 'BE/A' : fbl }

    df = pd.DataFrame ( data )

    df.to_csv ( data_file_name + '.csv' , index = False )

#
## plot the synthesized data.
#

if plot_flag == 1 :

    plt.scatter ( al , fbl )

    plt.grid ( ls = '--' , alpha = 0.6 )

    plt.xlabel ( 'Number of Nucleons, A [ Nucleon ]' )
    plt.ylabel ( 'Binding Energy per Nucleon, BE/A [ MeV / Nucleon ]' )
    plt.title ( 'Binding Energy per Nucleon of Stable Odd Nuclei' )

    if plot_save_flag == 1 :

        plt.savefig( plot_file_name )

    plt.show()

#------------------------------------------------------------------------------
"                               End Program                                   "
#==============================================================================