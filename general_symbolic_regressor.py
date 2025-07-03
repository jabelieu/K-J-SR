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

complexity_of_constants
Complexity of constants.
Default: 1

run_id
A unique identifier for the run. Will be generated using the current date and time if not provided.
Default: None
'''

# feature_file = 'j3_quants.txt'
# target_file = 'fort.20'

nuclei_list = ['ca40','ca44', 'ca48','sn100','sn132','pb208', 'pb266']

custom_run_tag = 'pc_7nucl'

now = datetime.datetime.now()
formatted_now = now.strftime("%d_%m_%Y_%H_%M_%S")

opl = [ "*" , '+' , '-' , '/' ] # basic arithmetic operators.

uopl= [ "square" , "cbrt" , 'neg' , 'cube', 'sqrt','inv']

variable_names = [ 'rp','rn','tp','tn','ssp','ssn','sp','sn' ]

niterations = int ( 2e4 ) 
populations = 501
population_size = int ( 1e2 )
ncycles_per_iteration = int ( 1e3 )
complexity_of_constants = int ( 2e1 )
batching = False
run_id = custom_run_tag + '_' + formatted_now


# print('Loading data from file(s) :')
# print(f'{feature_file}')
# print(f'Results will be saved to outputs/{run_id}\n')

# #
# ## Bookkeeping Scheme
# #

# with open ( 'targ_path.txt' , 'w' ) as f :
#     f.write ( f'outputs/{run_id}' )

# with open ( 'sr_parameters.txt' , 'w' ) as f :
#     f.write ( f'input_file: {feature_file}\n' )
#     f.write ( f'output_path: {run_id}\n' )
#     f.write ( f'niterations: {niterations}\n' )
#     f.write ( f'populations: {populations}\n' )
#     f.write ( f'population_size: {population_size}\n' )
#     f.write ( f'ncycles_per_iteration: {ncycles_per_iteration}\n' )
#     f.write ( f'batching: {batching}\n' )
#     f.write ( f'run_id: {run_id}\n' )
#     f.write ( f'Operators and functions: {opl} {uopl}' )

#-------------------------------------------------------------------------------
'                                LOADING DATA                                  '
#-------------------------------------------------------------------------------

master_r = np.array([])
master_rp = np.array([])
master_rn = np.array([])
master_tp = np.array([])
master_tn = np.array([])
master_ssp = np.array([])
master_ssn = np.array([])
master_sp = np.array([])
master_sn = np.array([])

master_target_data = np.array([])

for nuclei in nuclei_list :

    feature_file = nuclei + '_j3_quants.txt'
    target_file = nuclei + '_fort.20'

    raw_feature_data = np.loadtxt ( feature_file , skiprows = 1 , dtype = float )

    '''
    r : distance in fm
    rp : rho-proton in particle / fm^3
    rn : rho-neutron
    tp : tau-proton in MeV / fm^3
    tn : tau-neutron
    ssp : divergance of spin orbit density-proton
    ssn : divergance of spin orbit density-neutron
    sp : spin orbit density-proton
    sn : spin orbit density-neutron

    rc : rho-charge
    '''

    r = raw_feature_data [:,0]
    rp = raw_feature_data [:,1]
    rn = raw_feature_data [:,2]
    tp = raw_feature_data [:,3]
    tn = raw_feature_data [:,4]
    ssp = raw_feature_data [:,5]
    ssn = raw_feature_data [:,6]
    sp = raw_feature_data [:,7]
    sn = raw_feature_data [:,8]

    # Append to master arrays
    master_r = np.append(master_r, r)
    master_rp = np.append(master_rp, rp)
    master_rn = np.append(master_rn, rn)
    master_tp = np.append(master_tp, tp)
    master_tn = np.append(master_tn, tn)
    master_ssp = np.append(master_ssp, ssp)
    master_ssn = np.append(master_ssn, ssn)
    master_sp = np.append(master_sp, sp)
    master_sn = np.append(master_sn, sn)

    with open ( target_file , 'r' ) as f :
        lines = f.readlines()

    target_data_list = []

    for i in range ( 4 , 74 + 1 ) :
        target_data_value = lines [ i ].split() [ 2 ]
        target_data_list.append ( float ( target_data_value ) )

    target_data = np.array ( target_data_list )

    master_target_data = np.append(master_target_data, target_data)

fl = np.column_stack ( ( master_rp, master_rn, master_tp, master_tn,
                        master_ssp, master_ssn, master_sp, master_sn ) ) # list of features.

#
## Custom Loss Function
#

pre_objective = """
function default_objective(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    (prediction, completion) = eval_tree_array(tree, dataset.X, options)
    @assert length(prediction) % 71 == 0 "Prediction length must be divisible by 71 (got $(length(prediction)))"

    if !completion
        return L(Inf)
    end

    function simpsons_integrate(y::Vector{Float64}, h::Float64)
        n = length(y)
        @assert isodd(n) "Simpson's rule requires an odd number of points."

        even_sum = sum(y[3:2:end-2])
        odd_sum  = sum(y[2:2:end-1])

        return (h / 3) * (y[1] + 4 * odd_sum + 2 * even_sum + y[end])
    end

    wc1 = 0.5
    wc2 = 0.5

    a = 0.0
    b = 21.0
    h = 0.3
    n = {{STAND-IN}}
    r = a:h:b

    numb_nucl = Int(length( prediction ) / n)

    for i in 0:(numb_nucl - 1)

        start_idx = i * n + 1
        end_idx = min((i + 1) * n, length(prediction))

        dens_slice = prediction[start_idx:end_idx]
        targ_slice = dataset.y[start_idx:end_idx]

        pred_atomic_number = 4*pi*simpsons_integrate(dens_slice.* r .^ 2, h)
        targ_atomic_number = 4*pi*simpsons_integrate(targ_slice.* r .^ 2, h)

        N = round(targ_atomic_number)/ pred_atomic_number
        prediction[start_idx:end_idx] .*= N
        new_z = 4*pi*simpsons_integrate(dens_slice.* r .^ 2, h) # <-- left off here 26.06.25
        #println("new Predicted atomic number: ", new_z)

    end

    data_diff = abs.(prediction .- dataset.y)
    master_rc_diff = Float64[]

    for i in 0:(numb_nucl - 1)

        start_idx = i * n + 1
        end_idx = min((i + 1) * n, length(prediction))

        dens_slice = prediction[start_idx:end_idx]
        targ_slice = dataset.y[start_idx:end_idx]
        # r_slice = r[1:(end_idx - start_idx + 1)]
        # diff_slice = data_diff[start_idx:end_idx]

        # Simpson's rule: ∫f(x) dx ≈ h/3 [f₀ + 4f₁ + 2f₂ + ... + fₙ]
        dens_num = r .^ 4 .* dens_slice
        dens_denom = r .^ 2 .* dens_slice

        targ_num = r .^ 4 .* targ_slice
        targ_denom = r .^ 2 .* targ_slice

        int_dens_num = simpsons_integrate(dens_num, h)
        int_dens_denom = simpsons_integrate(dens_denom, h)
        int_targ_num = simpsons_integrate(targ_num, h)
        int_targ_denom = simpsons_integrate(targ_denom, h)

        predicted_sqcharge_radius = ( int_dens_num / int_dens_denom ) ^ 2
        target_sqcharge_radius = ( int_targ_num / int_targ_denom ) ^ 2
        # push!(master_predicted_sqcharge_radius, predicted_sqcharge_radius)

        rc2_diff = abs(  predicted_sqcharge_radius - target_sqcharge_radius )/ target_sqcharge_radius
        push!(master_rc_diff, rc2_diff)

        pred_atomic_number = 4*pi*simpsons_integrate(dens_slice.* r .^ 2, h)
        targ_atomic_number = 4*pi*simpsons_integrate(targ_slice.* r .^ 2, h)
        N = round(targ_atomic_number)/ pred_atomic_number
        new_targ_z = 4*pi*N*simpsons_integrate(dens_slice.* r .^ 2, h)
        # println("check Predicted atomic number: ", pred_atomic_number)
        # println("Target atomic number: ", round(targ_atomic_number))
        # println("New target atomic number: ", new_targ_z)

        # Temper density difference
        pc_scale = abs(sum(dens_slice))
        # println("pre-scale sum: ",sum(data_diff[start_idx:end_idx]))
        # println("pre-scale area: ",simpson(collect(r),data_diff[start_idx:end_idx]))
        data_diff[start_idx:end_idx] ./= pc_scale
        # println("post-scale sum: ",sum(data_diff[start_idx:end_idx]))
        # println("post-scale area: ",simpson(collect(r),data_diff[start_idx:end_idx]))

    end

    # rc_diff = abs.(master_predicted_sqcharge_radius .- master_target_sqcharge_radius)
    rc_err = sum(master_rc_diff)
    pc_err = sum(data_diff)

    # println("rc_err: ", rc_err)
    # println("pc_err: ", pc_err)

    quad_diff = wc1 * rc_err + wc2 * pc_err

    return quad_diff
end
"""
ngrid = len ( r )
objective = pre_objective.replace("{{STAND-IN}}", str(ngrid))

#-------------------------------------------------------------------------------
'                            REGRESSOR INSTANCE                                '
#-------------------------------------------------------------------------------

print('Running PySR...\n')

model = srp.PySRRegressor ( binary_operators = opl , unary_operators = uopl ,
                            niterations = niterations ,batching = batching ,
                            populations = populations ,
                            population_size = population_size ,
                            ncycles_per_iteration = ncycles_per_iteration , 
                            run_id = run_id, loss_function= objective,
                            complexity_of_constants=complexity_of_constants)

model.fit ( fl , master_target_data , variable_names = variable_names)

'''
Find below v1 of an acceptable iteration of the custom loss function Kyle and I wrote.
We are saving it here for record purposes.
'''

# pre_objective = 
"""
function default_objective(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    (prediction, completion) = eval_tree_array(tree, dataset.X, options)
    @assert length(prediction) % 71 == 0 "Prediction length must be divisible by 71 (got $(length(prediction)))"

    if !completion
        return L(Inf)
    end

    function simpsons_integrate(y::Vector{Float64}, h::Float64)
        n = length(y)
        @assert isodd(n) "Simpson's rule requires an odd number of points."

        even_sum = sum(y[3:2:end-2])
        odd_sum  = sum(y[2:2:end-1])

        return (h / 3) * (y[1] + 4 * odd_sum + 2 * even_sum + y[end])
    end

    wc1 = 0.5
    wc2 = 0.5

    a = 0.0
    b = 21.0
    h = 0.3
    n = {{STAND-IN}}
    r = a:h:b

    numb_nucl = Int(length( prediction ) / n)

    for i in 0:(numb_nucl - 1)

        start_idx = i * n + 1
        end_idx = min((i + 1) * n, length(prediction))

        dens_slice = prediction[start_idx:end_idx]
        targ_slice = dataset.y[start_idx:end_idx]

        pred_atomic_number = 4*pi*simpsons_integrate(dens_slice.* r .^ 2, h)
        targ_atomic_number = 4*pi*simpsons_integrate(targ_slice.* r .^ 2, h)

        N = round(targ_atomic_number)/ pred_atomic_number
        prediction[start_idx:end_idx] .*= N
        new_z = 4*pi*simpsons_integrate(dens_slice.* r .^ 2, h) # <-- left off here 26.06.25
        #println("new Predicted atomic number: ", new_z)

    end

    data_diff = abs.(prediction .- dataset.y)
    master_rc_diff = Float64[]

    for i in 0:(numb_nucl - 1)

        start_idx = i * n + 1
        end_idx = min((i + 1) * n, length(prediction))

        dens_slice = prediction[start_idx:end_idx]
        targ_slice = dataset.y[start_idx:end_idx]
        # r_slice = r[1:(end_idx - start_idx + 1)]
        # diff_slice = data_diff[start_idx:end_idx]

        # Simpson's rule: ∫f(x) dx ≈ h/3 [f₀ + 4f₁ + 2f₂ + ... + fₙ]
        dens_num = r .^ 4 .* dens_slice
        dens_denom = r .^ 2 .* dens_slice

        targ_num = r .^ 4 .* targ_slice
        targ_denom = r .^ 2 .* targ_slice

        int_dens_num = simpsons_integrate(dens_num, h)
        int_dens_denom = simpsons_integrate(dens_denom, h)
        int_targ_num = simpsons_integrate(targ_num, h)
        int_targ_denom = simpsons_integrate(targ_denom, h)

        predicted_sqcharge_radius = ( int_dens_num / int_dens_denom ) ^ 2
        target_sqcharge_radius = ( int_targ_num / int_targ_denom ) ^ 2
        # push!(master_predicted_sqcharge_radius, predicted_sqcharge_radius)

        rc2_diff = abs(  predicted_sqcharge_radius - target_sqcharge_radius )/ target_sqcharge_radius
        push!(master_rc_diff, rc2_diff)

        pred_atomic_number = 4*pi*simpsons_integrate(dens_slice.* r .^ 2, h)
        targ_atomic_number = 4*pi*simpsons_integrate(targ_slice.* r .^ 2, h)
        N = round(targ_atomic_number)/ pred_atomic_number
        new_targ_z = 4*pi*N*simpsons_integrate(dens_slice.* r .^ 2, h)
        # println("check Predicted atomic number: ", pred_atomic_number)
        # println("Target atomic number: ", round(targ_atomic_number))
        # println("New target atomic number: ", new_targ_z)

        # Temper density difference
        pc_scale = abs(sum(dens_slice))
        # println("pre-scale sum: ",sum(data_diff[start_idx:end_idx]))
        # println("pre-scale area: ",simpson(collect(r),data_diff[start_idx:end_idx]))
        data_diff[start_idx:end_idx] ./= pc_scale
        # println("post-scale sum: ",sum(data_diff[start_idx:end_idx]))
        # println("post-scale area: ",simpson(collect(r),data_diff[start_idx:end_idx]))

    end

    # rc_diff = abs.(master_predicted_sqcharge_radius .- master_target_sqcharge_radius)
    rc_err = sum(master_rc_diff)
    pc_err = sum(data_diff)

    # println("rc_err: ", rc_err)
    # println("pc_err: ", pc_err)

    quad_diff = wc1 * rc_err + wc2 * pc_err

    return quad_diff
end
"""