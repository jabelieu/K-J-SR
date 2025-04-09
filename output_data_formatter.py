import numpy as np

input_file = 'dlocal_48X48Y48Z.dat'
output_file = 'dlocal_48X48Y48Z.txt'

print('Fomatting output file...')

dim = 48


                                                      # V spin
out = np.fromfile(input_file, dtype=np.float64).reshape(2,2,dim,dim,dim)
                                                        # ^ isospin


#out_su_0 = out[0,0,:,:,:] # Spin Up, nucleon 0
#out_sd_0 = out[1,0,:,:,:] # Spin Down, nucleon 0

#out_su_1 = out[0,1,:,:,:] # Spin Up, nucleon 1
#out_sd_1 = out[1,1,:,:,:] # Spin Down, nucleon 1

output = np.zeros((dim**3, 4))

for ix in range(dim):
    for iy in range(dim):
        for iz in range(dim):

            output[ix*dim*dim + iy*dim + iz, 0] = out[0,0, iz, iy, ix]
            output[ix*dim*dim + iy*dim + iz, 1] = out[0,1, iz, iy, ix]

            output[ix*dim*dim + iy*dim + iz, 2] = out[1,0, iz, iy, ix]
            output[ix*dim*dim + iy*dim + iz, 3] = out[1,1, iz, iy, ix]

col_names = ['Spin Up 0', 'Spin Down 0', 'Spin Up 1', 'Spin Down 1']

np.savetxt ( output_file , output , header = ' '.join(col_names) ) #, fmt = '%1.4e' )

print('Job Done!')