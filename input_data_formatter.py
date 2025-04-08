import numpy as np

input_file = 'rlambda_48X48Y48Z.dat'
output_file = 'rlambda_48X48Y48Z.txt'

def read_binary ( input_file ) :

    """
    "[...] numpy function to convert things from fortran ordered to c ordered "

    numpy.ascontiguousarray(a, dtype=None, *, like=None)

    Return a contiguous array (ndim >= 1) in memory (C order).

    """

    intchunk = np.dtype(np.int32).itemsize
    realchunk = np.dtype(np.float64).itemsize

    ncolx = np.fromfile(input_file, dtype=np.int32, count=1, offset=0)[0]
    ncoly =  np.fromfile(input_file, dtype=np.int32, count=1, offset=(0*realchunk+1*intchunk))[0]
    ncolz =  np.fromfile(input_file, dtype=np.int32, count=1, offset=(0*realchunk+2*intchunk))[0]

    #print(ncolx, ncoly, ncolz)

    nprod = ncolx*ncoly*ncolz

    rlam = np.fromfile(input_file, dtype=np.float64, count=nprod*2, offset=(0*realchunk+3*intchunk)).reshape(2,ncolz,ncoly,ncolx)

    rho = np.fromfile(input_file, dtype=np.float64, count=nprod*2, offset=(2*nprod*realchunk+3*intchunk)).reshape(2,ncolz,ncoly,ncolx)

    tau = np.fromfile(input_file, dtype=np.float64, count=nprod*2, offset=(4*nprod*realchunk+3*intchunk)).reshape(2,ncolz,ncoly,ncolx)

    currnt = np.fromfile(input_file, dtype=np.float64, count=nprod*2*3, offset=(6*nprod*realchunk+3*intchunk)).reshape(2,3,ncolz,ncoly,ncolx)

    sodens = np.fromfile(input_file, dtype=np.float64, count=nprod*2*3, offset=(12*nprod*realchunk+3*intchunk)).reshape(2,3,ncolz,ncoly,ncolx)

    spinden = np.fromfile(input_file, dtype=np.float64, count=nprod*2*3, offset=(18*nprod*realchunk+3*intchunk)).reshape(2,3,ncolz,ncoly,ncolx)

    kinvden = np.fromfile(input_file, dtype=np.float64, count=nprod*2*3, offset=(24*nprod*realchunk+3*intchunk)).reshape(2,3,ncolz,ncoly,ncolx)

    spincur = np.fromfile(input_file, dtype=np.float64, count=nprod*2*3*3, offset=(30*nprod*realchunk+3*intchunk)).reshape(2,3,3,ncolz,ncoly,ncolx)

    drhos = np.fromfile(input_file, dtype=np.float64, count=nprod*2*3*2, offset=(48*nprod*realchunk+3*intchunk)).reshape(2,3,2,ncolz,ncoly,ncolx)

    return ncolx, ncoly, ncolz, rlam, rho, tau, currnt, sodens, spinden, kinvden, spincur, drhos

ncolx, ncoly, ncolz, rlam, rho, tau, currnt, sodens,spinden,kinvden,spincur,drhos = read_binary( input_file )

# rho0 = rho[0,:,:,:]
# rho1 = rho[1,:,:,:]

# tau0 = tau[0,:,:,:]
# tau1 = tau[1,:,:,:]

# jx = currnt[0,0,:,:,:]
# jy = currnt[0,1,:,:,:]
# jz = currnt[0,2,:,:,:]

# jx1 = currnt[1,0,:,:,:]
# jy1 = currnt[1,1,:,:,:]
# jz1 = currnt[1,2,:,:,:]

# kinvdenx = kinvden[0,0,:,:,:]
# kinvdeny = kinvden[0,1,:,:,:]
# kinvdenz = kinvden[0,2,:,:,:]

# kinvdenx1 = kinvden[1,0,:,:,:]
# kinvdeny1 = kinvden[1,1,:,:,:]
# kinvdenz1 = kinvden[1,2,:,:,:]

# sodensx = sodens[0,0,:,:,:]
# sodensy = sodens[0,1,:,:,:]
# sodensz = sodens[0,2,:,:,:]

# sodensx1 = sodens[1,0,:,:,:]
# sodensy1 = sodens[1,1,:,:,:]
# sodensz1 = sodens[1,2,:,:,:]

# spindenx = spinden[0,0,:,:,:]
# spindeny = spinden[0,1,:,:,:]
# spindenz = spinden[0,2,:,:,:]

# spindenx1 = spinden[1,0,:,:,:]
# spindeny1 = spinden[1,1,:,:,:]
# spindenz1 = spinden[1,2,:,:,:]

# drhosx = drhos[0,0,0,:,:,:]+drhos[0,0,1,:,:,:]
# drhosy = drhos[0,1,0,:,:,:]+drhos[0,1,1,:,:,:]
# drhosz = drhos[0,2,0,:,:,:]+drhos[0,2,1,:,:,:]

# drhosx1 = drhos[1,0,0,:,:,:]+drhos[1,0,1,:,:,:]
# drhosy1 = drhos[1,1,0,:,:,:]+drhos[1,1,1,:,:,:]
# drhosz1 = drhos[1,2,0,:,:,:]+drhos[1,2,1,:,:,:]

# spincurx = spincur[0,0,0,:,:,:]+spincur[0,0,1,:,:,:]+spincur[0,0,2,:,:,:]
# spincury = spincur[0,1,0,:,:,:]+spincur[0,1,1,:,:,:]+spincur[0,1,2,:,:,:]
# spincurz = spincur[0,2,0,:,:,:]+spincur[0,2,1,:,:,:]+spincur[0,2,2,:,:,:]

# spincurx1 = spincur[1,0,0,:,:,:]+spincur[1,0,1,:,:,:]+spincur[1,0,2,:,:,:]
# spincury1 = spincur[1,1,0,:,:,:]+spincur[1,1,1,:,:,:]+spincur[1,1,2,:,:,:]
# spincurz1 = spincur[1,2,0,:,:,:]+spincur[1,2,1,:,:,:]+spincur[1,2,2,:,:,:]

col_names = ['rho0','rho1','tau0','tau1','jx0','jx1','jy0','jy1','jz0','jz1',
             'sodensx0','sodensx1','sodensy0','sodensy1','sodensz0','sodensz1',
             'spindenx0','spindenx1','spindeny0','spindeny1','spindenz0','spindenz1',
             'drhosx0','drhosx1','drhosy','drhosy10','drhosz','drhosz10','rlam0','rlam1']

# col_arrays = [rho0,rho1,tau0,tau1,jx,jx1,jy,jy1,jz,jz1,kinvdenx,kinvdenx1,
#               kinvdeny,kinvdeny1,kinvdenz,kinvdenz1,sodensx,sodensx1,sodensy,sodensy1,
#               sodensz,sodensz1,spindenx,spindenx1,spindeny,spindeny1,spindenz,spindenz1,
#               drhosx,drhosx1,drhosy,drhosy1,drhosz,drhosz1,spincurx,spincurx1,
#               spincury,spincury1,spincurz,spincurz1]

output = np.zeros((ncolx*ncoly*ncolz, 30))

for ix in range(ncolx):
    for iy in range(ncoly):
        for iz in range(ncolz):

            output[ix*ncoly*ncolz + iy*ncolz + iz, 0] = rho[0, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 1] = rho[1, iz, iy, ix]

            output[ix*ncoly*ncolz + iy*ncolz + iz, 2] = tau[0, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 3] = tau[1, iz, iy, ix]

            output[ix*ncoly*ncolz + iy*ncolz + iz, 4] = currnt[0, 0, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 5] = currnt[1, 0, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 6] = currnt[0, 1, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 7] = currnt[1, 1, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 8] = currnt[0, 2, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 9] = currnt[1, 2, iz, iy, ix]

            output[ix*ncoly*ncolz + iy*ncolz + iz, 10] = sodens[0, 0, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 11] = sodens[1, 0, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 12] = sodens[0, 1, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 13] = sodens[1, 1, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 14] = sodens[0, 2, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 15] = sodens[1, 2, iz, iy, ix]

            output[ix*ncoly*ncolz + iy*ncolz + iz, 16] = spinden[0, 0, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 17] = spinden[1, 0, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 18] = spinden[0, 1, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 19] = spinden[1, 1, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 20] = spinden[0, 2, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 21] = spinden[1, 2, iz, iy, ix]

            output[ix*ncoly*ncolz + iy*ncolz + iz, 22] = drhos[0, 0, 0, iz, iy, ix] +drhos[0, 0, 1, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 23] = drhos[1, 0, 0, iz, iy, ix] +drhos[1, 0, 1, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 24] = drhos[0, 1, 0, iz, iy, ix] +drhos[0, 1, 1, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 25] = drhos[1, 1, 0, iz, iy, ix] +drhos[1, 1, 1, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 26] = drhos[0, 2, 0, iz, iy, ix] +drhos[0, 2, 1, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 27] = drhos[1, 2, 0, iz, iy, ix] +drhos[1, 2, 1, iz, iy, ix]

            output[ix*ncoly*ncolz + iy*ncolz + iz, 28] = rlam[0, iz, iy, ix]
            output[ix*ncoly*ncolz + iy*ncolz + iz, 29] = rlam[1, iz, iy, ix]
            #output[ix*ncoly*ncolz + iy*ncolz + iz, 5] = kinvden[0, 0, iz, iy, ix] + kinvden[1, 0, iz, iy, ix]
            #output[ix*ncoly*ncolz + iy*ncolz + iz, 6] = spincur[0, 0, 0, iz, iy, ix] + spincur[1, 0, 0, iz, iy, ix] # this is wrong

np.savetxt ( output_file , output , header = ' '.join(col_names) ) #, fmt = '%1.4e' )