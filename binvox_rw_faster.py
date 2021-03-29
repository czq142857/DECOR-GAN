import numpy as np
import cutils
import struct

#this file includes *cutils* that uses Cython to speed up writing
#the original binvox-rw.py can be found here: https://github.com/dimatura/binvox-rw-py


class Voxels(object):
    """ Holds a binvox model.
    data is a three-dimensional numpy boolean array (dense representation)
    dims, translate and scale are the model metadata.
    dims are the voxel dimensions, e.g. [32, 32, 32] for a 32x32x32 model.
    scale and translate relate the voxels to the original model coordinates.
    """

    def __init__(self, data, dims, translate, scale, axis_order):
        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale
        assert (axis_order in ('xzy', 'xyz'))
        self.axis_order = axis_order
        

def read_header(fp):
    """ Read binvox header. Mostly meant for internal use.
    """
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    return dims, translate, scale

def read_as_3d_array(fp, fix_coords=False):
    """ Read binary binvox format as array.
    Returns the model with accompanying metadata.
    Voxels are stored in a three-dimensional numpy array, which is simple and
    direct, but may use a lot of memory for large models. (Storage requirements
    are 8*(d^3) bytes, where d is the dimensions of the binvox model. Numpy
    boolean arrays use a byte per element).
    Doesn't do any checks on input except for the '#binvox' line.
    """
    dims, translate, scale = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    # if just using reshape() on the raw data:
    # indexing the array as array[i,j,k], the indices map into the
    # coords as:
    # i -> x
    # j -> z
    # k -> y
    # if fix_coords is true, then data is rearranged so that
    # mapping is
    # i -> x
    # j -> y
    # k -> z
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(np.bool)
    data = data.reshape(dims)
    if fix_coords:
        # xzy to xyz TODO the right thing
        data = np.ascontiguousarray(np.transpose(data, (0, 2, 1)))
        axis_order = 'xyz'
    else:
        axis_order = 'xzy'
    return Voxels(data, dims, translate, scale, axis_order)

def bwrite(fp,s):
    fp.write(s.encode())

def write_pair(fp,state, ctr):
    fp.write(struct.pack('B',state))
    fp.write(struct.pack('B',ctr))
    
def write_voxel(voxel_model, filename):
    with open(filename, 'wb') as fp:
        buffer_size = 256*256*32 #change the buffer size if the input voxel is large
        state_ctr = np.zeros([buffer_size,2], np.int32)
        dimx,dimy,dimz = voxel_model.shape
        if voxel_model.dtype != np.uint8:
            voxel_model = voxel_model.astype(np.uint8)
        if not voxel_model.flags['C_CONTIGUOUS']:
            voxel_model = np.ascontiguousarray(voxel_model, dtype=np.uint8)

        cutils.get_state_ctr(voxel_model,state_ctr)

        bwrite(fp,'#binvox 1\n')
        bwrite(fp,'dim '+' '.join(map(str, [dimx,dimy,dimz]))+'\n')
        bwrite(fp,'translate '+' '.join(map(str, [0, 0, 0]))+'\n')
        bwrite(fp,'scale '+str(1)+'\n')
        bwrite(fp,'data\n')

        c = 0
        while True:
            write_pair(fp, state_ctr[c,0], state_ctr[c,1])
            c += 1
            if state_ctr[c,0]==2: break

