import numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def get_state_ctr(char[:, :, ::1] img, int[:, ::1] state_ctr):
    cdef int dimx,dimy,dimz
    cdef int state,ctr
    cdef int p,i,j,k

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]
    state = 0
    ctr = 0
    p = 0

    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                if img[i,j,k]==state:
                    ctr += 1
                    if ctr==255:
                        state_ctr[p,0] = state
                        state_ctr[p,1] = ctr
                        p += 1
                        ctr = 0
                else:
                    if ctr>0:
                        state_ctr[p,0] = state
                        state_ctr[p,1] = ctr
                        p += 1
                    state = img[i,j,k]
                    ctr = 1

    if ctr > 0:
        state_ctr[p,0] = state
        state_ctr[p,1] = ctr
        p += 1

    state_ctr[p,0] = 2





@cython.boundscheck(False)
@cython.wraparound(False)
def get_patches(char[:, :, ::1] img, char[:, :, :, ::1] patches, int patch_size):
    cdef int dimx,dimy,dimz
    #cdef int buffer_size
    cdef int margin_size
    cdef int p,i,j,k

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]
    #buffer_size = patches.shape[0]
    margin_size = (patch_size-2)//2
    p = 0

    for i in range(margin_size,dimx-margin_size-1):
        for j in range(margin_size,dimy-margin_size-1):
            for k in range(margin_size,dimz-margin_size-1):
                if (img[i,j,k]==0 or img[i,j,k+1]==0 or img[i,j+1,k]==0 or img[i,j+1,k+1]==0 or img[i+1,j,k]==0 or img[i+1,j,k+1]==0 or img[i+1,j+1,k]==0 or img[i+1,j+1,k+1]==0) and (img[i,j,k]==1 or img[i,j,k+1]==1 or img[i,j+1,k]==1 or img[i,j+1,k+1]==1 or img[i+1,j,k]==1 or img[i+1,j,k+1]==1 or img[i+1,j+1,k]==1 or img[i+1,j+1,k+1]==1):
                    patches[p] = img[i-margin_size:i+margin_size+2,j-margin_size:j+margin_size+2,k-margin_size:k+margin_size+2]
                    p += 1

    return p





@cython.boundscheck(False)
@cython.wraparound(False)
def get_patches_edge_dilated(char[:, :, ::1] img, char[:, :, ::1] edge, char[:, :, ::1] dilated, char[:, :, :, ::1] patches,  char[:, :, :, ::1] patches_edge, char[:, :, :, ::1] patches_dilated, int patch_size):
    cdef int dimx,dimy,dimz
    cdef int margin_size
    cdef int p,i,j,k

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]
    margin_size = (patch_size-2)//2
    p = 0

    for i in range(margin_size,dimx-margin_size-1):
        for j in range(margin_size,dimy-margin_size-1):
            for k in range(margin_size,dimz-margin_size-1):
                if (img[i,j,k]==0 or img[i,j,k+1]==0 or img[i,j+1,k]==0 or img[i,j+1,k+1]==0 or img[i+1,j,k]==0 or img[i+1,j,k+1]==0 or img[i+1,j+1,k]==0 or img[i+1,j+1,k+1]==0) and (img[i,j,k]==1 or img[i,j,k+1]==1 or img[i,j+1,k]==1 or img[i,j+1,k+1]==1 or img[i+1,j,k]==1 or img[i+1,j,k+1]==1 or img[i+1,j+1,k]==1 or img[i+1,j+1,k+1]==1):
                    patches[p] = img[i-margin_size:i+margin_size+2,j-margin_size:j+margin_size+2,k-margin_size:k+margin_size+2]
                    patches_edge[p] = edge[i-margin_size:i+margin_size+2,j-margin_size:j+margin_size+2,k-margin_size:k+margin_size+2]
                    patches_dilated[p] = dilated[i-margin_size:i+margin_size+2,j-margin_size:j+margin_size+2,k-margin_size:k+margin_size+2]
                    p += 1

    return p





@cython.boundscheck(False)
@cython.wraparound(False)
def get_patches_sparse(char[:, :, ::1] img, int[:, :, ::1] patches, int patch_size):
    cdef int dimx,dimy,dimz
    #cdef int buffer_size
    cdef int margin_size
    cdef int p,i,j,k,x,y,z,sx,sy,sz,c

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]
    #buffer_size = patches.shape[0]
    margin_size = (patch_size-2)//2
    p = 0

    for i in range(margin_size,dimx-margin_size-1):
        for j in range(margin_size,dimy-margin_size-1):
            for k in range(margin_size,dimz-margin_size-1):
                if (img[i,j,k]==0 or img[i,j,k+1]==0 or img[i,j+1,k]==0 or img[i,j+1,k+1]==0 or img[i+1,j,k]==0 or img[i+1,j,k+1]==0 or img[i+1,j+1,k]==0 or img[i+1,j+1,k+1]==0) and (img[i,j,k]==1 or img[i,j,k+1]==1 or img[i,j+1,k]==1 or img[i,j+1,k+1]==1 or img[i+1,j,k]==1 or img[i+1,j,k+1]==1 or img[i+1,j+1,k]==1 or img[i+1,j+1,k+1]==1):
                    #sparse
                    sx = i-margin_size
                    sy = j-margin_size
                    sz = k-margin_size
                    c = 1
                    for x in range(patch_size):
                        for y in range(patch_size):
                            for z in range(patch_size):
                                if img[sx+x,sy+y,sz+z]==1:
                                    patches[p,c,0] = x
                                    patches[p,c,1] = y
                                    patches[p,c,2] = z
                                    c += 1
                    patches[p,0,0] = c-1
                    p += 1

    return p






@cython.boundscheck(False)
@cython.wraparound(False)
def eval_match_CD(int[:, ::1] img, int[:, :, ::1] patches, float threshold):
    cdef int dict_size
    cdef int p,i,j,x,y,z,dx,dy,dz,cd,cd_acc,cd_tmp,vnum1,vnum2,threshold_int1
    cdef int [:, ::1] patch

    dict_size = patches.shape[0]
    vnum1 = img[0,0]
    threshold_int1 = int(threshold*vnum1)
    
    for p in range(dict_size):
        patch = patches[p]
        vnum2 = patch[0,0]
        cd = 0

        for i in range(1,vnum1+1):
            cd_acc = 32768
            x = img[i,0]
            y = img[i,1]
            z = img[i,2]
            for j in range(1,vnum2+1):
                dx = patch[j,0] - x
                dy = patch[j,1] - y
                dz = patch[j,2] - z
                cd_tmp = dx*dx + dy*dy + dz*dz
                if cd_tmp<cd_acc:
                    cd_acc = cd_tmp
            cd += cd_acc
            if cd > threshold_int1: break

        if cd<=threshold_int1:
            return 1

    return 0


