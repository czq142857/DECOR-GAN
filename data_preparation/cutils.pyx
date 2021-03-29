import numpy as np
cimport cython


cdef int render_depth_img_size = 2560

@cython.boundscheck(False)
@cython.wraparound(False)
def floodfill(char[:, :, ::1] img, int[:, ::1] queue, int[:, ::1] state_ctr):
    cdef int dimx,dimy,dimz,max_queue_len
    cdef int pi = 0
    cdef int pj = 0
    cdef int pk = 0
    cdef int queue_start = 0
    cdef int queue_end = 1

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]
    max_queue_len = queue.shape[0]

    img[0,0,0] = 0
    queue[queue_start,0] = 0
    queue[queue_start,1] = 0
    queue[queue_start,2] = 0

    while queue_start != queue_end:
        pi = queue[queue_start,0]
        pj = queue[queue_start,1]
        pk = queue[queue_start,2]
        queue_start += 1
        if queue_start==max_queue_len:
            queue_start = 0

        pi = pi+1
        if pi<dimx and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0

        pi = pi-2
        if pi>=0 and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0

        pi = pi+1
        pj = pj+1
        if pj<dimy and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0

        pj = pj-2
        if pj>=0 and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0

        pj = pj+1
        pk = pk+1
        if pk<dimz and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0

        pk = pk-2
        if pk>=0 and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0


    cdef int state = 0
    cdef int ctr = 0
    cdef int p = 0
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0

    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                if img[i,j,k]>0:
                    img[i,j,k] = 1
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
def expand_then_shrink(char[:, :, ::1] img, char[:, :, ::1] tmp):
    cdef int dimx,dimy,dimz
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    cdef int flag = 0

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]

    #do it efficiently -> only look at inside voxels, which are far less than outside voxels

    #expand
    for i in range(1,dimx-1):
        for j in range(1,dimy-1):
            for k in range(1,dimz-1):
                if img[i,j,k] == 1:
                    tmp[i-1,j-1,k-1] = 1
                    tmp[i-1,j-1,k] = 1
                    tmp[i-1,j-1,k+1] = 1
                    tmp[i-1,j,k-1] = 1
                    tmp[i-1,j,k] = 1
                    tmp[i-1,j,k+1] = 1
                    tmp[i-1,j+1,k-1] = 1
                    tmp[i-1,j+1,k] = 1
                    tmp[i-1,j+1,k+1] = 1
                    tmp[i,j-1,k-1] = 1
                    tmp[i,j-1,k] = 1
                    tmp[i,j-1,k+1] = 1
                    tmp[i,j,k-1] = 1
                    tmp[i,j,k] = 1
                    tmp[i,j,k+1] = 1
                    tmp[i,j+1,k-1] = 1
                    tmp[i,j+1,k] = 1
                    tmp[i,j+1,k+1] = 1
                    tmp[i+1,j-1,k-1] = 1
                    tmp[i+1,j-1,k] = 1
                    tmp[i+1,j-1,k+1] = 1
                    tmp[i+1,j,k-1] = 1
                    tmp[i+1,j,k] = 1
                    tmp[i+1,j,k+1] = 1
                    tmp[i+1,j+1,k-1] = 1
                    tmp[i+1,j+1,k] = 1
                    tmp[i+1,j+1,k+1] = 1
    
    #shrink
    for i in range(1,dimx-1):
        for j in range(1,dimy-1):
            for k in range(1,dimz-1):
                if tmp[i,j,k] == 1 and img[i,j,k] == 0:
                    flag = 0
                    flag += tmp[i-1,j-1,k-1]
                    flag += tmp[i-1,j-1,k]
                    flag += tmp[i-1,j-1,k+1]
                    flag += tmp[i-1,j,k-1]
                    flag += tmp[i-1,j,k]
                    flag += tmp[i-1,j,k+1]
                    flag += tmp[i-1,j+1,k-1]
                    flag += tmp[i-1,j+1,k]
                    flag += tmp[i-1,j+1,k+1]
                    flag += tmp[i,j-1,k-1]
                    flag += tmp[i,j-1,k]
                    flag += tmp[i,j-1,k+1]
                    flag += tmp[i,j,k-1]
                    flag += tmp[i,j,k]
                    flag += tmp[i,j,k+1]
                    flag += tmp[i,j+1,k-1]
                    flag += tmp[i,j+1,k]
                    flag += tmp[i,j+1,k+1]
                    flag += tmp[i+1,j-1,k-1]
                    flag += tmp[i+1,j-1,k]
                    flag += tmp[i+1,j-1,k+1]
                    flag += tmp[i+1,j,k-1]
                    flag += tmp[i+1,j,k]
                    flag += tmp[i+1,j,k+1]
                    flag += tmp[i+1,j+1,k-1]
                    flag += tmp[i+1,j+1,k]
                    flag += tmp[i+1,j+1,k+1]

                    if flag==27:
                        img[i,j,k] = 1




@cython.boundscheck(False)
@cython.wraparound(False)
def depth_fusion_XZY(char[:, :, ::1] img, int[:, :, ::1] rendering, int[:, ::1] state_ctr):
    cdef int dimx,dimy,dimz

    cdef int hdis = render_depth_img_size//2 #half depth image size
    
    cdef int c = 0
    cdef int u = 0
    cdef int v = 0
    cdef int d = 0
    
    cdef int outside_flag = 0
    
    cdef int x = 0
    cdef int y = 0
    cdef int z = 0
    
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    
    cdef int state = 0
    cdef int ctr = 0
    cdef int p = 0

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]
    
    #--model
    # 0 - X - front
    # 1 - Z - left
    # 2 - Y - up
    #--rendering [render_depth_img_size,render_depth_img_size,17]
    # 0 - top-down from Y
    # 1,2,3,4 - from X | Z
    # 5,6,7,8 - from X&Z
    # 9,10,11,12 - from X&Y | Z&Y
    # 13,14,15,16 - from X&Y&Z
    #read my figure for details (if you can find it)
    
    #get rendering
    for x in range(dimx):
        for y in range(dimy):
            for z in range(dimz):
                if img[x,z,y]>0:
                    #z-buffering
                    
                    c = 0
                    u = x + hdis
                    v = z + hdis
                    d = -y #y must always be negative in d to render from top
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 1
                    u = y + hdis
                    v = z + hdis
                    d = x
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 2
                    u = y + hdis
                    v = z + hdis
                    d = -x
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 3
                    u = x + hdis
                    v = y + hdis
                    d = z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 4
                    u = x + hdis
                    v = y + hdis
                    d = -z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 5
                    u = y + hdis
                    v = x-z + hdis
                    d = x+z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u,v+1,c]>d: #block 2
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 2
                        rendering[u,v-1,c]=d
                        
                    c = 6
                    u = y + hdis
                    v = x+z + hdis
                    d = x-z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u,v+1,c]>d: #block 2
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 2
                        rendering[u,v-1,c]=d
                        
                    c = 7
                    u = y + hdis
                    v = -x-z + hdis
                    d = -x+z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u,v+1,c]>d: #block 2
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 2
                        rendering[u,v-1,c]=d
                        
                    c = 8
                    u = y + hdis
                    v = -x+z + hdis
                    d = -x-z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u,v+1,c]>d: #block 2
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 2
                        rendering[u,v-1,c]=d
                        
                    c = 9
                    u = z + hdis
                    v = x+y + hdis
                    d = x-y
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u,v+1,c]>d: #block 2
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 2
                        rendering[u,v-1,c]=d
                        
                    c = 10
                    u = z + hdis
                    v = -x+y + hdis
                    d = -x-y
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u,v+1,c]>d: #block 2
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 2
                        rendering[u,v-1,c]=d
                        
                    c = 11
                    u = x + hdis
                    v = z+y + hdis
                    d = z-y
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u,v+1,c]>d: #block 2
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 2
                        rendering[u,v-1,c]=d
                        
                    c = 12
                    u = x + hdis
                    v = -z+y + hdis
                    d = -z-y
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u,v+1,c]>d: #block 2
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 2
                        rendering[u,v-1,c]=d
                        
                    c = 13
                    u = x+y + hdis
                    v = -y-z + hdis
                    d = x-y+z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u+1,v,c]>d: #block 6
                        rendering[u+1,v,c]=d
                    if rendering[u-1,v,c]>d: #block 6
                        rendering[u-1,v,c]=d
                    if rendering[u,v+1,c]>d: #block 6
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 6
                        rendering[u,v-1,c]=d
                    if rendering[u+1,v-1,c]>d: #block 6
                        rendering[u+1,v-1,c]=d
                    if rendering[u-1,v+1,c]>d: #block 6
                        rendering[u-1,v+1,c]=d
                        
                    c = 14
                    u = -x+y + hdis
                    v = -y-z + hdis
                    d = -x-y+z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u+1,v,c]>d: #block 6
                        rendering[u+1,v,c]=d
                    if rendering[u-1,v,c]>d: #block 6
                        rendering[u-1,v,c]=d
                    if rendering[u,v+1,c]>d: #block 6
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 6
                        rendering[u,v-1,c]=d
                    if rendering[u+1,v-1,c]>d: #block 6
                        rendering[u+1,v-1,c]=d
                    if rendering[u-1,v+1,c]>d: #block 6
                        rendering[u-1,v+1,c]=d
                        
                    c = 15
                    u = x+y + hdis
                    v = -y+z + hdis
                    d = x-y-z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u+1,v,c]>d: #block 6
                        rendering[u+1,v,c]=d
                    if rendering[u-1,v,c]>d: #block 6
                        rendering[u-1,v,c]=d
                    if rendering[u,v+1,c]>d: #block 6
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 6
                        rendering[u,v-1,c]=d
                    if rendering[u+1,v-1,c]>d: #block 6
                        rendering[u+1,v-1,c]=d
                    if rendering[u-1,v+1,c]>d: #block 6
                        rendering[u-1,v+1,c]=d
                        
                    c = 16
                    u = -x+y + hdis
                    v = -y+z + hdis
                    d = -x-y-z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u+1,v,c]>d: #block 6
                        rendering[u+1,v,c]=d
                    if rendering[u-1,v,c]>d: #block 6
                        rendering[u-1,v,c]=d
                    if rendering[u,v+1,c]>d: #block 6
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 6
                        rendering[u,v-1,c]=d
                    if rendering[u+1,v-1,c]>d: #block 6
                        rendering[u+1,v-1,c]=d
                    if rendering[u-1,v+1,c]>d: #block 6
                        rendering[u-1,v+1,c]=d
                    
    
    
    #depth fusion
    for x in range(dimx):
        for y in range(dimy):
            for z in range(dimz):
                outside_flag = 0
                
                c = 0
                u = x + hdis
                v = z + hdis
                d = -y
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 1
                u = y + hdis
                v = z + hdis
                d = x
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 2
                u = y + hdis
                v = z + hdis
                d = -x
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 3
                u = x + hdis
                v = y + hdis
                d = z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 4
                u = x + hdis
                v = y + hdis
                d = -z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 5
                u = y + hdis
                v = x-z + hdis
                d = x+z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 6
                u = y + hdis
                v = x+z + hdis
                d = x-z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 7
                u = y + hdis
                v = -x-z + hdis
                d = -x+z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 8
                u = y + hdis
                v = -x+z + hdis
                d = -x-z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 9
                u = z + hdis
                v = x+y + hdis
                d = x-y
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 10
                u = z + hdis
                v = -x+y + hdis
                d = -x-y
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 11
                u = x + hdis
                v = z+y + hdis
                d = z-y
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 12
                u = x + hdis
                v = -z+y + hdis
                d = -z-y
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 13
                u = x+y + hdis
                v = -y-z + hdis
                d = x-y+z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 14
                u = -x+y + hdis
                v = -y-z + hdis
                d = -x-y+z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 15
                u = x+y + hdis
                v = -y+z + hdis
                d = x-y-z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 16
                u = -x+y + hdis
                v = -y+z + hdis
                d = -x-y-z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                
                if outside_flag==0:
                    img[x,z,y] = 1
    
    
    #get state_ctr
    

    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                if img[i,j,k]>0:
                    img[i,j,k] = 1
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
def depth_fusion_XZY_5views(char[:, :, ::1] img, int[:, :, ::1] rendering, int[:, ::1] state_ctr):
    cdef int dimx,dimy,dimz
    
    cdef int hdis = render_depth_img_size//2 #half depth image size
    
    cdef int c = 0
    cdef int u = 0
    cdef int v = 0
    cdef int d = 0
    
    cdef int outside_flag = 0
    
    cdef int x = 0
    cdef int y = 0
    cdef int z = 0
    
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    
    cdef int state = 0
    cdef int ctr = 0
    cdef int p = 0

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]
    
    #--model
    # 0 - X - front
    # 1 - Z - left
    # 2 - Y - up
    #--rendering [render_depth_img_size,render_depth_img_size,17]
    # 0 - top-down from Y
    # 1,2,3,4 - from X | Z
    #read my figure for details (if you can find it)
    
    #get rendering
    for x in range(dimx):
        for y in range(dimy):
            for z in range(dimz):
                if img[x,z,y]>0:
                    #z-buffering
                    
                    c = 0
                    u = x + hdis
                    v = z + hdis
                    d = -y #y must always be negative in d to render from top
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 1
                    u = y + hdis
                    v = z + hdis
                    d = x
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 2
                    u = y + hdis
                    v = z + hdis
                    d = -x
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 3
                    u = x + hdis
                    v = y + hdis
                    d = z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 4
                    u = x + hdis
                    v = y + hdis
                    d = -z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
  
    
    
    #depth fusion
    for x in range(dimx):
        for y in range(dimy):
            for z in range(dimz):
                outside_flag = 0
                
                c = 0
                u = x + hdis
                v = z + hdis
                d = -y
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 1
                u = y + hdis
                v = z + hdis
                d = x
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 2
                u = y + hdis
                v = z + hdis
                d = -x
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 3
                u = x + hdis
                v = y + hdis
                d = z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 4
                u = x + hdis
                v = y + hdis
                d = -z
                if rendering[u,v,c]>d:
                    outside_flag += 1

                if outside_flag==0:
                    img[x,z,y] = 1
    
    
    #get state_ctr
    

    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                if img[i,j,k]>0:
                    img[i,j,k] = 1
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

























