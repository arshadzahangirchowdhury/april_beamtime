import numpy as np
def load_ply(fileobj):
    """Same as load_ply, but takes a file-like object"""
    def nextline():
        """Read next line, skip comments"""
        while True:
            line = fileobj.readline()
#             assert line != ''  # eof
            if not line.startswith('comment'):
                return line.strip()

    assert nextline() == 'ply'
    assert nextline() == 'format ascii 1.0'
    line = nextline()
    assert line.startswith('element vertex')
    nverts = int(line.split()[2])
    # print 'nverts : ', nverts
    assert nextline() == 'property float x'
    assert nextline() == 'property float y'
    assert nextline() == 'property float z'
    line = nextline()

#     print(line.startswith('property uchar'))
    
    if line.startswith('property uchar'):
        vertex_color_flag = 1
        assert line == 'property uchar red'
        assert nextline() == 'property uchar green'
        assert nextline() == 'property uchar blue'
        line = nextline()
        
        
    
    assert line.startswith('element face')
    nfaces = int(line.split()[2])
    # print 'nfaces : ', nfaces
    assert nextline() == 'property list uchar int vertex_indices'
    line = nextline()
    has_texcoords = line == 'property list uchar float texcoord'
    if has_texcoords:
        assert nextline() == 'end_header'
    else:
        assert line == 'end_header'

    # Verts
    verts = np.zeros((nverts, 3))
    verts_color = np.zeros((nverts, 3))
    for i in range(nverts):
        vals = nextline().split()
        verts[i, :] = [float(v) for v in vals[:3]]
        if vertex_color_flag ==1:
            verts_color[i, :] = [float(v) for v in vals[3:]]
        
        
            
    # Faces
    faces = []
    faces_uv = []
    for i in range(nfaces):
        vals = nextline().split()
        assert int(vals[0]) == 3
        faces.append([int(v) for v in vals[1:4]])
#         import pdb; pdb.set_trace()

        if has_texcoords:
            assert len(vals) == 11
            assert int(vals[4]) == 6
            faces_uv.append([(float(vals[5]), float(vals[6])),
                             (float(vals[7]), float(vals[8])),
                             (float(vals[9]), float(vals[10]))])
            # faces_uv.append([float(v) for v in vals[5:]])
        else:
            assert len(vals) == 4
            
    if vertex_color_flag ==1:
        return verts, faces, verts_color
    else:
        return verts, faces, faces_uv





# f = open('big_void.ply', 'r')
f = open('coarse_map.ply', 'r')
  
# text = f.read(500) ; print(text)  # keep it disabled else it will move readline counter

verts, faces, faces_uv = load_ply(f)
f.close()
