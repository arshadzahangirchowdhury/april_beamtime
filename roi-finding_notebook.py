import void_clustering
from void_clustering import *

size = 1000
features = np.empty((size,3))
for idx in range(size):
    features[idx][0]=float(random.randint(-100, 100))
    features[idx][1]=float(random.randint(-100, 100))
    features[idx][2]=float(random.randint(-100, 100))

print(features.shape)

df= pd.DataFrame()
df['x']=features[:,0]
df['y']=features[:,1]
df['z']=features[:,2]

core_voids_indices,core_voids_coordinates = find_core_voids(features, 
                              estimators = [
    ("DBSCAN", DBSCAN(eps=15,min_samples=5)),
#     ("OPTICS", OPTICS(min_samples=10, cluster_method='xi')),
],
                              keep_core_voids_only = True, visualizer = True)

def n_voids_in_sphere(core_void_coord, features, thresh_dist):
    '''
    Finds voids in and on sphere with radius of thres_dist
    '''
    
    distance = np.sqrt(np.sum((features - core_void_coord)**2, axis = 1))
    
    return np.sum(distance <= thresh_dist)

def n_voids_in_ellipsoid(core_void_coord, features, a,b,c):
    '''
    Finds voids in and on ellipsoid with a,b,c parameters of ellipsoid equation (x/a)**2 + (y/a)**2 + (z/c)**2 =1
    '''
    
    
    distance = np.sqrt(np.sum((features/np.array([a,b,c]) - core_void_coord/np.array([a,b,c]))**2, axis = 1))
    
    
    return np.sum(distance <= 1)


def rank_voids(mode='spherical'):

    inside_pts_list = []
    for core_void_coordinate in core_voids_coordinates:
        n_inside = n_voids_in_ellipsoid(core_void_coordinate, features,15,16,50)
        inside_pts_list.append(n_inside)
    #     print(f'core {core_void_coordinate}; n within ellipical core {n_inside}')

    # return the coordinates of the top three core voids that contain the highest tiny voids in an ellipsoid/sphere
    rank_one_inside = core_voids_coordinates[np.argsort(np.array(inside_pts_list))[-1]]
    rank_two_inside = core_voids_coordinates[np.argsort(np.array(inside_pts_list))[-2]]
    rank_three_inside = core_voids_coordinates[np.argsort(np.array(inside_pts_list))[-3]]
    
    
for core_void_coordinate in core_voids_coordinates:
    n_inside = n_voids_in_sphere(core_void_coordinate, features, 15)
    print(f'core {core_void_coordinate}; n within spherical core {n_inside}')
    
    
    
