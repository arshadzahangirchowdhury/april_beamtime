

estimators = [
#     ("DBSCAN", DBSCAN(eps=42,min_samples=5)),
    ("OPTICS", OPTICS(min_samples=10, cluster_method='xi')),
]

def find_high_density_small_voids(features,estimators, keep_high_density_voids_only = True):
    '''
    Function to cluster all voids using their x,y,z center coordinates.
    
    For DBSCAN, adjust eps based on elbow of reachability plot.
    For OPTICS, eps is auto-adjusted. 
    
    args:
    features: numpy array, x,y,z coordinates for each void center
    estimators: list, each clustering algorithm is provided as (name, estimator class)
    keep_high_density_voids_only: boolean, removes the voids which are large from the clustering plot. These are all points whom DBSCAN/OPTICS would label -1.
    returns:
    elbow plot for eps selection and clustering in plotly and saved figures
    
    '''
    fignum = 1
    

    df= pd.DataFrame()
    df['x']=features[:,0]
    df['y']=features[:,1]
    df['z']=features[:,2]
    
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(features)
    distances, indices = nbrs.kneighbors(features)

    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.plot(distances)
    plt.title('Reachability (sorted distances to closest neighbor)')
    plt.ylabel('distances');
    fig.write_image('figure' + '_reachability' +'.jpg')
    

    for name, est in estimators:
        

        fig = plt.figure(fignum, figsize=(4, 3))

        est.fit(df.to_numpy())
        labels = est.labels_
    #     core_voids_indices = est.core_sample_indices_  #dbscan only
        df['labels'] = labels
        print('name: ', name)
    #     print('labels: ',labels)
    #     print('core_voids_indices: ',core_voids_indices) #dbscan only

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters containing small high density voids: %d" % n_clusters_)
        print("Estimated number of (noisy, outlier) voids: %d" % n_noise_)

        if keep_high_density_voids_only == True:
            secondarydf = df[df['labels'] !=-1]
        else:
            keep_high_density_voids = df


        fig = px.scatter_3d(secondarydf, x='x', y='y', z='z', size_max=15,
                      color='labels', template='simple_white',width=600, height=600)


        fig.update_traces(marker_size = 4)
        fig.update_coloraxes(colorbar_orientation='h')
        fig.update_layout(scene = dict(
                            xaxis_title='x',
                            yaxis_title='y',
                            zaxis_title='z'),
                          title=name
                            )


        fig.update_layout(
            legend=dict(
                x=0.8,
                y=1.0,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                ),
            )
        )



        fig.show()
        fig.write_image('figure' + str(fignum) +'.jpg')



        fignum = fignum + 1



