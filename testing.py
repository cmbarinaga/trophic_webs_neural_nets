
    #g_mimic.delete_vertices(apexes)
   
    #The objective of deleting the apex species is so that:
    #A consistent output layer size can be made w.r.t g_mimic
    #All apex predators can be "upgraded" to the output level


    #Use what's left to define layers
    #int cast all of the trophic levels to round down
    mode = "GMM"
    
    if mode == "GMM":
        #trophic levels are reset and recalculated due to elimination of apex
        trophic_level_indexer(g_mimic)
        
        #Find the "trophic level" of each EDGE
        #That way, you don't exclude long edges
        #And you can solve the exception with the trophic levels 1 and 2
        trophic_levels = []

        for edge in g_mimic.es:
            trophic_levels.append((g_mimic.vs[edge.source]["trophic_level"] + g_mimic.vs[edge.target]["trophic_level"]) / 2)
        
        trophic_levels = sorted(trophic_levels)


        bic_scores = []

        data = np.sort(np.array(trophic_levels).reshape(-1, 1), axis=0)
        # Fit GMMs for each number of components
        for n in range(1, int(max(trophic_levels)) + 1): #pretty wide range for num trophic means
            gm = GaussianMixture(n_components=n, random_state=42, init_params="kmeans", n_init=1)
            gm.fit(data)

            bic_scores.append(gm.bic(data))

        if plot:
            plt.plot(bic_scores)
            plt.show()

        optimal_components_bic = np.argmin(bic_scores) + 1


        gm = GaussianMixture(n_components = optimal_components_bic, random_state=42, init_params="kmeans", n_init=1).fit(data)

        sorted_indices = np.argsort(gm.means_.ravel())  # Indices to sort means

        # Create a mapping from original cluster indices to sorted indices
        cluster_mapping = {original: sorted_idx for sorted_idx, original in enumerate(sorted_indices)}

        # Get the cluster assignments
        cluster_assignments = gm.predict(data)

        # Reassign clusters based on the sorted order
        _, inter_trophic_level_edges = np.unique(np.array([cluster_mapping[cluster] for cluster in cluster_assignments]), return_counts = True)

        inter_trophic_level_edges = list(inter_trophic_level_edges)

        centers = np.sort(gm.means_.ravel())
        
        

        # for i in g_mimic.es:
        #     #see if the edge is between two adjacent trophic levels
        #     indices = np.where( (cutoffs >= (min(trophic_levels[i.source],trophic_levels[i.target]) - 0.001)) & (cutoffs <= (max(trophic_levels[i.source],trophic_levels[i.target]) + 0.001) ))[0]
        #     #Catches for when a species is exactly at a cutoff
        #     if len(indices) == 1:
        #         inter_trophic_level_edges[indices[0]] += 1
        #     elif abs(abs(trophic_levels[i.source] - trophic_levels[i.target]) - 1 ) < 0.0001:
        #         #i.e. if its basically 1, to catch for the case of 2s eating 1s
        #         val = min(trophic_levels[i.source],trophic_levels[i.target])

        #         closest_index = min(range(len(cutoffs)), key=lambda i: abs(cutoffs[i] - val))
        #         inter_trophic_level_edges[closest_index] += 1 



    
    # num_neurons = [0] * (len(inter_trophic_level_edges) + 2 ) #apex are being tagged along at end
    # #inter trophic level edges only model num of edges between each trophic level excluding the added apex layer
    # num_neurons[-1] = num_apex
    # num_neurons[0] = num_basal
    

    
    # #Odd indices:
    # odds = [x for x in range(0, len(inter_trophic_level_edges)) if x % 2 == 1]
    # evens = [x for x in range(0, len(inter_trophic_level_edges)) if x % 2 == 0]

    # print(odds)
    # print(evens)


    # if len(odds) == len(evens):
    #     scale_factor = (total_connections - (num_apex * num_basal) *  prod(inter_trophic_level_edges[i] for i in odds) / prod(inter_trophic_level_edges[i] for i in evens) ) / (sum(inter_trophic_level_edges))
    #     print("dis one")    
    # else: #length is odd
    #     scale_factor = total_connections / (sum(inter_trophic_level_edges) + (num_apex / num_basal) *  prod(inter_trophic_level_edges[i] for i in evens) / prod(inter_trophic_level_edges[i] for i in odds))

    

    # for i in range(1, len(num_neurons) - 1):
    #     num_neurons[i] = inter_trophic_level_edges[i - 1] * scale_factor / num_neurons[i - 1]
    
