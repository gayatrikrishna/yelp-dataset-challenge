################################################################################
#
# Using K-means & Mini-Batch K-means to discover similarites between users
# and businesses
#
# @author Jason Feriante
#################################################################################

# Can we use this for feature extraction? 
# http://scikit-learn.org/stable/modules/feature_extraction.html
import sys, time, helpers
import numpy as np
# http://matplotlib.org/users/pyplot_tutorial.html
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans 
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.manifold import MDS #multi-dimensional scaling (flatten things)
from random import shuffle

load_data = helpers.load_data
get_attributes = helpers.get_attributes
get_class_counts = helpers.get_class_counts
get_arguments = helpers.get_arguments
get_categories = helpers.get_categories

def get_subsets_path(num):
    # Select dataset....
    subsets_path = [
        '../data/arff/subsets/yelp_academic_dataset_user.arff',
        '../data/arff/subsets/yelp_academic_dataset_business.arff',
        '../data/arff/subsets/yelp_academic_dataset_checkin.arff',
        '../data/arff/subsets/yelp_academic_dataset_review.arff',
        '../data/arff/subsets/yelp_academic_dataset_tip.arff',
        ]

    return subsets_path[num]

def get_fulldata_path(num):
    fulldata_path = [
        '../data/arff/full_data/yelp_academic_dataset_user.arff',
        '../data/arff/full_data/yelp_academic_dataset_business.arff',
        '../data/arff/full_data/yelp_academic_dataset_checkin.arff',
        '../data/arff/full_data/yelp_academic_dataset_review.arff',
        '../data/arff/full_data/yelp_academic_dataset_tip.arff',
        ]

    return fulldata_path[num]

def get_subset(dataset):
    shuffle(dataset)
    if(len(dataset) > 1000):
        size = float(len(dataset)) * 0.02

        subset = []
        count = 0
        while count <= size:
            subset.append(dataset[count])
            count += 1

        return subset
    else:
        return dataset


def kmeans_comparison(dataset, n, n_clusters, n_init):
# turn array into numpy array so we can apply their statistical methods
    X = np.asarray(dataset)
    # convert data to a scipy.sparse.coo_matrix & then to a csr matrix
    X = coo_matrix(X).tocsr()

    print 'Multi-Dimensional Scaling'
    # Multi-dimensional Scaling (MDS)
    # http://scikit-learn.org/dev/modules/manifold.html#multi-dimensional-scaling-mds
    multi_dim_scaling = MDS(n_components=2, metric=True, n_init=3, max_iter=100, 
        verbose=0, eps=0.001, random_state=None, dissimilarity='euclidean')
    # this really squashes the data & makes it easier to process & allows us
    # to actually turn our 1000 dimension problem into a 2D problem... 
    # otherwise, the datasets are far too big to handle...
    X = multi_dim_scaling.fit_transform(X)


    # Standard K-Means
    print 'Standard K-Means'
    k_means = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=n,
        n_init=n_init)
    # tstamp = time.time()
    k_means.fit(X)
    # t_batch = time.time() - tstamp

    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_

    # Mini-Batch K-Means
    print 'Mini-Batch K-Means '
    minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++',
        max_iter=n, n_init=n_init)
    minibatch_kmeans.fit(X)
    minibatch_kmeans_labels = minibatch_kmeans.labels_
    minibatch_kmeans_cluster_centers = minibatch_kmeans.cluster_centers_

    print 'Plot Results'
    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.08, top=0.9)
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']

    # Keep same colors for the respective clusters generated by the K-Means 
    # and the Mini-Batch K-Means algorithms. 
    order = pairwise_distances_argmin(k_means_cluster_centers, minibatch_kmeans_cluster_centers)

    # KMeans
    graph = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(n_clusters), colors):
            my_members = k_means_labels == k
            cluster_center = k_means_cluster_centers[k]
            graph.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
            graph.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    graph.set_title('KMeans')
    graph.set_xticks(())
    graph.set_yticks(())

    # MiniBatchKMeans
    graph = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(n_clusters), colors):
            my_members = minibatch_kmeans_labels == order[k]
            cluster_center = minibatch_kmeans_cluster_centers[order[k]]
            graph.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
            graph.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    graph.set_title('Mini-Batch KMeans')
    graph.set_xticks(())
    graph.set_yticks(())

    # Differences
    different = (minibatch_kmeans_labels == 4)
    graph = fig.add_subplot(1, 3, 3)
    for l in range(n_clusters):
            different += ((k_means_labels == k) != (minibatch_kmeans_labels == order[k]))

    identical = np.logical_not(different)
    graph.plot(X[different, 0], X[different, 1], 'w', markerfacecolor='m', marker='.')
    graph.plot(X[identical, 0], X[identical, 1], 'w', markerfacecolor='#bbbbbb', marker='.')
    graph.set_title('Difference')
    graph.set_xticks(())
    graph.set_yticks(())

    # display the results
    plt.show()

def hours_to_float(hours):
    """ Turn hours into a float number """
    if(hours == 0):
        return 0
    else:
        parts = hours.split(':')
        # turn the date into a number & 
        hours_minutes = int(parts[0]) + (int(parts[1]) / 60.0) 
        # divide by 24 hours in a day
        return round(hours_minutes / 24.0, 3) 

def clean_business_atttributes(row, nominal_bus_attrs):
    """ Normalize business values """
    for key in nominal_bus_attrs:
        i = nominal_bus_attrs[key].get('index')
        if row[i] != 0:
            opt_hashes = nominal_bus_attrs[key].get('opt_hashes')
            cardinality = nominal_bus_attrs[key].get('cardinality')
            val = opt_hashes.get(row[i])
            normalized = val / cardinality
            row[i] = round(normalized, 4)
        else:
            pass

    return row


def get_nominal_bus_attrs(attributes):
    """ Figure out which nominal attributes need to get converted to numbers"""
    nominal_bus_attrs = {}
    binary_opt = ['F', 'T']
    for attr in attributes:
        curr_opt = attributes[attr].get('options')
        attr_type = attributes[attr].get('type')
        isHrs = str(attr).find("hours")

        if(curr_opt != binary_opt and isHrs < 0 and attr != 'city' 
            and attr != 'state' and attr_type != 'numeric'):
            nominal_bus_attrs[attr] = attributes[attr]
        else:
            pass

    for attr in nominal_bus_attrs:
        opt_hashes = {}
        curr_opts = nominal_bus_attrs[attr].get('options')
        nominal_bus_attrs[attr]['cardinality'] = float(len(curr_opts))
        count = 1
        for opt in curr_opts:
            opt_hashes[opt] = count
            count += 1

        nominal_bus_attrs[attr]['opt_hashes'] = opt_hashes

    return nominal_bus_attrs


def business_arff_subset():
    # arff_file = load_data(get_subsets_path(1))
    arff_file = load_data(get_fulldata_path(1))
    attributes = get_attributes(arff_file['attributes'])
    dataset = arff_file['data']

    business_index = attributes.get('business_id').get('index') #78
    state_index = attributes.get('state').get('index') # 908

    # convet states to numbers
    states = attributes.get('state').get('options')
    state_len = float(len(states))
    state = {}
    count = 1
    for s in states:
        state[s] = count
        count += 1  

    # build an object to translate cities to numbers
    cities = attributes.get('city').get('options')
    cities_len = float(len(cities))
    city_index = attributes.get('city').get('index')
    city = {}
    count = 1
    for c in cities:
        city[c] = count
        count += 1

    # get indexes for the various hours attributes:
    sunday_o = attributes.get('hours.Sunday.open').get('index')
    sunday_c = attributes.get('hours.Sunday.close').get('index')

    monday_o = attributes.get('hours.Monday.open').get('index')
    monday_c = attributes.get('hours.Monday.close').get('index')

    tuesday_o = attributes.get('hours.Tuesday.open').get('index')
    tuesday_c = attributes.get('hours.Tuesday.close').get('index')

    wednesday_o = attributes.get('hours.Wednesday.open').get('index')
    wednesday_c = attributes.get('hours.Wednesday.close').get('index')

    thursday_o = attributes.get('hours.Thursday.open').get('index')
    thursday_c = attributes.get('hours.Thursday.close').get('index')

    friday_o = attributes.get('hours.Friday.open').get('index')
    friday_c = attributes.get('hours.Friday.close').get('index')

    saturday_o = attributes.get('hours.Saturday.open').get('index')
    saturday_c = attributes.get('hours.Saturday.close').get('index')

    nominal_bus_attrs = get_nominal_bus_attrs(attributes)

    # the system can only handle numeric values; convert all strings to numbers
    for row in dataset:
        count = 0
        for x in row:
            if x == None or x == 'F':
                row[count] = 0
            else:
                pass

            if x == 'T':
                row[count] = 1
            else:
                pass

            count += 1

        # turn business hours into a float
        row[sunday_o] = hours_to_float(row[sunday_o])
        row[sunday_c] = hours_to_float(row[sunday_c])
        row[monday_o] = hours_to_float(row[monday_o])
        row[monday_c] = hours_to_float(row[monday_c])
        row[tuesday_o] = hours_to_float(row[tuesday_o])
        row[tuesday_c] = hours_to_float(row[tuesday_c])
        row[wednesday_o] = hours_to_float(row[wednesday_o])
        row[wednesday_c] = hours_to_float(row[wednesday_c])
        row[thursday_o] = hours_to_float(row[thursday_o])
        row[thursday_c] = hours_to_float(row[thursday_c])
        row[friday_o] = hours_to_float(row[friday_o])
        row[friday_c] = hours_to_float(row[friday_c])
        row[saturday_o] = hours_to_float(row[saturday_o])
        row[saturday_c] = hours_to_float(row[saturday_c])

        # fix city, state, & business indices (no strings allowed)
        row[city_index] = round(city.get(row[city_index]) / cities_len, 4)
        row[state_index] = round(state.get(row[state_index]) / state_len, 4)
        row[business_index] = 0

        row = clean_business_atttributes(row, nominal_bus_attrs)

    # extract a random sample from the dataset since 42k businesses is too much
    dataset = get_subset(dataset)

    return attributes, dataset


def user_arff_subset():
    # arff_file = load_data(get_subsets_path(0))
    arff_file = load_data(get_fulldata_path(0))
    attributes = get_attributes(arff_file['attributes'])
    dataset = arff_file['data']

    # the system can only handle numeric values; convert all strings to numbers
    for x in dataset:
        # convert the date to a usable number
        parts = x[20].split('-')
        # turn the date into a number & 
        x[20] = int(parts[0]) + (int(parts[1]) / 12.0)
        # remove the id_hash from each row
        x[16] = 0

    # extract a random sample from the dataset since 250k users is too much
    dataset = get_subset(dataset)
    
    return attributes, dataset


def main(args):
    n = 100 # number of times to iterate
    n_clusters = 5 # number clusters
    n_init = 4

    # attributes, dataset = user_arff_subset()
    attributes, dataset = business_arff_subset()

    # run the algorithm
    kmeans_comparison(dataset, n, n_clusters, n_init)


if __name__ == "__main__":
    main(sys.argv)
