################################################################################
#
# Using K-means to predict user businesses ratings
# 
# @author Jason Feriante
# 
# Paper on Clustering Methods For Collaborative Filtering:
# http://www.aaai.org/Papers/Workshops/1998/WS-98-08/WS98-08-029.pdf
# 
# PCA & K-Means both have problems with binary data -- and unfortunately the 
# majority of our data is binary. That might limit the effectiveness of these
# methods...
#################################################################################

# Can we use this for feature extraction? 
# http://scikit-learn.org/stable/modules/feature_extraction.html
import sys, time, helpers, copy, math
import pylibmc # use memcached to hold our dataset in memory because it loads way too slow otherwise.
import numpy as np
# http://matplotlib.org/users/pyplot_tutorial.html
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.sparse import coo_matrix
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans 
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.manifold import MDS #multi-dimensional scaling (flatten things)
from random import shuffle
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing

load_data = helpers.load_data
get_attributes = helpers.get_attributes
get_class_counts = helpers.get_class_counts
get_arguments = helpers.get_arguments
get_categories = helpers.get_categories

memcache = pylibmc.Client(["127.0.0.1"], binary=True, behaviors={"tcp_nodelay": True, "ketama": True})

def get_subsets_path(num):
    # Select dataset....
    subsets_path = [
        '../data/arff/subsets/yelp_academic_dataset_user.arff',
        '../data/arff/subsets/yelp_academic_dataset_business.arff',
        '../data/arff/subsets/yelp_academic_dataset_checkin.arff',
        '../data/arff/subsets/yelp_academic_dataset_review.arff',
        '../data/arff/subsets/yelp_academic_dataset_tip.arff',
        '../data/arff/_review_matrix/user_review_matrix_jason.arff',
        ]

    return subsets_path[num]

def get_fulldata_path(num):
    fulldata_path = [
        '../data/arff/full_data/yelp_academic_dataset_user.arff',
        '../data/arff/full_data/yelp_academic_dataset_business.arff',
        '../data/arff/full_data/yelp_academic_dataset_checkin.arff',
        '../data/arff/full_data/yelp_academic_dataset_review.arff',
        '../data/arff/full_data/yelp_academic_dataset_tip.arff',
        '../data/arff/_review_matrix/user_review_matrix_jason_big.arff',
        # '../data/arff/_review_matrix/user_review_matrix_jason_massive.arff',
        ]

    return fulldata_path[num]

def get_subset(dataset):
    # shuffle(dataset)
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

def kmeans_graph_comparison(dataset, n, n_clusters, n_init):
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
    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
    print 'Standard K-Means'
    k_means = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=n, n_init=n_init)
    # tstamp = time.time()
    k_means.fit(X)
    # t_batch = time.time() - tstamp
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_

    # Mini-Batch K-Means
    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans
    print 'Mini-Batch K-Means '
    minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', max_iter=n, n_init=n_init)
    minibatch_kmeans.fit(X)
    minibatch_kmeans_labels = minibatch_kmeans.labels_
    minibatch_kmeans_cluster_centers = minibatch_kmeans.cluster_centers_

    print 'Plot Results'
    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.08, top=0.9)
    # blue #0000FF - red #FF0000 - light blue #4EACC5 - orange #FF9C34 
    # light green #4E9A06 - yellow #FFFF00 - navy #000080 - fuchsia #FF00FF
    colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#0000FF', '#FF0000', '#FFFF00', '#000080', '#FF00FF']

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

def kmeans_graph(dataset, n, n_clusters, n_init):
# turn array into numpy array so we can apply their statistical methods
    X = np.asarray(dataset)
    # convert data to a scipy.sparse.coo_matrix & then to a csr matrix
    X = coo_matrix(X).tocsr()

    print 'Multi-Dimensional Scaling' # squash the results down to 2 dimensions
    # Multi-dimensional Scaling (MDS)
    # http://scikit-learn.org/dev/modules/manifold.html#multi-dimensional-scaling-mds
    # n_jobs=-1 means run this on all CPUs...
    multi_dim_scaling = MDS(n_components=2, metric=False, n_init=1, max_iter=300, 
        verbose=0, eps=0.001, n_jobs=-1, random_state=None, dissimilarity='euclidean')
    X = multi_dim_scaling.fit_transform(X)

    # Mini-Batch K-Means
    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans
    # print 'Mini-Batch K-Means '
    # minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', max_iter=n, n_init=n_init, compute_labels=True)
    # minibatch_kmeans.fit(X)
    # kmeans_labels = minibatch_kmeans.labels_
    # kmeans_centers = minibatch_kmeans.cluster_centers_
    # kmeans_inertia = minibatch_kmeans.inertia_

    print 'Standard K-Means'
    k_means = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=n, n_init=n_init)
    # tstamp = time.time()
    k_means.fit(X)
    # t_batch = time.time() - tstamp
    kmeans_labels = k_means.labels_
    kmeans_centers = k_means.cluster_centers_
    kmeans_inertia = k_means.inertia_

    print 'Plot Results'
    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.08, top=0.9)
    # blue #0000FF - red #FF0000 - light blue #4EACC5 - orange #FF9C34 
    # light green #4E9A06 - yellow #FFFF00 - navy #000080 - fuchsia #FF00FF
    colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#0000FF', '#FF0000', '#FFFF00', '#000080', '#FF00FF']

    # MiniBatchKMeans
    graph = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(n_clusters), colors):
            graph.plot(X[kmeans_labels, 0], X[kmeans_labels, 1], 'w', markerfacecolor=col, markersize=4)
            graph.plot(kmeans_centers[k][0], kmeans_centers[k][1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=8)
    graph.set_title('Mini-Batch KMeans')
    graph.set_xticks(())
    graph.set_yticks(())

    # display the results
    plt.show()

def split_dataset(dataset):
    """Split the dataset into a training_set and test_set"""
    # shuffle(dataset)
    size = len(dataset)
    test_set_size = int(round(size * 0.1, 0))

    # 10% of the instances are the test set
    test_set = []
    for i in range(test_set_size):
        test_set.append(dataset[i])

    # the rest of the instances are the training set
    training_set = []
    for i in range(test_set_size, size):
        training_set.append(dataset[i])

    return training_set, test_set

def get_baseline(data):
    counts = [0.0] * len(data[0])
    sums = [0.0] * len(data[0])
    baseline = [0.0] * len(data[0])

    # if we have no predictions at all, this becomes our prediction
    universal_average = 0.0

    for row in range(len(data)):
        for col in range(len(data[0])):
            if(data[row][col] > 0):
                counts[col] += 1
                sums[col] += data[row][col]

    for i in range(len(data[0])):
        if(counts[i] > 0):
            # round up to the nearest whole number
            baseline[i] = round(sums[i] / counts[i], 2)
        else:
            baseline[i] = 0
    
    return baseline

def kmeans_prediction(dataset, n, n_clusters, n_init, attributes, to_predict, baseline):
    # turn array into numpy array so we can apply their statistical methods
    train = np.asarray(dataset)
    # convert data to a scipy.sparse.coo_matrix & then to a csr matrix
    train = coo_matrix(dataset).tocsr()

    print 'K-Means'
    # k_means = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=n, n_init=n_init)
    k_means = KMeans(n_clusters=n_clusters, init='random', max_iter=n, n_init=n_init, n_jobs=4)
    # k_means = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', max_iter=n, n_init=n_init, compute_labels=True, reassignment_ratio=.7)
    k_means.fit_transform(train)
    kmeans_labels = k_means.labels_
    kmeans_centers = k_means.cluster_centers_
    kmeans_inertia = k_means.inertia_

    biz_index = attributes.get('B0').get('index')

    # initialize a row to put our predictions in
    init_row = []
    for i in range(0, len(dataset[0])):
        init_row.append(0)

    # initialize cluster predictions
    init_row = [0.0] * len(dataset[0])

    # init empty data clusters
    data_clusters = []
    for i in range(0, n_clusters):
        data_cluster = []
        data_clusters.append(data_cluster)

    for i in range(0, len(dataset)):
        # assign each row to the right cluster
        data_clusters[kmeans_labels[i]].append(dataset[i])

    print 'Make K-means Predictions'
    cluster_index = 0
    cluster_predictions = []
    for i in range(n_clusters):
        cluster_predictions.append(init_row)

    # kmeans_labels[0] <--e.g. tells us what cluster each user belongs to
    # now we can build our prediction matrix for each cluster
    empty_clusters = 0
    for d_set in data_clusters:
        if(len(d_set) > 0):
            # for each row in this dataset, make a prediction (based on the average rating)
            # if there's no rating, use the business average rounded to the nearest star
            # (since we have nothing else to go on)
            # we will do this column, by column.
            col = biz_index # optimization
            while(col < len(d_set[0])):
                review_count = 0.0
                total_stars = 0.0
                avg_stars = 0.0
                for row in range(len(d_set)):
                    if(d_set[row][col] > 0):
                        review_count += 1 
                        total_stars += d_set[row][col]
                    else:
                        pass
                    
                if(review_count > 0):
                    avg_stars = total_stars / review_count
                else:
                    pass
                cluster_predictions[cluster_index][col] = avg_stars
                col += 1
        else:
            empty_clusters += 1

        cluster_index += 1

    results = []
    correct = 0

    # we will make 1x prediction per user (that's all); maybe we should make more?
    for i in range(len(to_predict)):
        cluster_index = kmeans_labels[i]
        # expected stars
        exp_stars = to_predict[i].get('stars')
        # the col to predict
        col = to_predict[i].get('col') 

        pred_stars = cluster_predictions[cluster_index][col]

        if(pred_stars < 1):
            # each column is a business; get the average for it... 
            # since we have nothing else to predict with
            pred_stars = baseline[col]

        else:
            pass

        results.append({ 'expected_stars': exp_stars, 'predicted_stars': pred_stars })
        difference = abs(exp_stars - pred_stars)

        if difference <= 1:
        # if difference <= 0.5:
            results[i]['is_correct'] = True
            # print 'predicted: %f, actual: %f, is_correct: %s' % (results[i]['predicted_stars'], results[i]['expected_stars'], 'Yes')
            correct += 1
        else:
            results[i]['is_correct'] = False
            # print 'predicted: %f, actual: %f, is_correct: %s' % (results[i]['predicted_stars'], results[i]['expected_stars'], 'No')

    size = len(dataset)
    print 'predicted: %d, correct: %d %%:%f, clusters: %d, empty: %d' % (size, correct, correct / float(size), n_clusters, empty_clusters)
    return correct / float(size)


def kmeans_business_prediction(dataset, n, n_clusters, n_init, attributes):
    training_set, test_set = split_dataset(dataset)
    # turn array into numpy array so we can apply their statistical methods
    test = np.asarray(test_set)
    print test
    train = np.asarray(training_set)

    # convert data to a scipy.sparse.coo_matrix & then to a csr matrix
    test = coo_matrix(test).tocsr()
    train = coo_matrix(train).tocsr()


    # build star sets
    # (this ONLY works if the attributes actually have a stars rating)
    stars_row = attributes.get('stars').get('index')

    train_star_labels = []
    for row in range(len(training_set)):
        train_star_labels.append(training_set[row][stars_row])

    # wrap the list in a list.. because sci-kit wants this

    test_star_labels = []
    for row in range(len(test_set)):
        test_star_labels.append(test_set[row][stars_row])

    # build labels that sci-kit understands
    le = preprocessing.LabelEncoder()
    train_star_labels = le.fit_transform(train_star_labels)
    test_star_labels = le.fit_transform(test_star_labels)

    # chi-squared & select-k-best
    # http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
    # http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
    chi_squared = SelectKBest(chi2, k=50)
    X_train = chi_squared.fit_transform(train, train_star_labels)
    chi_scores = chi_squared.scores_
    chi_pvalues = chi_squared.pvalues_
    # X_test = chi_squared.transform(test)

    # train a subset of the data with the better chi_scores
    
    # print chi_pvalues
    # print 'len(chi_pvalues)'
    # print len(chi_pvalues)
    # # print X_train
    # exit(0)

    # another possible solution: RandomizedLogisticRegression
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RandomizedLogisticRegression.html

    # Mini-Batch K-Means
    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans
    print 'Mini-Batch K-Means '
    # print 'K-Means '
    minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', max_iter=n, n_init=n_init, compute_labels=True, max_no_improvement=20)
    # minibatch_kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=n, n_init=n_init)
    minibatch_kmeans.fit(train)
    kmeans_train_labels = minibatch_kmeans.labels_
    kmeans_train_centers = minibatch_kmeans.cluster_centers_
    kmeans_train_inertia = minibatch_kmeans.inertia_


    # use the training labels to get the average stars for each business
    predict = []
    # each predict object will have average stars; this will serve as our generic prediction.
    for i in range(n_clusters):
        predict.append({'total_stars':0, 'instance_count':0, 'average_stars':0})

    for i in range(len(kmeans_train_labels)):
        # determine which cluster this instance was in
        cluster = kmeans_train_labels[i]
        stars = training_set[i][stars_row]
        # print 'stars: ' + str(stars)
        if(stars == 0):
            # zero's aren't possible; just here as a precaution.
            # i.e. this should never happen
            pass
        else:
            # update predictions for this cluster
            predict[cluster]['total_stars'] += stars
            predict[cluster]['instance_count'] += 1

    # determine the average stars for each cluster; this becomes our prediction.
    empty_clusters = 0
    for i in range(n_clusters):
        if(predict[i]['instance_count']) > 0:
            predict[i]['average_stars'] = predict[i]['total_stars'] / float(predict[i]['instance_count'])
            print predict[i]
        else:
            # print 'cluster #:' + str(i)
            empty_clusters += 1
            print predict[i]

    # now run our prediction for test instances:
    kmeans_test_labels = minibatch_kmeans.predict(test)
    
    results = []
    threshold = 0.5 # assume a prediction within 10% is correct
    correct = 0
    size = len(test_set)
    for instance in range(size):
        cluster = kmeans_test_labels[instance]
        results.append({ 'actual_stars': test_set[instance][stars_row], 'predicted_stars': predict[cluster]['average_stars'] })
        difference = abs(test_set[instance][stars_row] - predict[cluster]['average_stars'])
        if difference < threshold:
            results[instance]['is_correct'] = True
            # print 'predicted: %f, actual: %f, is_correct: %s' % (results[instance]['predicted_stars'], results[instance]['actual_stars'], 'Yes')
            correct += 1
        else:
            results[instance]['is_correct'] = False
            # print 'predicted: %f, actual: %f, is_correct: %s' % (results[instance]['predicted_stars'], results[instance]['actual_stars'], 'No')

    print 'predicted: %d, correct: %d, %% correct: %f, empty: %d, clusters: %d' % (size, correct, correct / float(size), empty_clusters, n_clusters - empty_clusters)
    return correct / float(size)


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
    print 'Running business arff subset'
    arff_file = load_data(get_subsets_path(1))
    # print 'Running business arff full_data'
    # arff_file = load_data(get_fulldata_path(1))
    attributes = get_attributes(arff_file['attributes'])
    dataset = arff_file['data']

    business_index = attributes.get('business_id').get('index') #78
    state_index = attributes.get('state').get('index') # 908

    longitude_index = attributes.get('longitude').get('index') # 908

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

        # fix longitude (chi-square allows no negatives)
        # longitude min is -180
        row[longitude_index] = row[longitude_index] + 180

    # extract a random sample from the dataset since 42k businesses is too much
    # dataset = get_subset(dataset)

    return attributes, dataset


def user_arff_subset():
    print 'Running user arff subset'
    arff_file = load_data(get_subsets_path(0))
    # print 'Running user arff full_data'
    # arff_file = load_data(get_fulldata_path(0))
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
    # dataset = get_subset(dataset)
    
    return attributes, dataset


def user_reviews_full():
    return user_reviews_inner(get_fulldata_path(5))

def user_reviews_subset():
    return user_reviews_inner(get_subsets_path(5))

def user_reviews_inner(path):
    print 'Running user arff subset'

    arff_file = load_data(path)
    # print 'Running user arff full_data'
    # arff_file = load_data(get_fulldata_path(0))
    attributes = get_attributes(arff_file['attributes'])
    dataset = arff_file['data']

    biz_index = attributes.get('B0').get('index')
    # the system can only handle numeric values; convert all strings to numbers
    to_predict = []
    for row in dataset:
        i = 0
        #0 yelping_since string
        date = datetime.strptime(row[0], '%Y-%m-%d')
        row[0] = time.mktime(date.timetuple())
        #1 compliments.plain numeric
        #2 compliments.more numeric
        #3 elite numeric
        #4 compliments.cute numeric
        #5 compliments.writer numeric
        #6 fans numeric
        #7 compliments.note numeric
        #8 compliments.hot numeric
        #9 compliments.cool numeric
        #10 compliments.profile numeric
        #11 average_stars numeric
        #12 review_count numeric
        #13 friends numeric
        #14 user_id string
        row[14] = 0
        #15 votes.cool numeric
        #16 compliments.list numeric
        #17 votes.funny numeric
        #18 compliments.photos numeric
        #19 compliments.funny numeric
        #20 votes.useful numeric
        while(i < biz_index):
            row[i] = 0
            i += 1

        # find one review to predict for each user...
        i = biz_index
        _max = len(row) - 1
        while(i < _max):
            if(row[i] != None and row[i] > 0):
                to_predict.append({'col':i, 'stars':row[i]})
                row[i] = 0 # withold this value from our dataset!
                _max = _max + 2
                break
            else:
                pass
            i += 1

    # extract a random sample from the dataset since 250k users is too much
    # dataset = get_subset(dataset)

    return attributes, dataset, to_predict

# use the rule of thumb to get the number of clusters, & use a feature selection
# algorithm to maximumize predictive accuracy
def main(args):
    n = 100 # number of times to iterate
    # n_clusters = 50 # number clusters
    n_init = 20

    # attributes, dataset = user_arff_subset()
    # attributes, dataset = business_arff_subset()
    attributes, dataset, to_predict = user_reviews_subset()
    # attributes, dataset, to_predict = user_reviews_full()

    # assume test set is 10% the size of the dataset. this will then also 
    # determine our ideal number of clusters

    # use the "rule of thumb" to get the right number of clusters
    n_clusters = int( math.sqrt(len(dataset) / 2) )

    # run K-means minibatch
    # kmeans_graph(dataset, n, n_clusters, n_init)

    # compare K-means vs K-means Mini-Batch
    # kmeans_graph_comparison(dataset, n, n_clusters, n_init)
    
    # select features; 0 to 20 (the rest -- reviews, are necessary)
    # kmeans_business_prediction(dataset, n, n_clusters, n_init, attributes)

    # the baseline is the universal average
    baseline = get_baseline(dataset)

    accuracy = 0
    for i in range(10):
        accuracy += kmeans_prediction(dataset, n, n_clusters, n_init, attributes, to_predict, baseline)

    print "Average accuracy: %.2f%%" % ((100 * accuracy) / 10.0)
    
    # feature_selection(dataset, n, n_clusters, n_init, attributes)
    


if __name__ == "__main__":
    main(sys.argv)
