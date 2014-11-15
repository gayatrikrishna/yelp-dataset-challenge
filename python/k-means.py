import sys, time, helpers
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import * 
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans 
from sklearn.metrics.pairwise import pairwise_distances_argmin

load_data = helpers.load_data
get_attributes = helpers.get_attributes
get_class_counts = helpers.get_class_counts
get_arguments = helpers.get_arguments
get_categories = helpers.get_categories

# Select dataset....
subsets_path = [
		'../data/arff/subsets/yelp_academic_dataset_user.arff',
		'../data/arff/subsets/yelp_academic_dataset_business.arff',
		'../data/arff/subsets/yelp_academic_dataset_checkin.arff',
		'../data/arff/subsets/yelp_academic_dataset_review.arff',
		'../data/arff/subsets/yelp_academic_dataset_tip.arff',
]

fulldata_path = [
		'../data/arff/full_data/yelp_academic_dataset_user.arff',
		'../data/arff/full_data/yelp_academic_dataset_business.arff',
		'../data/arff/full_data/yelp_academic_dataset_checkin.arff',
		'../data/arff/full_data/yelp_academic_dataset_review.arff',
		'../data/arff/full_data/yelp_academic_dataset_tip.arff',
]

def kmeans_comparison(dataset, n, n_clusters):
# turn array into numpy array so we can apply their statistical methods
	X = np.asarray(dataset)
	# convert data to a scipy.sparse.coo_matrix & then to a csr matrix
	data_matrix = coo_matrix(X).tocsr()

	# Standard K-Means
	k_means = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=n, n_init=20)
	tstamp = time.time()
	k_means.fit(data_matrix)
	t_batch = time.time() - tstamp

	k_means_labels = k_means.labels_
	k_means_cluster_centers = k_means.cluster_centers_

	# Mini-Batch K-Means
	minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', max_iter=n, n_init=20)
	tstamp = time.time()
	minibatch_kmeans.fit(data_matrix)
	t_mini_batch = time.time() - tstamp

	minibatch_kmeans_labels = minibatch_kmeans.labels_
	minibatch_kmeans_cluster_centers = minibatch_kmeans.cluster_centers_

	# Plot results
	fig = plt.figure(figsize=(8, 3))
	fig.subplots_adjust(left=0.02, right=0.98, bottom=0.5, top=0.9)
	colors = ['#4EACC5', '#FF9C34', '#4E9A06']

	# We want to have the same colors for the same cluster from the
	# MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
	# closest one.
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
	plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (
			t_batch, k_means.inertia_))

	# MiniBatchKMeans
	graph = fig.add_subplot(1, 3, 2)
	for k, col in zip(range(n_clusters), colors):
			my_members = minibatch_kmeans_labels == order[k]
			cluster_center = minibatch_kmeans_cluster_centers[order[k]]
			graph.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
			graph.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
	graph.set_title('MiniBatchKMeans')
	graph.set_xticks(())
	graph.set_yticks(())
	plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %
					 (t_mini_batch, minibatch_kmeans.inertia_))

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


def business_arff_subset():
	arff_file = load_data(subsets_path[1])
	attributes = get_attributes(arff_file['attributes'])
	dataset = arff_file['data']

	# the system can only handle numeric values; convert all strings to numbers
	for row in dataset:
		count = 0
		for x in row:
			if x == None or x == 'F':
				row[count] = 0
			else:
				pass

			if x == 'T':
				row[count] = 0
			else:
				pass

			count += 1

	print dataset[0]
	exit(0)
	return attributes, dataset


def user_arff_subset():
	arff_file = load_data(subsets_path[0])
	attributes = get_attributes(arff_file['attributes'])
	dataset = arff_file['data']

	# the system can only handle numeric values; convert all strings to numbers
	for x in dataset:
		# convert the date to a usable number
		parts = x[20].split('-')
		# turn the date into a number & 
		x[20] = int(parts[0]) + (int(parts[1]) / 12.0)
		# remove the id_hash from each row
		del(x[16])

	return attributes, dataset


def main(args):
	n = 100 # number of times to iterate
	n_clusters = 8 # number clusters

	# attributes, dataset = user_arff_subset()
	attributes, dataset = business_arff_subset()

	# run the algorithm
	kmeans_comparison(dataset, n, n_clusters)



if __name__ == "__main__":
	main(sys.argv)