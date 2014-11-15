import sys, numpy, scipy, matplotlib, helpers
from sklearn.cluster import KMeans


load_data = helpers.load_data
get_attributes = helpers.get_attributes
get_class_counts = helpers.get_class_counts
get_arguments = helpers.get_arguments
get_categories = helpers.get_categories

def run_kmeans():
	KMeans(
		8, # n_clusters
		'k-means++', #init
		10, # n_init
		10, # max_iter
		0.0001, #tol
		True, #precompute_distances
		# 0, verbose
		None, #random_state
		True, #copy_x
		1, # n_jobs
	)
	

def arff_load(filepath):
	handle = open(filepath)
	return arffread(handle)

def print_sklearn_version():
	print 'Version of Scikit: ' + sklearn.__version__  
	path = sklearn.__path__
	print 'Path to Scikit: ' + str(path[0])
	exit(0)

def main(args):
	# print_sklearn_version()
	# run_kmeans()
	path = [
		'../data/arff/subsets/yelp_academic_dataset_user.arff',
		'../data/arff/subsets/yelp_academic_dataset_business.arff',
		'../data/arff/subsets/yelp_academic_dataset_checkin.arff',
		'../data/arff/subsets/yelp_academic_dataset_review.arff',
		'../data/arff/subsets/yelp_academic_dataset_tip.arff',
	]

	arff_file = load_data(path[0])
	attributes = get_attributes(arff_file['attributes'])
	categories = get_categories(attributes)
	data = arff_file['data']

	# print attributes
	print data[0]
	exit(0)

if __name__ == "__main__":
	main(sys.argv)


