################################################################################
#
# Using Non-Negative Matrix Factorization to predict consumer businesses ratings
# 
# @author Jason Feriante
# 
#################################################################################

# Can we use this for feature extraction? 
# http://scikit-learn.org/stable/modules/feature_extraction.html
import sys, time, helpers, copy, math
import pylibmc # use memcached to hold our dataset in memory because it loads way too slow otherwise.
import numpy as np
# http://matplotlib.org/users/pyplot_tutorial.html
from datetime import datetime
from scipy.sparse import coo_matrix
from random import shuffle
from random import randint
# np.random.seed(0) #@todo remove this later.......

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
        '../data/arff/_review_matrix/user_review_matrix_final2.arff',
        # '../data/arff/_review_matrix/user_review_matrix_jason_massive.arff',
        ]

    return fulldata_path[num]

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
            baseline[i] = sums[i] / float(counts[i])
        else:
            baseline[i] = 0
    
    return baseline, counts

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

        # wipe out early rows
        i = 0
        while(i < biz_index):
            row[i] = 0
            i += 1

    # extract a random sample from the dataset since 250k users is too much
    # dataset = get_subset(dataset)

    return attributes, dataset

def update(V, W, H, WH, V_div_WH):
    """ 
        --- Non-Negative Matrix Factorization ---
        -V is the NxM matrix to factor, where N is the number of customers
            and M is the number of businesses which were reviewed. 
        -V factors into WH (which is smaller / compressed) where: 
            W = N * r
            H = r * M
    """
    H *= (np.dot(V_div_WH.T, W) / W.sum(axis=0)).T # eq5

    WH = W.dot(H)
    V_div_WH = V / WH
    W *= np.dot(V_div_WH, H.T) / H.sum(axis=1)

    WH = W.dot(H)
    V_div_WH = V / WH
    return W, H, WH, V_div_WH


def factorize(V, L, iterations=100):
    """ 
        --- Non-Negative Matrix Factorization ---
        -V is the NxM matrix to factor, where N is the number of customers
            and M is the number of businesses which were reviewed. 
        -W is tall & skinny (compared to V)
        -H is short & wide  (compared to V)
        -Based on local divergence optima, we will decompose V into WH
        -L latent factors / compression
    """

    Vexpec = V.mean() #expected value of a matrix entry

    N, M = V.shape

    # initialize H to some random starting point
    H = np.random.random(L * M)
    # reshape the matrix to the dimensions NMF requires
    H = H.reshape(L, M) * Vexpec

    # initialize W to some random starting point
    W = np.random.random(N * L)
    # reshape the matrix to the dimensions NMF requires
    W = W.reshape(N, L) * Vexpec

    WH = W.dot(H)
    V_div_WH = V / WH

    for i in range(iterations):
        W, H, WH, V_div_WH = update(V, W, H, WH, V_div_WH)

        kbdivergence = ((V * np.log(V_div_WH)) - V + WH).sum() # eq3
        # debug
        # print "At iteration %d, the Kullback-Liebler divergence is %.8f" % (i, kbdivergence)

    return W, H

def nmf(V, latent_factors, iterations=100, const=0, normalize=0):
    # factorize into W & H
    W, H = factorize(V, latent_factors, iterations)
    
    # get the dotproduct with our predictions... 
    X = W.dot(H)

    return denormalize(X, normalize, const)

def denormalize(X, normalize, const):
    N, M = X.shape
    # X = X.tolist()
    for i in range(N):
        for j in range(M):
            X[i][j] = (X[i][j] / normalize) - const
            if(X[i][j] > 5):
                X[i][j] = 5
            elif(X[i][j] < 1):
                X[i][j] = 1
            else:
                pass

    return X

# get a matrix with our baseline estimates built-in
def get_baselines(dataset, normalize):
    V = np.asarray(dataset)

    # expected value across all reviews
    Vexpec = V.sum() / np.count_nonzero(V)
    row, col = V.shape

    # get the average for each business
    biz_avgs = V.sum(axis=0)
    biz_counts = [0] * len(biz_avgs)

    # get averages for business; default to the universal average if it's unknown
    X = V.T
    for i in range(len(biz_avgs)):
        for j in range(len(dataset)):
            if(X[i][j] > 0):
                biz_counts[i] += 1
            else:
                pass

    for i in range(len(biz_avgs)):
        if(biz_counts[i] == 0):
            biz_avgs[i] = 0
        else:
            biz_avgs[i] = (float(biz_avgs[i]) / biz_counts[i]) - Vexpec

    # get user averages; there should always be at least one rating
    cold_start_count = 0
    user_avgs = [0] * len(dataset)
    cold_start_rows = []
    for i in range(len(dataset)):
        nz_count = np.count_nonzero(V[i])
        if(nz_count > 0):
            # needs to consider business average which I don't have.
            user_avgs[i] = (float(V[i].sum()) / np.count_nonzero(V[i])) - Vexpec
        else:
            rand = randint(1,9) / 1000.0
            user_avgs[i] = rand # we know nothing about this user; they had zero test-set ratings
            cold_start_count += 1
            cold_start_rows.append(i)

    # combine our results to populate a baseline prediction
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):

            # if(V[i][j] > 5):
            #     V[i][j] = 5
            # else:
            #     pass

            if(V[i][j] == 0):
                # fill it with the expected value...
                baseline_predict = Vexpec + biz_avgs[j] + user_avgs[i]
                V[i][j] = baseline_predict
            else:
                # we already had a real value; don't change it...
                pass

    # normalize our data
    arr_min = np.amin(V)

    const = abs(arr_min) + 0.000001 # avoid zero values
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            V[i][j] += const
            V[i][j] = V[i][j] * normalize

    cold_start_percent = (cold_start_count * 1.0) / len(dataset)

    return V, cold_start_percent, cold_start_rows, const

def split_maxtrix(dataset, t_height=0.25, t_width=0.4):
    """ t_height: test set height, t_width: test set width """
    """ Split our dataset into a 90/10 learning training set """
    """ Withold 40 percent of the ratings for 25 percent of the users """
    """ (a 10 percent trainingset) """
    trainingset = copy.deepcopy(dataset)
    test_size = int(round(t_height * float(len(dataset))))
    num_cols = int(round(t_width * (len(dataset[0]))))

    # build predictions & remove testset from trainingset
    testset = []
    predict_count = 0
    for i in range(test_size):
        row = []
        for j in range(num_cols):
            val = dataset[i][j]
            row.append(val)
            # remove the data from trainingset
            trainingset[i][j] = 0
            if(val > 0): 
                predict_count += 1
            else:
                pass
        # add our row....                
        testset.append(row)

    return trainingset, testset, predict_count

def display_results(testset, predict, is_baseline = False):
    numTest = 0
    totalSum = 0.0
    totalMAE = 0.0 # mean absolute error
    for i in range(len(testset)):
        for j in range(len(testset[0])):
            if(testset[i][j] > 0):
                expected = testset[i][j]
                predicted = predict[i][j]

                totalSum += pow(expected - predicted, 2)
                totalMAE += abs(expected - predicted)
                numTest += 1

    if(is_baseline != True):
        print "RMSE = %f MAE = %f " % (math.sqrt(totalSum/numTest), totalMAE/numTest)
    else:
        print "baseline: RMSE = %f MAE = %f " % (math.sqrt(totalSum/numTest), totalMAE/numTest)

def generic_baseline(dataset, normalize):
    V = np.asarray(dataset)

    # expected value across all reviews
    Vexpec = V.sum() / np.count_nonzero(V)
    row, col = V.shape

    # get user averages; there should always be at least one rating
    cold_start_count = 0
    user_avgs = [0] * len(dataset)
    cold_start_rows = []
    for i in range(len(dataset)):
        nz_count = np.count_nonzero(V[i])
        if(nz_count > 0):
            # needs to consider business average which I don't have.
            user_avgs[i] = (float(V[i].sum()) / np.count_nonzero(V[i])) - Vexpec
        else:
            rand = randint(1,9) / 1000.0
            user_avgs[i] = rand # we know nothing about this user; they had zero test-set ratings
            cold_start_count += 1
            cold_start_rows.append(i)

    # combine our results to populate a baseline prediction
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            if(V[i][j] == 0):
                # fill it with the expected value...
                baseline_predict = Vexpec + user_avgs[i]
                V[i][j] = baseline_predict
            else:
                # we already had a real value; don't change it...
                pass

    # normalize our data
    arr_min = np.amin(V)

    const = abs(arr_min) + 0.000001 # avoid zero values
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            V[i][j] += const
            V[i][j] = V[i][j] * normalize

    cold_start_percent = (cold_start_count * 1.0) / len(dataset)

    return V, cold_start_percent, cold_start_rows, const

def main(args):
    attributes, dataset = user_reviews_full()
    # attributes, dataset = user_reviews_subset()

    t_height = 0.25
    t_width = 0.4
    normalize = 1.0/100.0
    # normalize = 1.0
    latent_factors = 20
    iterations = 100

    trainingset, testset, predict_count = split_maxtrix(dataset, t_height, t_width)
    testset = np.asarray(testset)

    # turn array into numpy array so we can apply their statistical methods
    # V, cold_start_percent, cold_start_rows, const = get_baselines(trainingset, normalize)
    V, cold_start_percent, cold_start_rows, const = get_baselines(trainingset, normalize)

    # generic = copy.deepcopy(trainingset)
    # Vb, bcold_start_percent, bcold_start_rows, bconst = get_baselines(generic, normalize)

    # NMF definition & performance:
    # http://arxiv.org/pdf/1205.3193.pdf <--NMF performed best on sparse data.
    # http://hebb.mit.edu/people/seung/papers/ls-lponm-99.pdf <-- NMF for facial recog
    # http://en.wikipedia.org/wiki/Non-negative_matrix_factorization <--wikipedia
    # 
    # An NMF algorithm: ** (a more simple algorithm)
    # http://hebb.mit.edu/people/seung/papers/nmfconverge.pdf
    # 
    # Contrasting NMF:
    # http://users.ics.aalto.fi/rozyang/preprints/icann2011.pdf
    # http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    # 
    # NMF for collaborative filtering: **
    # (learning from incomplete ratings)
    # http://www.siam.org/meetings/sdm06/proceedings/059zhangs2.pdf

    # convert data to a scipy.sparse.coo_matrix & then to a csr matrix
    # V = coo_matrix(V).tocsr()
    # V = np.arange(0.01,1.01,0.01).reshape(10,10)

    print 'latent factors: %d' % (latent_factors)
    predict = nmf(V, latent_factors, iterations, const, normalize)

    # display baseline results
    # display_results(testset, denormalize(Vb, normalize, bconst), is_baseline=True)

    # display overall results
    display_results(testset, predict)

    # compare predictions to expected...
    # this is a subset of the parent matrix... compare the overlapping values
    # where a value exists in the testset

    # compare to baseline.......



if __name__ == "__main__":
    main(sys.argv)








