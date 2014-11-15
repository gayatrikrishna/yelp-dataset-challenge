from lib import arff 

def get_arguments(args):
  if len(args) == 5:
    train_set_file = str(args[1])
    n = int(args[2])
    l = float(args[3])
    e = int(args[4])
  else:
    print 'usage neuralnet.py <data-set-file> n l e'
    exit(0)

  return train_set_file, n, l, e

def count_data_with_attr(data, attr):
  count = 0
  for row in data:
    for item in row:
      print item
      print attr
      exit(0)
      if str(item).decode("utf-8") == str(attr).decode("utf-8"):
        count = count + 1
        break
      else:
        pass

  print 'the count:' + str(count)
  return count

def dump_splits(splits):
  print '>>>>>>>>          *** Dumping Splits ***          <<<<<<<<'

  for x in splits:
    type_of = str(type(splits[x]))
    if type_of == "<class 'decision_tree.NominalCandidateSplit'>":
      print splits[x] 
    elif type_of == "<class 'decision_tree.NumericCandidateSplit'>":
      print splits[x] 
    else:
      pass

  exit(0)

# turn attributes into something manageable
def get_attributes(attr_data):
  attributes = {}
  index = 0

  for key in attr_data:
    attribute_data = {'name':key[0], 'index': index, 'options': key[1] }

    if key[0] == 'class':
      attribute_data['type'] = 'class' 
    elif type(key[1]) == unicode or type(key[1]) == str:
      attribute_data['type'] = 'numeric'
    else:
      attribute_data['type'] = 'nominal'

    attributes[key[0]] = attribute_data
    index = index + 1

  return attributes
  
def homogenous_check(data, negative, positive):
  # are all examples one type? (all pos / all neg)
  positive_found = False
  negative_found = False
  
  for row in data:
    if row[-1] == negative:
      negative_found = True
    elif row[-1] == positive:
      positive_found = True
    else:
      pass
    if negative_found and positive_found:
      return False, None

  if negative_found and not positive_found:
    # data[0][-1] is the class
    return True, negative
  elif positive_found and not negative_found:
    return True, positive
  else:
    # this should never happen
    return False, None

# call the same way as homogenous_check
def get_class_counts(data, negative, positive):
  # are all examples one type? (all pos / all neg)
  positive_count = 0
  negative_count = 0

  for row in data:
    if row[-1] == negative:
      negative_count += 1
    elif row[-1] == positive:
      positive_count += 1
    else:
      pass

  return positive_count, negative_count


def load_data(path = None):
  if not path:
    raise ValueError('A valid path to training data must be be specified')
    # fp = open('examples/heart_test.arff')
    # fp = open('examples/heart_train.arff')
    # fp = open('examples/diabetes_test.arff')
    # fp = open('examples/diabetes_train.arff')
  else:
    fp = open(path)
    
  data = arff.load(fp)
  return data;


def dump_attributes(data):
  #attributes is a python list - stepping, slicing, etc.
  #bracket = list
  #paren = tuple
  #brace = dict
  attributes = data['attributes']
  # attribute types: NUMERIC, REAL, INTEGER, STRING, and NOMINAL
  #  NUMERIC, REAL, INTEGER == convert to int
  # list == NOMINAL
  
  # Class items
  # print attributes[-1][1][0]
  # print attributes[-1][1][1]

  # blah = [1,2,3]
  # blah2 = (1,2,3)
  # print attributes[0]
  # print attributes[1]
  # print attributes[2]
  # print attributes[3]

  # print blah
  # print blah2
  # return 
  for key in attributes:
    if type(key[0]) is unicode and type(key[1]) is unicode:
      print key[0] + ", " + key[1]
    if type(key[0]) is unicode and type(key[1]) is list:
      options = ''
      count = False
      curr_list = key[1]
      for opt in curr_list:
        if not count:
          options += opt
        else:
          options += ", " + opt
        count = True
      # if key[0] == 'class':
      #  print 'found class!' 
      # else:
      #   pass
      print key[0] + ", nominal: " + options
    # if type([]) is list -- list is array
    # if type({}) is dict -- dict is object
    # if type('') is str or unicode
    # if type(0) is int