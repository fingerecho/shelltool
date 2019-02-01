from sklearn import datasets

from __init__ import write_log

log_file = "_test_cnn_9_4.log"

diabetes = datasets.load_diabetes()

write_log(str(len(diabetes.data[0])), file=log_file)
