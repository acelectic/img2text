from sklearn.metrics import precision_recall_fscore_support as score
import random
#import tensorflow as tf
import numpy as np
import sklearn as sk
pre = ["ก","ข","ค","ค","ง"]
y =    ["ก","ข","ค","ง","ก"]
#
# precision, recall, fscore, support = score(y_test, predicted)
#
# print('precision: {}'.format(precision))
# print('recall: {}'.format(recall))
# print('fscore: {}'.format(fscore))
# print('support: {}'.format(support))

#
# a = [random.randint(0,5) for i in range(100)]
# b = [random.randint(0,5) for i in range(100)]
#
# x = tf.placeholder(tf.float32, shape=[None,100])
# y = tf.placeholder(tf.float32, shape=[None,100])
#
# sess =tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
#
# c,d = sess.run(tf.metrics.recall(x,y))
#
#
# print(a)
# print(b)
# print(c,d)
# #
# precision, recall, fscore, support = score(a, b)
#
# print('precision: {}'.format(precision))
# print('recall: {}'.format(recall))
# print('fscore: {}'.format(fscore))
# print('support: {}'.format(support))


# y_true = np.argmax(test_label,1)
print( "Precision", sk.metrics.precision_score(y, pre, average=None))
print ("Recall", sk.metrics.recall_score(y, pre,average=None))
print ("f1_score", sk.metrics.f1_score(y, pre,average=None))
print ("confusion_matrix")
print (sk.metrics.confusion_matrix(y, pre))
# fpr, tpr, tresholds = sk.metrics.roc_curve(y, pre)