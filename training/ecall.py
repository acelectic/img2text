from sklearn.metrics import precision_recall_fscore_support as score
import random
#import tensorflow as tf
import numpy as np
import sklearn as sk
import itertools
import matplotlib.pyplot as plt

pre = ["ก","ข","ค","ค","ง"]
y =    ["ก","ข","ค","ง","ก"]

# pre = [random.randint(0,1) for i in range(100)]
# y = [random.randint(0,1) for i in range(100)]
# pre = sorted(pre)
# y = sorted(y)
# print(pre)
#
#
# map = []
# for i in pre:
#     if i not in map:
#         map.append(i)
#
# map = sorted(map)
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
precision, recall, fscore, support = score(pre, y)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
# temp = sk.metrics.confusion_matrix(y, pre)

# # y_true = np.argmax(test_label,1)
# print( "Precision", sk.metrics.precision_score(y, pre, average=None))
# print ("Recall", sk.metrics.recall_score(y, pre,average=None))
# print ("f1_score", sk.metrics.f1_score(y, pre,average=None))
# print ("confusion_matrix")
# print (sk.metrics.confusion_matrix(y, pre))
# # fpr, tpr, tresholds = sk.metrics.roc_curve(y, pre)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#
# plt.figure()
# plot_confusion_matrix(temp, classes=map,
#                       title='Confusion matrix, without normalization')
#
# # Plot normalized confusion matrix
# # plt.figure()
# # plot_confusion_matrix(new_out, classes=charMap, normalize=True,
# #                       title='Normalized confusion matrix')
#
# plt.show()