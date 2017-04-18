
# -*- coding: utf-8 -*-

import sys,os,pickle,random
import numpy as np
import tensorflow as tf
import time
import itertools
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score as acc_score

# training data previously stored as a dictionary and written to a pickle dump
training_data = pickle.load( open( "dict.pickle", "rb" ) )
# character image width and height
imgheight, imgwidth = 64, 64

# charMap serves as a lookup table for determining what character is denoted by the index, example 0 is 2. and 27 is z.
#charMap = ['2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','k','m','n','p','r','s','t','v','w','x','y','z']
#charMap = ['ก','ข','ฃ','ค','ฅ','ฆ','ง','จ','ฉ','ช','ซ','ฌ','ญ','ฎ','ฏ','ฐ','ฑ','ฒ','ณ','ด','ต','ถ','ท','ธ','น','บ','ป','ผ','ฝ','พ','ฟ','ภ','ม','ย','ร','ล','ว','ศ','ษ','ส','ห','ฬ','อ','ฮ']
# charMapR serves as a reverse lookup for determining the index of a character
charMap =['เ','โ','ใ','ไ','ก','ข','ฃ','ค','ฅ','ฆ','ง','จ','ฉ','ช','ซ','ฌ','ญ','ฎ','ฏ','ฐ','ฑ','ฒ','ณ','ด','ต','ถ','ท','ธ','น','บ','ป','ผ','ฝ','พ','ฟ','ภ','ม','ย','ร','ล','ว','ศ','ษ','ส','ห','ฬ','์','อ','็','่','้','๊','๋','ั','ำ','ิ','ี','ึ','ื','ุ','ู','ฮ','ฯ','า','ๆ']
charMapR = {}

charMapR['เ'] = 0
charMapR['โ'] = 1
charMapR['ใ'] = 2
charMapR['ไ'] = 3
charMapR['ก'] = 4
charMapR['ข'] = 5
charMapR['ฃ'] = 6
charMapR['ค'] = 7
charMapR['ฅ'] = 8
charMapR['ฆ'] = 9
charMapR['ง'] = 10
charMapR['จ'] = 11
charMapR['ฉ'] = 12
charMapR['ช'] = 13
charMapR['ซ'] = 14
charMapR['ฌ'] = 15
charMapR['ญ'] = 16
charMapR['ฎ'] = 17
charMapR['ฏ'] = 18
charMapR['ฐ'] = 19
charMapR['ฑ'] = 20
charMapR['ฒ'] = 21
charMapR['ณ'] = 22
charMapR['ด'] = 23
charMapR['ต'] = 24
charMapR['ถ'] = 25
charMapR['ท'] = 26
charMapR['ธ'] = 27
charMapR['น'] = 28
charMapR['บ'] = 29
charMapR['ป'] = 30
charMapR['ผ'] = 31
charMapR['ฝ'] = 32
charMapR['พ'] = 33
charMapR['ฟ'] = 34
charMapR['ภ'] = 35
charMapR['ม'] = 36
charMapR['ย'] = 37
charMapR['ร'] = 38
charMapR['ล'] = 39
charMapR['ว'] = 40
charMapR['ศ'] = 41
charMapR['ษ'] = 42
charMapR['ส'] = 43
charMapR['ห'] = 44
charMapR['ฬ'] = 45
charMapR['์'] = 46
charMapR['อ'] = 47
charMapR['็'] = 48
charMapR['่'] = 49
charMapR['้'] = 50
charMapR['๊'] = 51
charMapR['๋'] = 52
charMapR['ั'] = 53
charMapR['ำ'] = 54
charMapR['ิ'] = 55
charMapR['ี'] = 56
charMapR['ึ'] = 57
charMapR['ื'] = 58
charMapR['ุ'] = 59
charMapR['ู'] = 60
charMapR['ฮ'] = 61
charMapR['ฯ'] = 62
charMapR['า'] = 63
charMapR['ๆ'] = 64



# PREPARING DATA
inputData_x = [] # image, which is an input vector , 2925x1
inputData_y = [] # output, denoting the character from the image, 28x1
testData_x = []
testData_y = []

size = 0
r_train = 0

for i in training_data:
    train_index = []
    test_index = []

    size = len(training_data[i])
    r_train = int(0.8 * size)

    train_index = random.sample(range(size),r_train)
    for j in range(size):
        if j not in train_index:
            test_index.append(j)

    for j in train_index:
        inputData_x.append(training_data[i][j][0])
        inputData_y.append(training_data[i][j][1])
    for j in test_index:
        testData_x.append(training_data[i][j][0])
        testData_y.append(training_data[i][j][1])

# print(len(testData_x[0]))



# add segmented data to well formatted list
inputData_x = np.array(inputData_x)
inputData_y = np.array(inputData_y)
testData_x = np.array(testData_x)
testData_y = np.array(testData_y)


# shuffle the data
inputData = list(zip(inputData_x,inputData_y))
testData = list(zip(testData_x,testData_y))
random.shuffle(inputData)
random.shuffle(testData)
inputData_x,inputData_y = zip(*inputData)
testData_x,testData_y = zip(*testData)



# **********************************************************************

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# NEURAL NET
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
# ----------
x = tf.placeholder(tf.float32, shape=[None, 4096])
y_ = tf.placeholder(tf.float32, shape=[None, len(charMap)])
sess = tf.InteractiveSession()

W = tf.Variable(tf.zeros([4096,65]))
b = tf.Variable(tf.zeros([65]))

W_conv1 = weight_variable([5, 5, 1, 16])
b_conv1 = bias_variable([16])

x_image = tf.reshape(x, [-1,32,32,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 32, 64])
b_conv3 = bias_variable([64])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = weight_variable([5, 5, 64, 128])
b_conv4 = bias_variable([128])

h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

W_fc1 = weight_variable([8*8*32, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool4, [-1, 8*8*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 65])
b_fc2 = bias_variable([65])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#################################################################### graph
# Add summary ops to collect data
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)

with tf.name_scope("cost_function") as scope:
    tf.summary.scalar("cost_function", cross_entropy)

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


def train():
    # TRAINING
    # --------

    batch_x = []
    batch_y = []

    summary_writer = tf.summary.FileWriter(r"C:\Users\miniBear\Desktop\nn\training\model", sess.graph)

    total_time = 0
    round_t = 0
    localtime = time.asctime(time.localtime(time.time()))
    print("Local current time :", localtime)

    ##################################################################################################


    random_rate = int(0.2 * len(inputData_x))
    size_data = len(inputData_x) + len(testData_x)
    print("size of data: ", size_data, "shape : [", len(inputData_x[0]), ",", len(inputData_y), "]")
    print("training : ", len(inputData_x))
    print("testing  : ", len(testData_x))
    print("img per round :", random_rate, "of", len(inputData_x))

    max_acc = 0
    max_index = 0

    for i in range(3000):

        nt = time.time()
        batch_x = []
        batch_y = []
        rlist = []

        train_accuracy = accuracy.eval(feed_dict={x: testData_x, y_: testData_y, keep_prob: 1.0})

        if train_accuracy > max_acc:
            max_acc = train_accuracy
            max_index = i

        while len(rlist) < random_rate:
            tmp = random.randint(0, len(inputData_x) - 1)
            if tmp not in rlist:
                rlist.append(tmp)

        # print(rlist)

        for r in rlist:
            batch_x.append(inputData_x[r])
            batch_y.append(inputData_y[r])

        train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.3})

        summary_str = sess.run(merged_summary_op, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
        summary_writer.add_summary(summary_str, i * len(batch_x) + i)

        if i % 100 == 0:
            print("step %d, training accuracy %g" % (i, train_accuracy))

            now_time = time.strftime('%H:%M:%S', time.gmtime(round_t))
            total_time += round_t
            total_t = time.strftime('%H:%M:%S', time.gmtime(total_time))
            print("time (now / total)", now_time, "/", total_t)
            round_t = 0

        round_t += time.time() - nt

    print("best acc:", max_acc, "on step:", max_index)

    save_path = saver.save(sess, r"C:\Users\miniBear\Desktop\nn\training\model\softmaxNNModel.model")
    print("Model saved to file: %s" % save_path)


def avg_list(item):
    return sum(item)/len(item)


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

def solve():
    save_path = saver.restore(sess, r"C:\Users\miniBear\Desktop\nn\training\model\softmaxNNModel.model")
    print("Model load to file: %s" % save_path)

    pre = []
    la = []

    for fol_name in training_data:

        for i in range(len(training_data[fol_name])):
            if i >= 20:
                break

            file = training_data[fol_name][i]
            data = [file[0]]
            label = [file[1]]
            tmp = label[0]

            la_index = tmp.tolist().index(1)
            la.append(la_index)

            out = sess.run(y_conv, feed_dict={x: data, y_: label, keep_prob: 1.0})
            # print(len(data),len(label))
            out = out[0]


            max_i = max(out)
            index = out.tolist().index(max_i)

            pre.append(index)

    print(len(pre),len(la))
    print(pre)
    print(la)

    # print("Precision", sk.metrics.precision_score(la, pre))
    # print("Recall", sk.metrics.recall_score(la, pre, average='metrics'))
    # print("f1_score", sk.metrics.f1_score(la, pre, average='metrics'))
    # print("confusion_matrix")

    precision, recall, fscore, support = score(pre, la)

    acc = acc_score(pre,la).tolist()
    print("acc :",acc)
    cc = support.tolist()
    dd = precision.tolist()
    rr = recall.tolist()
    ff = fscore.tolist()

    for c in range(len(cc)):
        print('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format( charMap[c],dd[c],rr[c],ff[c],cc[c]))

    avg_pre = avg_list(dd)
    avg_recall = avg_list(rr)
    avg_fscore = avg_list(ff)

    print('avg precision\t{:.10f}\navg recall\t{:.10f}\navg fscore\t{:.10f}'.format(avg_pre,avg_recall,avg_fscore))


    size = len(precision)
    temp = sk.metrics.confusion_matrix(la, pre).tolist()
    tee = np.reshape(temp,(size,size))

    for k in range(size):
        if k ==0:
            print("      ",end="")
            for j in charMap:
                print(j," ",end="")
            print()
        tmp = tee[k].tolist()
        print(charMap[k],"\t",tmp)

    # Plot non-normalized confusion matrix
    #
    # plt.figure()
    # plot_confusion_matrix(tee, classes=charMap,
    #                       title='Confusion matrix, without normalization')
    #
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(tee, classes=charMap, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

if __name__ == '__main__':
    # train()
    solve()




