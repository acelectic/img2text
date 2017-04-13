

import sys,os,pickle,random
import numpy as np
import tensorflow as tf
import time
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

#d segmented data to well formatted list
for c in training_data:
    for i in range(len(training_data[c])):
        inputData_x.append(training_data[c][i][0])
        label_temp = [0]*65
        label_temp[charMapR[c]] = 1
        inputData_y.append(label_temp)

# add segmented data to well formatted list
inputData_x = np.array(inputData_x)
inputData_y = np.array(inputData_y)
print("total : ",len(inputData_x),len(inputData_y))

# shuffle the data
inputData = list(zip(inputData_x,inputData_y))
random.shuffle(inputData)
inputData_x,inputData_y = zip(*inputData)

# separate training and testing data
separateRatio = 0.20
testing_data_len = int(len(inputData_x) * separateRatio)
trainingData_x = inputData_x[:len(inputData_x)-testing_data_len]
trainingData_y = inputData_y[:len(inputData_y)-testing_data_len]

testingData_x = np.array(inputData_x[len(inputData_x)-testing_data_len:])
testingData_y = np.array(inputData_y[len(inputData_y)-testing_data_len:])

print("training : ",len(trainingData_x),len(trainingData_y))
print("testing  : ",len(testingData_x),len(testingData_y))


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

W_conv3 = weight_variable([3, 3, 32, 64])
b_conv3 = bias_variable([64])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([8*8*64, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, 8*8*64])
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
sess.run(tf.global_variables_initializer())
# error function
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# INIT NNet
# ---------
#################################################################### graph
# Add summary ops to collect data
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)

with tf.name_scope("cost_function") as scope:
    # Minimize error using cross entropy
    # Cross entropy
    #cost_function = -tf.reduce_sum(y_ * tf.log(new_y))
    # Create a summary to monitor the cost function
    tf.summary.scalar("cost_function", cross_entropy)

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()
# training
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# # ready to run
init = tf.global_variables_initializer()
# # Add ops to save and restore all the variables.
saver = tf.train.Saver()
# define session
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.InteractiveSession()
# run session (doesn't start training)
#sess.run(init)
sess.run(tf.global_variables_initializer())

# TRAINING
# --------
# run training
# batch_x = []
# batch_y = []
localtime = time.asctime( time.localtime(time.time()))
print ("Local current time :", localtime)
batch_x = []
batch_y = []

#random
# rlist=[]
# rate_train = int(0.4*len(trainingData_x))
# while len(rlist)<rate_train:
#     tmp = random.randint(0,len(trainingData_x))
#     if tmp not in rlist:
#         rlist.append(tmp)
#
# print(rlist)
#
# for r in range(0,len(trainingData_x)):
#     batch_x.append(trainingData_x[r])
#     batch_y.append(trainingData_y[r])


# print (len(batch_x),len(batch_y))
# print (batch_x,"\n",batch_y)

summary_writer = tf.summary.FileWriter(r"C:\Users\miniBear\Desktop\nn\training\model", sess.graph)
# data_size = len(batch_x)
old_time = time.time()
for i in range(5000):

    batch_x = []
    batch_y = []
    rlist = []
    rate_train = int(0.1 * len(trainingData_x))
    while len(rlist) < rate_train:
        tmp = random.randint(0, len(trainingData_x)-1)
        if tmp not in rlist:
            rlist.append(tmp)

    # print(rlist)

    for r in rlist:
        batch_x.append(trainingData_x[r])
        batch_y.append(trainingData_y[r])

    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:testingData_x, y_: testingData_y, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        print(correct_prediction)

    train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.3})

    summary_str = sess.run(merged_summary_op, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
    summary_writer.add_summary(summary_str, i * len(batch_x) + i)



#print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# for i in range(100):
#     # create batch of 100 random training samples
#     # batch_indices = random.sample(range(0, len(trainingData_x)), 20)
#     batch_x = []
#     batch_y = []
#
#     for r in range(20):
#         batch_x.append(trainingData_x[r])
#         batch_y.append(trainingData_y[r])
#     batch_x = np.array(batch_x)
#     batch_y = np.array(batch_y)
#     print("batch_x SIZE:",len(batch_x))
#     sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
#     # if i % 500==0:
#     #     correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#     #     # accuracy
#     #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     #
#     #     print("Accuracy :", sess.run(accuracy, feed_dict={x: testingData_x, y_: testingData_y,keep_prob: 1.0}))
#
save_path = saver.save(sess,r"C:\Users\miniBear\Desktop\nn\training\model\softmaxNNModel.model")
print("Model saved to file: %s" % save_path)

# TESTING
# -------

# testing
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# # accuracy
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#
# print("Accuracy :",sess.run(accuracy, feed_dict={x: testingData_x, y_: testingData_y}))
#

# Save the variables to disk.
#yop = tf.nn.softmax(tf.matmul(x, W) + b)
#op = sess.run(yop, feed_dict={x: batch_x})
#print (op)
timesec = time.strftime('%H:%M:%S', time.gmtime(int(time.time()-old_time)))
print("total time :",timesec)