

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

print(len(testData_x[0]))



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

# TRAINING
# --------

batch_x = []
batch_y = []


summary_writer = tf.summary.FileWriter(r"C:\Users\miniBear\Desktop\nn\training\model", sess.graph)

total_time = 0
round_t = 0
localtime = time.asctime( time.localtime(time.time()))
print ("Local current time :", localtime)

##################################################################################################


random_rate = int(0.5*len(inputData_x))
size_data = len(inputData_x)+len(testData_x)
print("size of data: ",size_data,"shape : [",len(inputData_x[0]),",",len(inputData_y),"]")
print("training : ",len(inputData_x))
print("testing  : ",len(testData_x))
print("img per round :",random_rate,"of",len(inputData_x))

max_acc = 0
max_index = 0

for i in range(1000):

    nt = time.time()
    batch_x = []
    batch_y = []
    rlist = []

    train_accuracy = accuracy.eval(feed_dict={x: testData_x, y_: testData_y, keep_prob: 1.0})

    if train_accuracy > max_acc:
        max_acc = train_accuracy
        max_index = i


    while len(rlist) < random_rate:
        tmp = random.randint(0, len(inputData_x)-1)
        if tmp not in rlist:
            rlist.append(tmp)

    # print(rlist)

    for r in rlist:
        batch_x.append(inputData_x[r])
        batch_y.append(inputData_y[r])

    train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.3})

    summary_str = sess.run(merged_summary_op, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
    summary_writer.add_summary(summary_str, i * len(batch_x) + i)

    if i%100 == 0:

        print("step %d, training accuracy %g"%(i, train_accuracy))

        now_time = time.strftime('%H:%M:%S', time.gmtime(round_t))
        total_time += round_t
        total_t = time.strftime('%H:%M:%S', time.gmtime(total_time))
        print("time (now / total)",now_time,"/", total_t)
        round_t = 0

    round_t += time.time() - nt


print ("best acc:", max_acc ,"on step:",max_index)

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

#yop = tf.nn.softmax(tf.matmul(x, W) + b)
#op = sess.run(yop, feed_dict={x: batch_x})
#print (op)

