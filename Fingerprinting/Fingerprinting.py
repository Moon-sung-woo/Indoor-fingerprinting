import tensorflow as tf
import numpy as np
#---------------------------------------------------------------------------------------
'''
#UJI 데이터 사용
data = np.loadtxt('building2.csv', delimiter=",") #5196 * 529
test = np.loadtxt('building2_test.csv', delimiter=',')#307 * 529

x_test = test[:, 0:-9] #307 * 520
y_test = test[:, 520:523] #307 * 3
floor_y_test = test[:, 522]

x_data = data[:, 0:-9] #5196 * 520
y_data = data[:, 520:523]#5194 * 3 경도 위도 층 내가 예측하고 싶어 하는 것
floor_y_data = data[:, 522]
'''
#Tampere University
data = np.loadtxt('Test_rss_21Aug17.csv', delimiter=",") #5196 * 529
data_floor = np.loadtxt('Test_coordinates_21Aug17.csv', delimiter=",")
test = np.loadtxt('Training_rss_21Aug17.csv', delimiter=',')#307 * 529
test_floor = np.loadtxt('Training_coordinates_21Aug17.csv', delimiter=',')

x_test = test[:, :] #3951, 992
floor_y_test = test_floor[:, 2]

x_data = data[:, :] #5196 * 520
floor_y_data = data_floor[:, 2]

#---------------------------------------------------------------------------------------
'''
#UJI 데이터 사용
input_dim = 520
seq_length = 1
hidden_dim = 256#cell의 수
floor_batch_size = 1
batch_size = 500
learning_rate = 0.005
output_floor = 1
floor_size = int(floor_y_data.max()) + 1#5
encoder_hidden_num1 = 350
encoder_hidden_num2 = 260
'''

#TUT 데이터 사용
input_dim = 992
seq_length = 1
hidden_dim = 256#cell의 수
floor_batch_size = 1
batch_size = 500
learning_rate = 0.005
output_floor = 1
floor_size = int(floor_y_data.max()) + 1
encoder_hidden_num1 = 600
encoder_hidden_num2 = 400

#---------------------------------------------------------------------------------------
X = tf.placeholder(tf.float32, [None, x_data.shape[1]])
Y = tf.placeholder(tf.int32, [None, 2])

in_batch_size = tf.placeholder(shape=[], dtype=tf.int32)#한번에 몇개를 넣을건지
in_drop_out = tf.placeholder(tf.float32, [])

Y_onehot = tf.one_hot(Y, floor_size, dtype=tf.int32)
#---------------------------------------------------------------------------------------
'''
#fully_connected
layer1 = tf.contrib.layers.fully_connected(X, int(1.73768654e+03), activation_fn=tf.nn.elu)
# layer1 = tf.contrib.layers.fully_connected(X, int(hidden_leyers[0]), activation_fn=tf.nn.elu)
drop_1 = tf.nn.dropout(layer1, in_drop_out)
layer2 = tf.contrib.layers.fully_connected(drop_1, int(2.92564023e+03), activation_fn=tf.nn.elu)
# layer2 = tf.contrib.layers.fully_connected(drop_1, int(hidden_leyers[1]), activation_fn=tf.nn.elu)
drop_2 = tf.nn.dropout(layer2, in_drop_out)
layer3 = tf.contrib.layers.fully_connected(drop_1, int(3.04389218e+03), activation_fn=tf.nn.elu)
# layer3 = tf.contrib.layers.fully_connected(drop_2, int(hidden_leyers[2]), activation_fn=tf.nn.elu)
drop_3 = tf.nn.dropout(layer3, in_drop_out)
'''
#---------------------------------------------------------------------------------------

#Auto_encoder
# input -> encode -> decode -> output
W_encode1 = tf.Variable(tf.random_normal([input_dim, encoder_hidden_num1]))#520 -> 350
b_encode1 = tf.Variable(tf.random_normal([encoder_hidden_num1]))
# sigmoid(X * W + b)
encoder1 = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode1), b_encode1))

W_encode2 = tf.Variable(tf.random_normal([encoder_hidden_num1, encoder_hidden_num2]))#350 ->260
b_encode2 = tf.Variable(tf.random_normal([encoder_hidden_num2]))
encoder2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder1, W_encode2), b_encode2))

W_decode1 = tf.Variable(tf.random_normal([encoder_hidden_num2, encoder_hidden_num1]))
b_decode1 = tf.Variable(tf.random_normal([encoder_hidden_num1]))
# 디코더 레이어 구성
decoder1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder2, W_decode1), b_decode1))#260->350
# 히든 레이어의 구성과 특성치을 뽑아내는 알고리즘을 변경하여 다양한 오토인코더를 만들 수 있습니다.
W_decode2 = tf.Variable(tf.random_normal([encoder_hidden_num1, input_dim]))
b_decode2 = tf.Variable(tf.random_normal([input_dim]))
# 디코더 레이어 구성, 이 디코더가 최종 모델
autoEncoder_output = tf.nn.sigmoid(tf.add(tf.matmul(decoder1, W_decode2), b_decode2))#350->520 [None, 520]

#---------------------------------------------------------------------------------------

lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)

initial_state = lstm_cell.zero_state(in_batch_size, dtype=tf.float32)
#rnn shape가 3차원이라 3차원으로 만들어 줘야 한다.

X_re = tf.reshape(autoEncoder_output, [-1, seq_length, autoEncoder_output.shape[1]])

lstm_output , lstm_state = tf.nn.dynamic_rnn(lstm_cell, X_re, initial_state=initial_state,
                                             dtype=tf.float32 , scope='lstm1')
last_outputs = lstm_output[:, -1]

lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
initial_state_2 = lstm_cell.zero_state(in_batch_size, dtype=tf.float32)
lstm_output_2 , lstm_state_2 = tf.nn.dynamic_rnn(lstm_cell_2, lstm_output, initial_state=initial_state_2,
                                                 dtype=tf.float32, scope='lstm2')
last_outputs = lstm_output_2[:, -1]    #5개를 넣어줬을 때 최종 마지막의 위치만을 보기 위해서                                             



output = tf.contrib.layers.fully_connected(last_outputs, floor_size, activation_fn=None)

output = tf.contrib.layers.fully_connected(autoEncoder_output, floor_size, activation_fn=None)
predict = tf.argmax( tf.nn.softmax(output), axis=1)

soft_cross = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y_onehot)
cost = tf.reduce_mean(soft_cross)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

print("test")
import random

def train_data_random_sample(batch_size = 32):
    train_x_data = []
    train_y_data = []
    for i in range(batch_size):
        idx = random.randint(seq_length, x_data.shape[0]) - 1
        train_x_data.extend(  x_data[idx-seq_length+1 : idx+1]  )
        train_y_data.append( floor_y_data[idx] )
    return (train_x_data, train_y_data)

def test_data_random_sample(batch_size = 32):
    test_x_data = []
    test_y_data = []
    for i in range(batch_size):
        idx = random.randint(seq_length, x_test.shape[0]) - 1
        test_x_data.extend(  x_test[idx-seq_length+1 : idx+1]  )
        test_y_data.append( floor_y_test[idx] )
    return (test_x_data, test_y_data)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #학습
    for i in range(10001):
        train_x, train_y = train_data_random_sample(batch_size)
        step_cost, _  = sess.run([cost, optimizer], feed_dict={X: train_x, Y: train_y, in_batch_size: batch_size})
        if(i%200==0):
            print("step : ", i, "cost : ", step_cost)
    #검증
    test_x, test_y = test_data_random_sample(batch_size)
    y_predict = sess.run(predict, feed_dict={X: test_x, in_batch_size: batch_size})
    cnt = 0
    for i, data in enumerate(y_predict):
        cnt += 1 if test_y[i] == data else 0
    print(cnt, (cnt/y_predict.shape[0]) * 100)
