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
data_0_floor = np.loadtxt('Test_0_floor+data.csv', delimiter=",")
data_1_floor = np.loadtxt('Test_1_floor+data.csv', delimiter=",")
data_2_floor = np.loadtxt('Test_2_floor+data.csv', delimiter=",")
data_3_floor = np.loadtxt('Test_3_floor+data.csv', delimiter=",")
data_4_floor = np.loadtxt('Test_4_floor+data.csv', delimiter=",")

test = np.loadtxt('Training_rss_21Aug17.csv', delimiter=',')#307 * 529
test_floor = np.loadtxt('Training_coordinates_21Aug17.csv', delimiter=',')
test_0_floor = np.loadtxt('Training_0_floor+data.csv', delimiter=",")
test_1_floor = np.loadtxt('Training_1_floor+data.csv', delimiter=",")
test_2_floor = np.loadtxt('Training_2_floor+data.csv', delimiter=",")
test_3_floor = np.loadtxt('Training_3_floor+data.csv', delimiter=",")
test_4_floor = np.loadtxt('Training_4_floor+data.csv', delimiter=",")

x_test = test[:, :] #3951, 992
floor_y_test = test_floor[:, 2]
loc_y_test = test_floor[:, 0:2]
x_0_loc_test = test_0_floor[:, 0:-3]
x_1_loc_test = test_1_floor[:, 0:-3]
x_2_loc_test = test_2_floor[:, 0:-3]
x_3_loc_test = test_3_floor[:, 0:-3]
x_4_loc_test = test_4_floor[:, 0:-3]

y_0_loc_test =  test_0_floor[:, -3:-1]
y_1_loc_test =  test_1_floor[:, -3:-1]
y_2_loc_test =  test_2_floor[:, -3:-1]
y_3_loc_test =  test_3_floor[:, -3:-1]
y_4_loc_test =  test_4_floor[:, -3:-1]

x_data = data[:, :] #5196 * 520
floor_y_data = data_floor[:, 2]
loc_y_data = data_floor[:, 0:2]
x_0_loc_train = data_0_floor[:, 0:-3]
x_1_loc_train = data_1_floor[:, 0:-3]
x_2_loc_train = data_2_floor[:, 0:-3]
x_3_loc_train = data_3_floor[:, 0:-3]
x_4_loc_train = data_4_floor[:, 0:-3]

y_0_loc_train =  data_0_floor[:, -3:-1]
y_1_loc_train =  data_1_floor[:, -3:-1]
y_2_loc_train =  data_2_floor[:, -3:-1]
y_3_loc_train =  data_3_floor[:, -3:-1]
y_4_loc_train =  data_4_floor[:, -3:-1]

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
seq_length = 2
hidden_dim = 992#256#cell의 수
floor_batch_size = 1
batch_size = 226#1000
output_floor = 1
floor_size = int(floor_y_data.max()) + 1#5
loc_size = 2 #x,y
encoder_hidden_num1 = 650
encoder_hidden_num2 = 480
drop_out_param = 8.0196379e-01
learning_rate = 1.0000000e-03
EPOCH = 10000

#---------------------------------------------------------------------------------------
X = tf.placeholder(tf.float32, [None, x_0_loc_train.shape[1]])
Y = tf.placeholder(tf.float32, [None, 2])

in_drop_out = tf.placeholder(tf.float32, [])
in_batch_size = tf.placeholder(shape=[], dtype=tf.int32)#한번에 몇개를 넣을건지
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


output = tf.contrib.layers.fully_connected(last_outputs, loc_size, activation_fn=None)

distance_loss = tf.reduce_sum(tf.square(Y - output), axis=1)

result_loss = tf.sqrt(distance_loss)

cost = tf.reduce_mean(distance_loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

print("test")
#---------------------------------------------------------------------------------------
import random

def train_data_random_sample(batch_size = 32):
    train_x_data = []
    train_y_data = []
    for i in range(batch_size):
        idx = random.randint(seq_length, x_0_loc_train.shape[0]) - 1
        train_x_data.extend(  x_0_loc_train[idx-seq_length+1: idx+1]  )
        train_y_data.append( y_0_loc_train[idx] )
    return (train_x_data, train_y_data)

def test_data_random_sample(batch_size = 32):
    test_x_data = []
    test_y_data = []
    for i in range(batch_size):
        idx = random.randint(seq_length, x_0_loc_test.shape[0]) - 1
        test_x_data.extend(  x_0_loc_test[idx-seq_length+1 : idx+1]  )
        test_y_data.append( y_0_loc_test[idx] )
    return (test_x_data, test_y_data)
#---------------------------------------------------------------------------------------
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #학습
    for i in range(EPOCH+1):
        train_x, train_y = train_data_random_sample(batch_size)
        tr_output, step_cost, _ = sess.run([output, cost, optimizer],
                                           feed_dict={X: train_x, Y: train_y, in_batch_size: batch_size, in_drop_out: drop_out_param})
        if (i % 200 == 0):
            print("step : ", i, "cost : ", step_cost),  # 'output : ', tr_output, 'Y_답:', train_y)
    #검증
    test_x, test_y = test_data_random_sample(x_0_loc_test.shape[0])
    loss = sess.run([result_loss], feed_dict={X: test_x, Y: test_y, in_batch_size: batch_size, in_drop_out: 1.0})
    mean_loss = np.mean(loss)
    print(np.max(loss), np.min(loss), mean_loss, np.std(loss))
