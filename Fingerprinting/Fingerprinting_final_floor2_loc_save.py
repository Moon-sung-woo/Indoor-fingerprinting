import tensorflow as tf
import numpy as np
#---------------------------------------------------------------------------------------
domain = [{'name': 'lr_rate',
          'type': 'continuous',
          'domain': (0.001, 0.01),
           'dimensionality': 1},
            {'name': 'dropout',
          'type': 'continuous',
          'domain': (0.4, 1),
           'dimensionality': 1},
          {'name': 'epoch',
           'type': 'discrete',
           'domain':(5000, 6000, 10000, 50000),
           'dimensionality': 1},
          {'name': 'hidden size',
           'type': 'discrete',
           'domain': (1500, 2500, 3000),
           'dimensionality': 3},
          {'name': 'batch size',
           'type': 'discrete',
           'domain': (500, 1000, 2000),
           'dimensionality': 1}]
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
data_0_floor = np.loadtxt('Test_0_floor.csv', delimiter=",")
data_1_floor = np.loadtxt('Test_1_floor.csv', delimiter=",")
data_2_floor = np.loadtxt('Test_2_floor.csv', delimiter=",")
data_3_floor = np.loadtxt('Test_3_floor.csv', delimiter=",")
data_4_floor = np.loadtxt('Test_4_floor.csv', delimiter=",")

test = np.loadtxt('Training_rss_21Aug17.csv', delimiter=",")#307 * 529
test_floor = np.loadtxt('Training_coordinates_21Aug17.csv', delimiter=",")
test_0_floor = np.loadtxt('Training_0_floor.csv', delimiter=",")
test_1_floor = np.loadtxt('Training_1_floor.csv', delimiter=",")
test_2_floor = np.loadtxt('Training_2_floor.csv', delimiter=",")
test_3_floor = np.loadtxt('Training_3_floor.csv', delimiter=",")
test_4_floor = np.loadtxt('Training_4_floor.csv', delimiter=",")

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

#UJI 데이터 사용
input_dim = 992
seq_length = 1
hidden_dim = 256#cell의 수
floor_batch_size = 1
batch_size = 1000
learning_rate = 0.01
output_floor = 1
floor_size = int(floor_y_data.max()) + 1#5
loc_size = 2 #x,y
encoder_hidden_num1 = 350
encoder_hidden_num2 = 260
drop_out_param = 1.0
EPOCH = 6909
load_model = True
random_num = int(x_2_loc_test.shape[0] * 0.1)
minimum_loss = 999999
minimum_snapshot = []
'''
#TUT 데이터 사용
input_dim = 992 
seq_length = 1
hidden_dim = 256#cell의 수
floor_batch_size = 1
batch_size = 1000
learning_rate = 0.005
output_floor = 1
floor_size = int(floor_y_data.max()) + 1
encoder_hidden_num1 = 600
encoder_hidden_num2 = 400
drop_out_param = 0.9
EPOCH = 5000
'''

import random

def train_data_random_sample(batch_size = 32):
    train_x_data = []
    train_y_data = []
    for i in range(batch_size):
        idx = random.randint(seq_length, x_2_loc_train.shape[0]) - 1
        train_x_data.extend(x_2_loc_train[idx-seq_length+1: idx+1])
        train_y_data.append( y_2_loc_train[idx] )
    return (train_x_data, train_y_data)

def test_data_random_sample(batch_size = 32):
    test_x_data = []
    test_y_data = []
    for i in range(batch_size):
        idx = random.randint(seq_length, x_2_loc_test.shape[0]) - 1
        test_x_data.extend(  x_2_loc_test[idx-seq_length+1 : idx+1]  )
        test_y_data.append( y_2_loc_test[idx] )
    return (test_x_data, test_y_data)
'''
def train_validation(arg):
    #하이퍼 파라미터 추출
    #0행 : learning_rate drop_out_param EPOCH hidden_layer(3개) batch_size
    learning_rate = arg[0,0]
    drop_out_param = arg[0,1]
    EPOCH = int(arg[0,2])
    hidden_leyers = arg[0,3:6]
    batch_size = int(arg[0, 6])


    return 1 - acc
print('옵티마이즈 세팅')
myBopt = BayesianOptimization(f=train_validation, domain=domain, initial_design_numdata=5)
print('옵티마이즈 시작')
myBopt.run_optimization(max_iter=15)
print('옵티마이즈 결과 : ', myBopt.x_opt)
print()
'''
with tf.variable_scope('final_2_loc'):
    # ---------------------------------------------------------------------------------------
    X = tf.placeholder(tf.float32, [None, x_data.shape[1]])
    Y = tf.placeholder(tf.float32, [None, 2])

    in_drop_out = tf.placeholder(tf.float32, [])

    in_batch_size = tf.placeholder(shape=[], dtype=tf.int32)  # 한번에 몇개를 넣을건지

    # ---------------------------------------------------------------------------------------
    # input -> encode -> decode -> output
    layer1 = tf.contrib.layers.fully_connected(X, int(2.46973615e+03), activation_fn=tf.nn.elu)
    # layer1 = tf.contrib.layers.fully_connected(X, int(hidden_leyers[0]), activation_fn=tf.nn.elu)
    drop_1 = tf.nn.dropout(layer1, in_drop_out)
    layer2 = tf.contrib.layers.fully_connected(drop_1, int( 2.92521405e+03), activation_fn=tf.nn.elu)
    # layer2 = tf.contrib.layers.fully_connected(drop_1, int(hidden_leyers[1]), activation_fn=tf.nn.elu)
    drop_2 = tf.nn.dropout(layer2, in_drop_out)
    layer3 = tf.contrib.layers.fully_connected(drop_2, int(1.55666547e+03), activation_fn=tf.nn.elu)
    # layer3 = tf.contrib.layers.fully_connected(drop_2, int(hidden_leyers[2]), activation_fn=tf.nn.elu)
    drop_3 = tf.nn.dropout(layer3, in_drop_out)
    # layer3 = tf.contrib.layers.fully_connected(layer2, floor_size, activation_fn=None)
    # ---------------------------------------------------------------------------------------
    '''
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
    '''
    output = tf.contrib.layers.fully_connected(drop_3, loc_size, activation_fn=None)
    distance_loss = tf.reduce_sum(tf.square(Y - output), axis=1)

    result_loss = tf.sqrt(distance_loss)

    cost = tf.reduce_mean(distance_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

print("test")
saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    if load_model:
        print('load model...')
        ckpt = tf.train.get_checkpoint_state('./final_2_loc')
        saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(1000):
            test_x, test_y = test_data_random_sample(x_2_loc_test.shape[0])
            loss = sess.run([result_loss], feed_dict={X: test_x, Y: test_y, in_batch_size: batch_size, in_drop_out: 1.0})
            mean_loss = np.mean(loss)
            print( np.max(loss), np.min(loss), mean_loss, np.std(loss))
            if minimum_loss > mean_loss:
                minimum_snapshot = [np.max(loss), np.min(loss), mean_loss, np.std(loss)]
                minimum_loss = mean_loss
        with open("final_full_loc_2_best.txt", "w") as f:
            f.write("%s %s %s %s" %(minimum_snapshot[0], minimum_snapshot[1], minimum_snapshot[2], minimum_snapshot[3]))


    else:
        for i in range(EPOCH + 1):
            train_x, train_y = train_data_random_sample(batch_size)
            tr_output, step_cost, _ = sess.run([output, cost, optimizer],
                                               feed_dict={X: train_x, Y: train_y, in_batch_size: batch_size,
                                                          in_drop_out: drop_out_param})
            if (i % 200 == 0):
                print("step : ", i, "cost : ", step_cost)#, 'output : ', tr_output, 'Y_답:', train_y)
    # 검증
        test_x, test_y = test_data_random_sample(x_1_loc_test.shape[0])
        loss = sess.run([result_loss], feed_dict={X: test_x, Y: test_y, in_batch_size: batch_size, in_drop_out: 1.0})
        print( np.max(loss), np.min(loss), np.mean(loss), np.std(loss))
        saver.save(sess, './final_2_loc/final_loc_floor2_test_result.ckpt')