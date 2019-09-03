import tensorflow as tf
import numpy as np
from GPyOpt.methods import BayesianOptimization
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

tests=[
    test_0_floor,
    test_1_floor,
    test_2_floor,
    test_3_floor,
    test_4_floor
]

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
hidden_dim = 256#256#cell의 수
floor_batch_size = 1
batch_size = 1000
output_floor = 1
floor_size = int(floor_y_data.max()) + 1#5
loc_size = 2 #x,y
encoder_hidden_num1 = 350
encoder_hidden_num2 = 260
drop_out_param = 8.0196379e-01
learning_rate = 1.0000000e-03
EPOCH = 10000
#---------------------------------------------------------------------------------------
domain = [{'name': 'lr_rate',
          'type': 'continuous',
          'domain': (0.001, 0.01),
           'dimensionality': 1},
            {'name': 'dropout',
          'type': 'continuous',
          'domain': (0.5, 1),
           'dimensionality': 1},
          {'name': 'seq length',
           'type': 'discrete',
           'domain':(2, 3,4, 5,6, 7),
           'dimensionality': 1},
          {'name': 'epoich',
           'type': 'continuous',
           'domain':(1000, 10000),
           'dimensionality': 1}]
import random

def train_data_random_sample(batch_size = 32, seq_length=1):
    train_x_data = []
    train_y_data = []
    for i in range(batch_size):
        idx = random.randint(1, x_0_loc_train.shape[0]) - 1
        for j in range(seq_length):
            train_x_data.extend(x_0_loc_train[idx:idx+1])
        train_y_data.append(y_0_loc_train[idx])

    return (train_x_data, train_y_data)

def test_data_sample(x_test,y_test, seq_length=1):
    test_x_data = []
    test_y_data = []
    for i, x in enumerate(x_test):
        for j in range(seq_length):
            test_x_data.extend( [x])
        #test_x_data.extend( np.random.choice(x_0_loc_test, seq_length-1))
        test_y_data.append( y_test[i] )
    return (test_x_data, test_y_data)

def train_data_random_sample1(batch_size = 32, seq_length=1):
    train_x_data = []
    train_y_data = []
    for i in range(batch_size):
        idx = random.randint(1, x_1_loc_train.shape[0]) - 1
        for j in range(seq_length):
            train_x_data.extend(x_1_loc_train[idx:idx+1])
        train_y_data.append(y_1_loc_train[idx])

    return (train_x_data, train_y_data)


def train_data_random_sample2(batch_size = 32, seq_length=1):
    train_x_data = []
    train_y_data = []
    for i in range(batch_size):
        idx = random.randint(1, x_2_loc_train.shape[0]) - 1
        for j in range(seq_length):
            train_x_data.extend(x_2_loc_train[idx:idx+1])
        train_y_data.append(y_2_loc_train[idx])

    return (train_x_data, train_y_data)

def train_data_random_sample3(batch_size = 32, seq_length=1):
    train_x_data = []
    train_y_data = []
    for i in range(batch_size):
        idx = random.randint(1, x_3_loc_train.shape[0]) - 1
        for j in range(seq_length):
            train_x_data.extend(x_3_loc_train[idx:idx+1])
        train_y_data.append(y_3_loc_train[idx])

    return (train_x_data, train_y_data)


def train_data_random_sample4(batch_size = 32, seq_length=1):
    train_x_data = []
    train_y_data = []
    for i in range(batch_size):
        idx = random.randint(1, x_4_loc_train.shape[0]) - 1
        for j in range(seq_length):
            train_x_data.extend(x_4_loc_train[idx:idx+1])
        train_y_data.append(y_4_loc_train[idx])

    return (train_x_data, train_y_data)

def test_data_random_sample(batch_size = 32, seq_length=1):
    test_x_data = []
    test_y_data = []
    for i in range(batch_size):
        idx = random.randint(1, x_0_loc_test.shape[0]) - 1
        for j in range(seq_length):
            test_x_data.extend( x_0_loc_test[idx:idx+1])
        #test_x_data.extend( np.random.choice(x_0_loc_test, seq_length-1))
        test_y_data.append( y_0_loc_test[idx] )
    return (test_x_data, test_y_data)


#--------------------------------------------------------------------------------------
test_x_datas=[
    x_0_loc_test,
    x_1_loc_test,
    x_2_loc_test,
    x_3_loc_test,
    x_4_loc_test
]
test_y_datas=[
    y_0_loc_test,
    y_1_loc_test,
    y_2_loc_test,
    y_3_loc_test,
    y_4_loc_test
]

train_x_datas=[
    x_0_loc_train,
    x_1_loc_train,
    x_2_loc_train,
    x_3_loc_train,
    x_4_loc_train
]
train_y_datas=[
    y_0_loc_train,
    y_1_loc_train,
    y_2_loc_train,
    y_3_loc_train,
    y_4_loc_train
]

train_sample_funcs = [ train_data_random_sample, train_data_random_sample1,train_data_random_sample2,train_data_random_sample3,train_data_random_sample4 ]

minimum_loss_list = []
minimum_snapshot_list = []
#----------------------------------------------------------------------------------



minimum_loss = 999999
minimum_loss1 = 999999
minimum_loss2 = 999999
minimum_loss3 = 999999
minimum_loss4 = 999999
minimum_snapshot = []
minimum_snapshot_1 = []
minimum_snapshot_2 = []
minimum_snapshot_3 = []
minimum_snapshot_4 = []

def train_validation(arg):
    global batch_size
    global train_xs
    global train_ys
    global test_xs
    global test_ys
    global sample_func

    learning_rate = arg[0, 0]
    drop_out_param = arg[0, 1]
    seq_length = int(arg[0, 2])
    EPOCH = int(arg[0, 3])
    tf.reset_default_graph()
    #---------------------------------------------------------------------------------------
    X = tf.placeholder(tf.float32, [None, train_xs.shape[1]])
    in_filter = tf.placeholder(tf.float32, [None, train_xs.shape[1]])

    Y = tf.placeholder(tf.float32, [None, 2])

    in_drop_out = tf.placeholder(tf.float32, [])
    in_batch_size = tf.placeholder(shape=[], dtype=tf.int32)#한번에 몇개를 넣을건지
    in_filter_size = tf.placeholder(shape=[], dtype=tf.int32)  # 한번에 몇개를 넣을건지
    #---------------------------------------------------------------------------------------

    W_filter = tf.Variable(tf.random_normal([input_dim]))#520 -> 350
    b_filter = tf.Variable(tf.random_normal([input_dim]))

    matrix_b = tf.tile( b_filter, [in_filter_size] )
    matrix_b = tf.reshape( matrix_b, [-1, input_dim] )

    filter1 = tf.add(tf.multiply(X, W_filter), tf.multiply(in_filter, matrix_b))

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
    encoder1 = tf.nn.sigmoid(tf.add(tf.matmul(filter1, W_encode1), b_encode1))

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
    #drop_out = tf.nn.dropout(autoEncoder_output, in_drop_out)

    #---------------------------------------------------------------------------------------

    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim)
    drops = [tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=drop_out_param)]
    re_lstm_cell = tf.contrib.rnn.MultiRNNCell(drops)

    initial_state = re_lstm_cell.zero_state(in_batch_size, dtype=tf.float32)

    # rnn shape가 3차원이라 3차원으로 만들어 줘야 한다.
    X_re = tf.reshape(autoEncoder_output, [-1, seq_length, autoEncoder_output.shape[1]])
    lstm_output, lstm_state = tf.nn.dynamic_rnn(lstm_cell, X_re, initial_state=initial_state[0],
                                                dtype=tf.float32, scope='lstm1')


    #last_outputs = lstm_output[:, -1]

    #lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim)

    initial_state_2 = lstm_cell_2.zero_state(in_batch_size, dtype=tf.float32)
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

    #---------------------------------------------------------------------------------------
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        filter_size = batch_size*seq_length
        #학습
        for i in range(EPOCH+1):
            train_x, train_y = sample_func(batch_size, seq_length)
            train_filter = [np.where(x == 0, 0, 1) for x in train_x]

            tr_output, step_cost, _ = sess.run([output, cost, optimizer],
                                               feed_dict={X: train_x, Y: train_y, in_batch_size: batch_size,
                                                          in_drop_out: drop_out_param, in_filter:train_filter,
                                                          in_filter_size:filter_size })
            if (i % 200 == 0):
                print("step : ", i, "cost : ", step_cost),  # 'output : ', tr_output, 'Y_답:', train_y)
        #검증

        verification = test_xs
        filter_size = verification.shape[0] * seq_length

        test_x, test_y = test_data_sample(verification, test_ys, seq_length)
        test_filter = [np.where(x == 0, 0, 1) for x in test_x]
        loss = sess.run([result_loss], feed_dict={X: test_x, Y: test_y, in_batch_size: verification.shape[0],
                                                  in_drop_out: 1.0, in_filter: test_filter,
                                                  in_filter_size: filter_size})
        mean_loss = np.mean(loss)
        print(np.max(loss), np.min(loss), mean_loss, np.std(loss))

        global minimum_loss

        if minimum_loss > mean_loss:
            global minimum_snapshot
            minimum_snapshot = [np.max(loss), np.min(loss), mean_loss, np.std(loss)]
            minimum_loss = mean_loss

    return mean_loss

for floor_step in range(len(train_x_datas)):
    train_xs = train_x_datas[floor_step]
    train_ys = train_y_datas[floor_step]
    test_xs = test_x_datas[floor_step]
    test_ys = test_y_datas[floor_step]
    sample_func = train_sample_funcs[floor_step]
    minimum_loss = 999999
    minimum_snapshot = []
    print('옵티마이즈 세팅')
    myBopt = BayesianOptimization(f=train_validation, domain=domain, initial_design_numdata=6)
    print('옵티마이즈 시작')
    myBopt.run_optimization(max_iter=6)
    print('%s층 옵티마이즈 결과 : ' % floor_step, myBopt.x_opt)
    print('%s층 최적의 하이퍼 파라미터의 결과 : ' % floor_step, minimum_snapshot)
    with open('optimize_result_%s_floor.txt' % floor_step, 'w') as f:
        for param in myBopt.x_opt:
            f.write(str(param)+" ")
    with open('min_optimize_result_%s_floor.txt' % floor_step, 'w') as f:
        for param in minimum_snapshot:
            f.write(str(param)+" ")
    #minimum_loss_list.append(minimum_loss)
    #minimum_snapshot_list.append(minimum_snapshot)



