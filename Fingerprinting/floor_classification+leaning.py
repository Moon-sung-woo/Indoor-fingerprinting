import tensorflow as tf
import Read_do as rd
import numpy as np
import math
import os
import shutil
from sklearn.cluster import AffinityPropagation


def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


def distanceNumpy(a, b):
    sum_dist = 0

    for i in range(len(a)):
        sum_dist = sum_dist + np.sqrt(np.sum(np.power(a[i]-b[i], 2)))

    return sum_dist/len(a)


def distanceLine(a, b):
    return  np.sqrt(np.sum(np.power(a-b, 2)))


def accurateTest(x, keep_prob, sess,h_fc2, x_test, y_test, batch_size):

    error_map = h_fc2.eval(session=sess,
                           feed_dict={x: x_test[0:(batch_size-1)],
                                      keep_prob: 1.0})
    error_map = error_map - y_test[0:(batch_size-1)]
    error_total = 0
    for error_arr in error_map:
        error_total += math.sqrt(pow(error_arr[0], 2) + pow(error_arr[1], 2))

    error_result = error_total / len(error_map)

    return error_result


def bdist(a, b, sigma, eps, th, lth=-85, div=10):
    diff = a - b

    proba = 1/(np.sqrt(2*np.pi)*sigma)*np.exp( \
        -np.power(diff, 2)/(2.0*sigma**2))

    proba[np.isnan(proba)] = eps
    proba[proba < th] = eps
    proba = np.log(proba)
    if a.ndim == 2:
        cost = np.sum(proba, axis=1)
    else:
        cost = np.sum(proba)

    inv = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        aa = np.logical_and(~np.isnan(a[i]), np.isnan(b))
        bb = np.logical_and(np.isnan(a[i]), ~np.isnan(b))

        nfound = np.concatenate((a[i,aa], b[bb]))
        for v in nfound[nfound > lth]:
            inv[i] += v - lth

    inv /= div
    cost -= inv

    return cost


def cluster_subset_affinityprop(clusters, labels, X_test, cuse = 10):
    subset = np.zeros(labels.shape[0]).astype(np.bool)

    d = bdist(clusters, X_test, 5, 1e-3, 1e-25)
    idx = np.argsort(d)[::-1]

    cused = 0
    for c in idx[:cuse]:       # 인접한 n개 클러스터 사용
        cused += 1
        subset = np.logical_or(subset, c == labels)
        if len(np.where(subset == True)[0]) > 100:
            break

    print(idx[0:cused], ' 번 클러스터 사용')
    return (subset, idx[0])


def dnn_learning_for_this(labeling_x_train, labeling_y_train, labeling_x_test, labeling_y_test, dic_name):
    path = './' + dic_name + '/'
    save_name = 'model_save.ckpt'

    X_test = labeling_x_test
    Y_test = labeling_y_test
    X_train = labeling_x_train
    Y_train = labeling_y_train

    tf.set_random_seed(777)  # reproducibility

    DIMEN = labeling_x_train.shape[1]
    LABEL_DIMEN = labeling_y_train.shape[1]

    X = tf.placeholder("float", [None, DIMEN])
    Y = tf.placeholder("float", [None, LABEL_DIMEN])

    dropout_rate = tf.placeholder(tf.float32)

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

        W1 = weight_variable([DIMEN, DIMEN * 4], name='w1')
        W2 = weight_variable([DIMEN * 4, DIMEN * 4], name='w2')
        # W3 = weight_variable([DIMEN * 3, DIMEN * 3], name='w3')
        W5 = weight_variable([DIMEN * 4, LABEL_DIMEN], name='w4')

        b1 = bias_variable([DIMEN * 4], name='b1')
        b2 = bias_variable([DIMEN * 4], name='b2')
        # b3 = bias_variable([DIMEN * 3], name='b3')
        b5 = bias_variable([LABEL_DIMEN], name='b4')

        _L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1), name='dL1')
        L1 = tf.nn.dropout(_L1, dropout_rate, name='L1')
        _L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2), name='dL2')
        L2 = tf.nn.dropout(_L2, dropout_rate, name='L2')
        # _L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3), name='dL3')
        # L3 = tf.nn.dropout(_L3, dropout_rate, name='L3')
        param_list = [W1, W2, W5, b1, b2, b5]
        saver = tf.train.Saver(param_list)

        hypothesis = tf.matmul(L2, W5) + b5

        cost = tf.reduce_mean(tf.pow(hypothesis - Y, 2))  # Regression

        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    best_cost = 999999
    best_accuracy = 0
    best_error = 999999
    best_test_error = 999999

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    with sess:
        sess.run(init)

        if not os.path.isdir(dic_name):
            os.mkdir(dic_name)

        if not os.path.isdir(path):
            os.mkdir(path)

        for step in range(100000):

            sess.run(optimizer,
                     feed_dict={X: X_train, Y: Y_train, dropout_rate: 0.7})

            error_map = hypothesis.eval(session=sess,
                                        feed_dict={X: X_test,
                                                   dropout_rate: 1.0});

            error_result = distanceNumpy(error_map, Y_test)

            if (best_error > error_result):
                # print('error changed : ', best_error ,' -> ', error_result)
                best_error = error_result

            test_means_y = hypothesis.eval(session=sess, feed_dict={X: labeling_x_test, dropout_rate: 1.0})
            test_error = distanceNumpy(test_means_y, labeling_y_test)

            currentCost = cost.eval({X: X_train, Y: Y_train, dropout_rate: 1})

            if best_test_error > test_error:
                best_test_error = test_error
                best_test_error_step = step
                best_test_error_cost = currentCost

            if currentCost < best_cost:
                best_cost = currentCost
                saver.save(sess, path + save_name, global_step=step)
                correct_prediction = tf.equal(tf.round(hypothesis), Y)  # regression
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                currentAccuracy = accuracy.eval({X: X_test, Y: Y_test, dropout_rate: 1})

                if best_accuracy < currentAccuracy:
                    best_accuracy = currentAccuracy

                print("\n\n-------------------------------------\n")
                print("Train Data Shape : ", X_train.shape)
                print("Test Data Shape : ", labeling_x_test.shape)
                print("DNN Current Step : ", step, " Current Cost : ", str(currentCost))
                print("DNN Best Cost:", best_cost)
                if step != 0:
                    means_y = hypothesis.eval(session=sess, feed_dict={X: X_test[:10], dropout_rate: 1.0})
                    for i in range(len(means_y)):
                        print(i, 'test_result :', means_y[i], ' / y_answer : ', Y_test[i], ' / error : ',
                              distanceLine(means_y[i], Y_test[i]))
                    print("error average : ", distanceNumpy(means_y, Y_test), 'm')

                    test_means_y = hypothesis.eval(session=sess, feed_dict={X: labeling_x_test, dropout_rate: 1.0})
                    print("test error average : ", distanceNumpy(test_means_y, labeling_y_test), 'm')
                print("best error average (m) : ", best_error)
                print("best test error average (m) : ", best_test_error)
                print("best test error step : ", best_test_error_step)
                print("best test error cost : ", best_test_error_cost)
                print("\n\n-------------------------------------\n")

                # if currentCost < 2.5:
                #     break

        sess.close()

        print("DNN Best Accuracy:", best_accuracy)
        print("DNN Best loss %g" % best_cost)
        print("DNN Best error (m) %g" % best_error)

        logFIle = open(path + 'log.txt', 'w')

        out = " step : " + str(step) + ", loss : " \
              + (str)(best_cost) \
              + "\ntrain data size : " + str(X_train.shape) \
              + "\ntest data size : " + str(labeling_x_test.shape) \
              + "\nbest train error (m) : " + str(best_error) \
              + "\nbest test error(m) : " + str(best_test_error) \
              + "\nbest test error cost : " + str(best_test_error_cost) \
              + "\nbest test error step : " + str(best_test_error_step)

        logFIle.write(str(out))
        logFIle.flush()
        logFIle.close()


def dnn_learning(DIMEN, train_data, validation_data, save_file_name, LABEL_DIMEN = 3, batch_size = 4):

    # DIMEN = 529

    tf.set_random_seed(777)  # reproducibility

    learning_rate = 0.01

    x_input = list()
    y_input = list()

    for i in range(batch_size):
        # train_xy = ru.getXYListClassificationBatch(train_ap_list,cell_width, i)
        # train_xy = rd.getXYListClassificationBatch(train_ap_list, width, height, cell_width, i)
        train_xy = rd.getXYListRegressionBatch2DRaw(train_data, i, batch_size)

        x_input.append(np.array(train_xy[0]))
        y_input.append(np.array(train_xy[1]))

    # test_xy = ru.getXYListClassification(validation_ap_list,cell_width)
    # test_xy = rd.getXYListClassification(validation_ap_list, width, height, cell_width)
    test_xy = rd.getXYListRegression2DRaw(train_data)

    x_test = np.array(np.array(test_xy[0]))
    y_test = np.array(np.array(test_xy[1]))

    X = tf.placeholder("float", [None, DIMEN])
    Y = tf.placeholder("float", [None, LABEL_DIMEN])
    dropout_rate = tf.placeholder(tf.float32)

    with tf.variable_scope(tf.get_variable_scope()) as scope:
        W1 = weight_variable([DIMEN, 8192])
        W2 = weight_variable([8192, 8192])
        W3 = weight_variable([8192, 8192])
        W5 = weight_variable([8192, LABEL_DIMEN])

        b1 = bias_variable([8192])
        b2 = bias_variable([8192])
        b3 = bias_variable([8192])
        b5 = bias_variable([LABEL_DIMEN])

        _L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
        L1 = tf.nn.dropout(_L1, dropout_rate)
        _L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
        L2 = tf.nn.dropout(_L2, dropout_rate)
        _L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))
        L3 = tf.nn.dropout(_L3, dropout_rate)

        hypothesis = tf.matmul(L3, W5) + b5

        # Cost function
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y)) # OHE
        cost = tf.reduce_mean(tf.pow(hypothesis - Y, 2))  # Regression

        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

        tf.get_variable_scope().reuse_variables()


    init = tf.global_variables_initializer()

    best_cost = 999999
    train_end_count = 0
    best_test_cost = 999999
    best_accuracy = 0
    best_error = 999999

    config = tf.ConfigProto()
    sess = tf.Session(config=config)

    finalRepresentations = []

    step = 0

    with sess:

        path = './' + save_file_name

        if os.path.exists(path):
            saver = tf.train.Saver()
            import_saver = tf.train.import_meta_graph(path + '/dnn_save.ckpt.meta')
            import_saver.restore(sess, tf.train.latest_checkpoint(path))
            print("저장된 모델 불러오기")
        else:
            sess.run(init)
            saver = tf.train.Saver()

        batch = -1
        for step in range(500000):
            if step % 1000 == 0:
                saver.save(sess, path + '/dnn_save.ckpt')

                # correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)) # OHE
                correct_prediction = tf.equal(tf.round(hypothesis), Y)  # regression
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                currentCost = cost.eval(
                    {X: x_input[batch % batch_size], Y: y_input[batch % batch_size], dropout_rate: 1})
                testCost = cost.eval({X: x_test, Y: y_test, dropout_rate: 1})
                currentAccuracy = accuracy.eval({X: x_test, Y: y_test, dropout_rate: 1})

                if best_accuracy < currentAccuracy:
                    best_accuracy = currentAccuracy

                if (currentCost < best_cost):
                    best_cost = currentCost
                    best_test_cost = testCost
                    train_end_count = 0
                else:
                    train_end_count = train_end_count + 1

                # print("\n\n-------------------------------------\n")
                # print("DNN Current Step : ", step, " Current Cost : ", currentCost)
                # print("DNN Test Cost : ", testCost)
                # print("DNN Current Accuracy : ", currentAccuracy)
                # print("DNN Best Test Cost:", best_test_cost)
                # print("hypothesis sample : ", hypothesis.eval(session=sess, feed_dict={X: x_input[batch%batch_size], dropout_rate: 1.0}))
                # print("answer : ", y_input[batch % 6])
                # print("\n\n-------------------------------------\n")
                #
                print("\n\n-------------------------------------\n")
                print("DNN Current Step : ", step, " Current Cost : ", str(currentCost))
                print("DNN Current Accuracy:", currentAccuracy)
                print("DNN Test Cost : ", str(testCost / len(validation_data)))
                print("DNN Best Test Cost:", str(best_test_cost / len(validation_data)))
                print("hypothesis sample : ", hypothesis.eval(session=sess,
                                                                        feed_dict={X: x_test[0:6],
                                                                                   dropout_rate: 1.0}))
                print("answer : ", y_test[0:6])

                print("best error (m) : ", best_error)
                print("\n\n-------------------------------------\n")

                # finalRepresentations.append(hypothesis.eval(session=sess, feed_dict={X: x_input[batch], dropout_rate: 1.0}))
                batch = batch + 1

                if train_end_count > 30:
                    break

            sess.run(optimizer, feed_dict={X: x_input[batch % batch_size], Y: y_input[batch % batch_size], dropout_rate: 0.7})

            error_map = hypothesis.eval(session=sess,
                                        feed_dict={X: x_test[0:(batch_size-1)],
                                                   dropout_rate: 1.0});
            error_map = error_map - y_test[0:(batch_size-1)]

            error_total = 0
            for error_arr in error_map:
                error_total += math.sqrt(pow(error_arr[0], 2) + pow(error_arr[1], 2))

            error_result = error_total / len(error_map)

            if (best_error > error_result):
                best_error = error_result
                best_hypothesis = hypothesis

        print("DNN Best Accuracy:", best_accuracy)
        print("DNN Best loss %g" % best_test_cost)

        print("DNN Best error (m) %g" % best_error)

        out = "[ loss : " \
              + (str)(best_test_cost) \
              + ", accuracy : " + (str)(best_accuracy) \
              + ", error (m) : " + (str)(best_error) + "]\n"

        drop_out_test_arr = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        for index in range(len(drop_out_test_arr)):

            DIMEN = 992
            SHAPE = 1024

            validation_ap_list2 = rd.readFile('Test_rss_21Aug17.csv', DIMEN, SHAPE, drop_out_test_arr[index])
            test_xy = rd.getXYListRegression2DRaw(validation_ap_list2)

            x_test = np.array(np.array(test_xy[0]))
            y_test = np.array(np.array(test_xy[1]))

            result = accurateTest(X, dropout_rate, sess, best_hypothesis, x_test, y_test, batch_size)
            out = out + "[ env_error_rate : " + str(drop_out_test_arr[index]) + ", error (m) : " + str(result) + "]\n"

        sess.close()
        shutil.rmtree(path)
        return out


def load_data(train_data, validation_data):

    # read training data
    X_train = list()
    y_train = list()
    Xz_train = list()
    yz_train = list()

    for data in train_data:

        if data.POINT_Z == 0:
            X_train.append(data.AP_ARR)
            y_train.append([data.POINT_X, data.POINT_Y])
            Xz_train.append(data.AP_ARR)
            yz_train.append([data.POINT_Z])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train[X_train == 100] = 0
    Xz_train[Xz_train == 100] = 0

    # read test data
    X_test = list()
    y_test = list()
    Xz_test = list()
    yz_test = list()

    for data in validation_data:
        if data.POINT_Z == 0 and data.POINT_X > 100 and data.POINT_Y > 50:
            X_test.append(data.AP_ARR)
            y_test.append([data.POINT_X, data.POINT_Y])
            Xz_test.append(data.AP_ARR)
            yz_test.append([data.POINT_Z])

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test[X_test == 100] = 0
    Xz_test[Xz_test == 100] = 0

    return (X_train, y_train, X_test, y_test, Xz_train, yz_train, Xz_test, yz_test)


def load_train_data(train_data):

    # read training data
    X_train = list()
    y_train = list()
    Xz_train = list()
    yz_train = list()

    for data in train_data:
        if data.POINT_Z == 0 and data.POINT_X > 100 and data.POINT_Y > 50:
            X_train.append(data.AP_ARR)
            y_train.append([data.POINT_X, data.POINT_Y])
            Xz_train.append(data.AP_ARR)
            yz_train.append([data.POINT_Z])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train[X_train == 100] = 0
    Xz_train[Xz_train == 100] = 0

    return (X_train, y_train, Xz_train, yz_train)


DIMEN = 992
SHAPE = 1024
SEG_TIMES = 2

print('loading cluster data...')
cluster_ap_list = rd.readFile('Training_rss_21Aug17.csv', DIMEN, width=0, height=0)
print('cluster data length :', cluster_ap_list.__len__(), '\n')
print('loading validation data...')
validation_ap_list = rd.readFile('Test_rss_21Aug17.csv', DIMEN, width=0, height=0)
print('validation data length :', validation_ap_list.__len__(), '\n')
print('loading train data...')
train_ap_list = rd.readFile('Training_1Aug.csv', DIMEN, width=0, height=0)
print('train data length :', train_ap_list.__len__(), '\n')

X_cluster, y_cluster, X_test, y_test, Xz_cluster, yz_cluster, Xz_test, yz_test = load_data(cluster_ap_list, validation_ap_list)
X_train, y_train, Xz_train, yz_train = load_train_data(train_ap_list)

out_file = open('171226_clustering+learning_out.txt','w');
dic_name = '171226_'

error = 0

dnn_learning_for_this(X_train, y_train, X_test, y_test, dic_name)
print('학습 완료')
out_file.close()