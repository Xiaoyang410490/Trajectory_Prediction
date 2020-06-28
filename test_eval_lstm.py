from Model_lstm import *
from findlane import *
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

def ang_diff(x,y):
    a = x - y
    a = (a + np.pi) % (2 * np.pi) - np.pi
    if abs(a) > 2.7:
        a = -np.sign(a) * (abs(a) - np.pi)
    return a

def train_step(input,target,dec_in):

    loss = 0

    with tf.GradientTape() as tape:

        enc_hidden = encoder.initialize_hidden_state()
        enc_cell = encoder.initialize_cell_state()

        enc_output, enc_hidden,enc_cell = encoder(input, enc_hidden, enc_cell)

        dec_hidden = enc_hidden
        dec_cell = enc_cell

        for t in range(7):

            dec_input = tf.expand_dims(dec_in[:, t], 1)

            predictions, dec_hidden, dec_cell = decoder(dec_input, enc_hidden, dec_hidden,dec_cell, enc_output, 'train')

            predictions = tf.squeeze(predictions,1)

            loss += tf.keras.losses.MSLE(target[:, t], predictions)

    batch_loss = loss/7

    #optimization_step
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def train_set():

    steps_per_epoch = len(x_train) // Batch_size

    #for faster experiments we first set the epoch with 2.
    for epoch in range(15):

        dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train))
        dataset_train = dataset_train.batch(Batch_size, drop_remainder=True)

        total_loss = 0
        index_save = 0

        for (batch, (input_batch, target_batch)) in enumerate(dataset_train.take(steps_per_epoch)):

            step_loss = 0

            for t in range(7, 23):

                input_encoder = input_batch[:,t - 7:t]
                input_decoder = input_batch[:,t:t + 7]
                target_decoder = target_batch[:,t:t + 7]

                loss = train_step(input_encoder, target_decoder, input_decoder)
                step_loss += loss

            index_save = index_save + 1
            if index_save == steps_per_epoch:
                checkpoint.save(file_prefix=checkpoint_prefix)

            step_loss = step_loss/16
            total_loss += step_loss

        total_loss = total_loss/steps_per_epoch
        print(total_loss)


#the main function

#preparation of dataset for train and validation
f = open("x_t.pickle",'rb')
x_train = pickle.load(f)
y_train = pickle.load(f)
m_train = pickle.load(f)
f1 = open("x_v.pickle",'rb')
x_valid = pickle.load(f1)
y_valid = pickle.load(f1)
m_valid = pickle.load(f1)
f2 = open("x_e.pickle",'rb')
x_test = pickle.load(f2)
y_test = pickle.load(f2)
m_test = pickle.load(f2)

Batch_size = 16
units = 32
out_size = 2

dataset_valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
dataset_valid = dataset_valid.batch(Batch_size,drop_remainder=True)

#set the encoder and decoder and do the initialization
encoder = Encoder(units, Batch_size)
decoder = Decoder(out_size, units, Batch_size)

#set the optimizer,commonly we use the adam optimizer
optimizer = tf.keras.optimizers.Adam()

#set the checkpoint
checkpoint_dir = './lstm_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

def metrics():

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    dataset_valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    dataset_valid = dataset_valid.batch(Batch_size, drop_remainder=True)

    Nummer = len(x_valid) // Batch_size

    file_path = "Amsterdamer_Intersection_Lanelet.osm"
    # initial the mapgeomerty class with the file_path and origin point for the projector
    mapgeometry = MapGeometry(file_path, 50.76599713889, 6.06099834167)
    map = mapgeometry.lanelet_map

    loss_whole = 0

    for (batch, (inp, targ)) in enumerate(dataset_valid.take(Nummer)):

        loss = 0
        t = 20
        dt = 0.4
        lr = 1.25
        l = 2.5
        # get the data we need
        input_encoder = inp[:, t - 7:t]
        input_decoder = inp[:, t:t + 7]
        mapdata = np.array(m_test)

        # seq2seq model
        enc_hidden = encoder.initialize_hidden_state()
        enc_cell = encoder.initialize_cell_state()
        enc_out, enc_hidden,enc_cell = encoder(input_encoder, enc_hidden, enc_cell)

        pre_x = []
        pre_y = []
        pre_pfai = []
        pre_beta = []
        pre_velo = []

        # draw the real curve
        for i in range(16):
            kurve = np.array(input_decoder[i])
            map_map = mapdata[i]

            # initial data of the position x,y and pfai,beta
            kurve_initial = kurve[0]
            pos_initial = map_map[0]
            pos_second = map_map[1]
            x_ = pos_initial[3]
            y_ = pos_initial[4]
            x_1 = pos_second[3]
            y_1 = pos_second[4]
            yc = y_1 - y_
            xc = x_1 - x_
            pfai_ = np.arctan2(yc, xc)

            beta_ = kurve_initial[3]
            velo_ = kurve_initial[4]

            pre_x.append(x_)
            pre_y.append(y_)
            pre_pfai.append(pfai_)
            pre_beta.append(beta_)
            pre_velo.append(velo_)

        # fistly we get the map features from decoder_input
        dec_input = tf.expand_dims(input_decoder[:, 0], 1)
        dec_hidden = enc_hidden
        dec_cell = enc_cell

        for ind in range(1, 7):

            predictions, dec_hidden, dec_cell = decoder(dec_input, enc_hidden, dec_hidden, dec_cell, enc_out, 'test')
            predictions = tf.squeeze(predictions, 1)

            next_inp = []

            for k in range(16):
                map_map_1 = mapdata[k]
                inp_map = map_map_1[ind]

                out_step = predictions[k]
                ang_step = out_step[0] / 10
                acc_step = out_step[1]

                input = []

                # The prediction of position,pfai,beta will be used in the next time step
                x_ = pre_x[k]
                y_ = pre_y[k]
                pfai_ = pre_pfai[k]
                beta_ = pre_beta[k]

                vel_step = pre_velo[k] + acc_step * dt
                x_ = x_ + vel_step * math.cos(pfai_ + beta_) * dt
                y_ = y_ + vel_step * math.sin(pfai_ + beta_) * dt
                beta_ = math.atan(lr * math.tan(ang_step) / l)
                pfai_ = pfai_ + vel_step * math.sin(beta_) * dt / lr

                relx_norm = x_ - pre_x[k]
                rely_norm = y_ - pre_y[k]

                pre_x[k] = x_
                pre_y[k] = y_
                pre_pfai[k] = pfai_
                pre_beta[k] = beta_
                pre_velo[k] = vel_step

                pfai_norm = pfai_ / math.pi
                beta_norm = beta_
                vel_norm = vel_step

                id1 = int(inp_map[0])
                id2 = int(inp_map[1])
                id3 = int(inp_map[2])
                lane1 = map.laneletLayer[id1]
                lane2 = map.laneletLayer[id2]
                lane3 = map.laneletLayer[id3]
                x_r = inp_map[3]
                y_r = inp_map[4]
                bx = float(x_)
                by = float(y_)

                loss += (x_ - x_r) ** 2 + (y_ - y_r) ** 2

                kappa_0, ori_0, rul_0 = future_parameter(0, mapgeometry, bx, by, id1, id2, id3, lane1, lane2, lane3)
                kappa_5, ori_5, rul_5 = future_parameter(5, mapgeometry, by, by, id1, id2, id3, lane1, lane2, lane3)
                kappa_10, ori_10, rul_10 = future_parameter(10, mapgeometry, by, by, id1, id2, id3, lane1, lane2,
                                                            lane3)
                kappa_15, ori_15, rul_15 = future_parameter(15, mapgeometry, bx, by, id1, id2, id3, lane1, lane2,
                                                            lane3)

                kap0 = kappa_0 / 0.1
                kap5 = kappa_5 / 0.1
                kap10 = kappa_10 / 0.1
                kap15 = kappa_15 / 0.1

                ori0 = ori_0 / math.pi
                ori5 = ori_5 / math.pi
                ori10 = ori_10 / math.pi
                ori15 = ori_15 / math.pi

                rul0 = rul_0
                rul5 = rul_5
                rul10 = rul_10
                rul15 = rul_15

                gam0 = ang_diff(pfai_, ori0) / math.pi
                gam5 = ang_diff(pfai_, ori5) / math.pi
                gam10 = ang_diff(pfai_, ori10) / math.pi
                gam15 = ang_diff(pfai_, ori15) / math.pi

                input.append([relx_norm, rely_norm, pfai_norm, beta_norm, vel_norm,
                              kap0, rul0, ori0, gam0,
                              kap5, rul5, ori5, gam5,
                              kap10, rul10, ori10, gam10,
                              kap15, rul15, ori15, gam15
                              ])

                next_inp.append(input)

            deco_in = tf.convert_to_tensor(next_inp[0], tf.float64)
            for r in range(1, 16):
                inp = tf.convert_to_tensor(next_inp[r], tf.float64)
                deco_in = tf.concat([deco_in, inp], 0)

            dec_input = tf.expand_dims(deco_in, 1)

        loss_whole += loss

    loss_rmse = math.sqrt(loss_whole/(Nummer*112))

    return loss_rmse


def visualization():

    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset_test = dataset_test.batch(Batch_size, drop_remainder=True)

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    input_test, target_test = next(iter(dataset_test))

    file_path = "Amsterdamer_Intersection_Lanelet.osm"
    # initial the mapgeomerty class with the file_path and origin point for the projector
    mapgeometry = MapGeometry(file_path, 50.76599713889, 6.06099834167)
    map = mapgeometry.lanelet_map

    t = 20
    dt = 0.4
    lr = 1.25
    l = 2.5
    # get the data we need
    input_encoder = input_test[:,t - 7:t]
    input_decoder = input_test[:,t:t + 7]
    mapdata = np.array(m_test)

    # seq2seq model
    hidden_init = encoder.initialize_hidden_state()
    cell_init = encoder.initialize_cell_state()
    enc_out, enc_hidden,enc_cell = encoder(input_encoder, hidden_init, cell_init,'test')
    dec_hidden = enc_hidden
    dec_cell = enc_cell

    x_pre = []
    y_pre = []
    x_real = []
    y_real = []

    pre_x = []
    pre_y = []
    pre_pfai = []
    pre_beta = []
    pre_velo = []

    # draw the real curve
    for i in range(16):

        kurve = np.array(input_decoder[i])
        map_map = mapdata[i]

        # initial data of the position x,y and pfai,beta
        kurve_initial = kurve[0]
        pos_initial = map_map[0]
        pos_second = map_map[1]
        x_ = pos_initial[3]
        y_ = pos_initial[4]
        x_1 = pos_second[3]
        y_1 = pos_second[4]
        yc = y_1 - y_
        xc = x_1 - x_
        pfai_ = np.arctan2(yc , xc)

        beta_ = kurve_initial[3]
        velo_ = kurve_initial[4]

        if i == 12:
            x_pre.append(x_)
            y_pre.append(y_)
            x_real.append(x_)
            y_real.append(y_)

        pre_x.append(x_)
        pre_y.append(y_)
        pre_pfai.append(pfai_)
        pre_beta.append(beta_)
        pre_velo.append(velo_)

    # fistly we get the map features from decoder_input
    dec_input = tf.expand_dims(input_decoder[:, 0], 1)

    for ind in range(1, 7):

        predictions, dec_hidden, dec_cell = decoder(dec_input, enc_hidden, dec_hidden,dec_cell, enc_out, 'test')
        predictions = tf.squeeze(predictions, 1)

        next_inp = []

        for k in range(16):

            map_map_1 = mapdata[k]
            inp_map = map_map_1[ind]

            out_step = predictions[k]
            ang_step = out_step[0] / 10
            acc_step = out_step[1]

            input = []

            # The prediction of position,pfai,beta will be used in the next time step
            x_ = pre_x[k]
            y_ = pre_y[k]
            pfai_ = pre_pfai[k]
            beta_ = pre_beta[k]

            vel_step = pre_velo[k] + acc_step * dt
            x_ = x_ + vel_step * math.cos(pfai_ + beta_) * dt
            y_ = y_ + vel_step * math.sin(pfai_ + beta_) * dt
            beta_ = math.atan(lr * math.tan(ang_step) / l)
            pfai_ = pfai_ + vel_step * math.sin(beta_) * dt / lr

            relx_norm = x_ - pre_x[k]
            rely_norm = y_ - pre_y[k]

            pre_x[k] = x_
            pre_y[k] = y_
            pre_pfai[k] = pfai_
            pre_beta[k] = beta_
            pre_velo[k] = vel_step

            if k == 12:
                x_pre.append(x_)
                y_pre.append(y_)

            pfai_norm = pfai_ / math.pi
            beta_norm = beta_
            vel_norm = vel_step

            id1 = int(inp_map[0])
            id2 = int(inp_map[1])
            id3 = int(inp_map[2])
            lane1 = map.laneletLayer[id1]
            lane2 = map.laneletLayer[id2]
            lane3 = map.laneletLayer[id3]
            x_r = inp_map[3]
            y_r = inp_map[4]
            bx = float(x_)
            by = float(y_)

            if k == 12:
                x_real.append(x_r)
                y_real.append(y_r)
                print(math.sqrt((x_ - x_r)**2 + (y_ - y_r)**2))

            kappa_0, ori_0, rul_0 = future_parameter(0, mapgeometry, bx, by, id1, id2, id3, lane1, lane2, lane3)
            kappa_5, ori_5, rul_5 = future_parameter(5, mapgeometry, by, by, id1, id2, id3, lane1, lane2, lane3)
            kappa_10, ori_10, rul_10 = future_parameter(10, mapgeometry, by, by, id1, id2, id3, lane1, lane2, lane3)
            kappa_15, ori_15, rul_15 = future_parameter(15, mapgeometry, bx, by, id1, id2, id3, lane1, lane2,lane3)

            kap0 = kappa_0 / 0.1
            kap5 = kappa_5 / 0.1
            kap10 = kappa_10 / 0.1
            kap15 = kappa_15 / 0.1

            ori0 = ori_0 / math.pi
            ori5 = ori_5 / math.pi
            ori10 = ori_10 / math.pi
            ori15 = ori_15 / math.pi

            rul0 = rul_0
            rul5 = rul_5
            rul10 = rul_10
            rul15 = rul_15

            gam0 = ang_diff(pfai_, ori0) / math.pi
            gam5 = ang_diff(pfai_, ori5) / math.pi
            gam10 = ang_diff(pfai_, ori10) / math.pi
            gam15= ang_diff(pfai_, ori15) / math.pi

            input.append([ relx_norm, rely_norm, pfai_norm, beta_norm, vel_norm,
                           kap0, rul0, ori0, gam0,
                           kap5, rul5, ori5, gam5,
                           kap10, rul10, ori10, gam10,
                           kap15, rul15, ori15, gam15
                          ])

            next_inp.append(input)

        deco_in = tf.convert_to_tensor(next_inp[0], tf.float64)
        for r in range(1, 16):
            inp = tf.convert_to_tensor(next_inp[r], tf.float64)
            deco_in = tf.concat([deco_in, inp], 0)

        dec_input = tf.expand_dims(deco_in, 1)

    # visualization process
    x_real = np.array(x_real)
    y_real = np.array(y_real)
    x_pre = np.array(x_pre)
    y_pre = np.array(y_pre)
    plt.figure(figsize=(8,8))

    plt.scatter(x_pre,y_pre)
    plt.scatter(x_real,y_real,s=5)
    plt.plot(x_pre, y_pre)
    plt.plot(x_real,y_real)
    plt.show()

train_set()
rmse = metrics()
print(rmse)