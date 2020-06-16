from Model_baseline import *
from findlane import *
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

def train_step(input,target,dec_in):

    loss = 0

    with tf.GradientTape() as tape:

        hidden = lstmlayer.initialize_hidden_state()
        cell = lstmlayer.initialize_cell_state()
        hidden_1 = lstmlayer.initialize_hidden_state()
        cell_1 = lstmlayer.initialize_cell_state()

        output, hidden, cell, hidden_1, cell_1= lstmlayer(input, hidden, cell,hidden_1, cell_1 ,'train')

        for t in range(7):

            dec_input = tf.expand_dims(dec_in[:, t], 1)

            predictions, hidden, cell, hidden_1, cell_1 = lstmlayer(dec_input, hidden, cell, hidden_1, cell_1,'train')

            predictions = tf.squeeze(predictions,1)

            loss += tf.keras.losses.MSLE(target[:, t], predictions)

    batch_loss = loss/7

    #optimization_step
    variables = lstmlayer.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def train_set():

    steps_per_epoch = len(x_train) // Batch_size

    #for faster experiments we first set the epoch with 2.
    for epoch in range(20):

        dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train))
        dataset_train = dataset_train.batch(Batch_size, drop_remainder=True)

        total_loss = 0

        for i in range(steps_per_epoch):

            step_loss = 0

            input_batch, target_batch = next(iter(dataset_train))

            for t in range(7,23):

                input_encoder = input_batch[:,t - 7:t]
                input_decoder = input_batch[:,t:t + 7]
                target_decoder = target_batch[:,t:t + 7]

                loss = train_step(input_encoder, target_decoder, input_decoder)
                step_loss += loss

            if i == (steps_per_epoch-1):
                checkpoint.save(file_prefix=checkpoint_prefix)

            step_loss = step_loss/16
            total_loss += step_loss

        total_loss = total_loss/steps_per_epoch
        print(total_loss)


def test_set():

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    Nummer = len(x_valid)//Batch_size

    total_loss = 0

    for i in range(Nummer):

        kurve_loss = 0

        input_valid, target_valid = next(iter(dataset_valid))

        for t in range(7, 23):

            input_encoder = input_valid[:,t - 7:t]
            input_decoder = input_valid[:,t:t + 7]
            target_decoder = target_valid[:,t:t + 7]

            hidden = lstmlayer.initialize_hidden_state()
            cell = lstmlayer.initialize_cell_state()
            hidden_1 = lstmlayer.initialize_hidden_state()
            cell_1 = lstmlayer.initialize_cell_state()

            enc_out, hidden, cell, hidden_1, cell_1 = lstmlayer(input_encoder, hidden, cell, hidden_1, cell_1,'test')

            loss = 0

            for ind in range(7):

                dec_input = tf.expand_dims(input_decoder[:, ind], 1)

                predictions, hidden, cell, hidden_1, cell_1 = lstmlayer(dec_input, hidden, cell, hidden_1, cell_1,'test')

                predictions = tf.squeeze(predictions, 1)

                loss += tf.keras.losses.MSLE(target_decoder[:, ind], predictions)

            loss = loss / 7
            kurve_loss += loss

        kurve_loss = kurve_loss / 16
        total_loss += kurve_loss

    total_loss = total_loss/Nummer
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

dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
dataset_test = dataset_test.batch(Batch_size,drop_remainder=True)

#set the encoder and decoder and do the initialization
lstmlayer = multi_lstm(units, Batch_size)

#set the optimizer,commonly we use the adam optimizer
optimizer = tf.keras.optimizers.Adam()

#set the checkpoint
checkpoint_dir = './baseline_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 lstmlayer=lstmlayer)

def metrics():

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    Nummer = len(x_valid) // Batch_size

    print(Nummer)

    loss_whole = 0

    for i in range(Nummer):

        input_valid, target_valid = next(iter(dataset_valid))

        # Use the projector to get the map
        projector = UtmProjector(lanelet2.io.Origin(50.76599713889, 6.06099834167))
        path = file_path
        map = lanelet2.io.load(path, projector)
        # Use routing to get the graph
        mapgeometry = MapGeometry(file_path, 50.76599713889, 6.06099834167)

        t = 20
        dt = 0.4
        lr = 1.25
        l = 2.5

        # get the data we need
        input_encoder = input_valid[:, t - 7:t]
        input_decoder = input_valid[:, t:t + 7]
        mapdata = np.array(m_test)
        map_data = mapdata[:, t:t + 7]

        # seq2seq model
        hidden = lstmlayer.initialize_hidden_state()
        cell = lstmlayer.initialize_cell_state()
        hidden_1 = lstmlayer.initialize_hidden_state()
        cell_1 = lstmlayer.initialize_cell_state()

        enc_out, hidden, cell, hidden_1, cell_1 = lstmlayer(input_encoder, hidden, cell, hidden_1, cell_1, 'test')

        pre_x = []
        pre_y = []
        pre_pfai = []
        pre_beta = []
        pre_velo = []

        for i in range(16):

            kurve = np.array(input_decoder[i])
            map_map = mapdata[i]

            # initial data of the position x,y and pfai,beta
            kurve_initial = kurve[0]
            pos_initial = map_map[0]
            x_ = pos_initial[3]
            y_ = pos_initial[4]
            pfai_ = kurve_initial[2] * math.pi
            beta_ = kurve_initial[3]
            velo_ = kurve_initial[4]

            pre_x.append(x_)
            pre_y.append(y_)
            pre_pfai.append(pfai_)
            pre_beta.append(beta_)
            pre_velo.append(velo_)

        # fistly we get the map features from decoder_input
        dec_input = tf.expand_dims(input_decoder[:, 0], 1)

        loss_per_batch = 0

        for ind in range(1, 7):

            predictions, hidden, cell, hidden_1, cell_1 = lstmlayer(dec_input, hidden, cell, hidden_1, cell_1, 'test')
            predictions = tf.squeeze(predictions, 1)

            input_map = map_data[:, ind]

            next_inp = []

            loss_per_step = 0

            for k in range(16):

                out_step = predictions[k]
                ang_step = out_step[0] / 10
                acc_step = out_step[1]

                input = []

                # The prediction of position,pfai,beta will be used in the next time step
                x_ = pre_x[k]
                y_ = pre_y[k]
                pfai_ = pre_pfai[k]
                beta_ = pre_beta[k]
                vel_step = pre_velo[k]

                x_ = x_ + vel_step * math.cos(pfai_ + beta_) * dt
                y_ = y_ + vel_step * math.sin(pfai_ + beta_) * dt
                beta_ = math.atan(lr * math.tan(ang_step) / l)
                pfai_ = pfai_ + vel_step * math.sin(beta_) * dt / lr
                vel_step = pre_velo[k] + acc_step * dt

                relx_norm = x_ - pre_x[k]
                rely_norm = y_ - pre_y[k]

                pre_x[k] = x_
                pre_y[k] = y_
                pre_pfai[k] = pfai_
                pre_beta[k] = beta_
                pre_velo[k] = vel_step

                inp_map = input_map[k]
                pfai_norm = pfai_ / math.pi
                beta_norm = beta_
                vel_norm = vel_step
                id1 = int(inp_map[0])
                id2 = int(inp_map[1])
                id3 = int(inp_map[2])
                x_r = inp_map[3]
                y_r= inp_map[4]

                lane1 = map.laneletLayer[id1]
                lane2 = map.laneletLayer[id2]
                lane3 = map.laneletLayer[id3]

                bx = float(x_)
                by = float(y_)

                print(math.sqrt(((x_ -  x_r)**2 + (y_ - y_r)**2)))
                loss_per_step += math.sqrt(((x_ -  x_r)**2 + (y_ - y_r)**2))

                width, kappa, orien, rul_, kappa_1, ori_1, rul_1, fai_1, kappa_3, ori_3, rul_3, fai_3, kappa_5, ori_5, rul_5, fai_5 = geo_rechnen(
                    lane1, lane2, lane3, id1, id2, id3, bx, by, mapgeometry)

                kappa = kappa / 0.1
                kappa_1 = kappa_1 / 0.1
                kappa_3 = kappa_3 / 0.1
                kappa_5 = kappa_5 / 0.1

                orien = orien
                ori_1 = ori_1 / math.pi
                ori_3 = ori_3 / math.pi
                ori_5 = ori_5 / math.pi

                fai_1 = fai_1 / math.pi
                fai_3 = fai_3 / math.pi
                fai_5 = fai_5 / math.pi

                input.append([relx_norm, rely_norm, pfai_norm, beta_norm,
                              vel_norm, kappa, rul_, orien,
                              kappa_1, rul_1, ori_1, fai_1,
                              kappa_3, rul_3, ori_3, fai_3,
                              kappa_5, rul_5, ori_5, fai_5
                              ])

                next_inp.append(input)

            deco_in = tf.convert_to_tensor(next_inp[0], tf.float64)

            for r in range(1, 16):
                inp = tf.convert_to_tensor(next_inp[r], tf.float64)
                deco_in = tf.concat([deco_in, inp], 0)

            dec_input = tf.expand_dims(deco_in, 1)

            print(loss_per_step/16)
            loss_per_batch += (loss_per_step / 16)

        loss_whole += (loss_per_batch / 6)

    RMSE_loss = loss_whole / Nummer

    return RMSE_loss



def visualization():

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    input_valid, target_valid = next(iter(dataset_test))

    # Use the projector to get the map
    projector = UtmProjector(lanelet2.io.Origin(50.76599713889, 6.06099834167))
    path = file_path
    map = lanelet2.io.load(path, projector)
    # Use routing to get the graph
    mapgeometry = MapGeometry(file_path, 50.76599713889, 6.06099834167)

    t = 20
    dt = 0.4
    lr = 1.25
    l = 2.5
    # get the data we need
    input_encoder = input_valid[:,t - 7:t]
    input_decoder = input_valid[:,t:t + 7]
    mapdata = np.array(m_test)
    map_data = mapdata[:,t:t+7]

    # seq2seq model
    hidden = lstmlayer.initialize_hidden_state()
    cell = lstmlayer.initialize_cell_state()
    hidden_1 = lstmlayer.initialize_hidden_state()
    cell_1 = lstmlayer.initialize_cell_state()

    enc_out, hidden, cell, hidden_1, cell_1 = lstmlayer(input_encoder, hidden, cell, hidden_1, cell_1, 'test')

    x_real = []
    y_real = []
    x_pre = []
    y_pre = []

    pre_x = []
    pre_y = []
    pre_pfai = []
    pre_beta = []
    pre_velo = []

    # draw the real curve
    for i in range(16):

        kurve = np.array(input_decoder[i])
        map_map = mapdata[i]

        # gather the real position data
        if i == 7:
            for time in range(7):
                vec_pos = map_map[time]
                x_real.append(vec_pos[3])
                y_real.append(vec_pos[4])

        # initial data of the position x,y and pfai,beta
        kurve_initial = kurve[0]
        pos_initial = map_map[0]
        x_ = pos_initial[3]
        y_ = pos_initial[4]
        pfai_ = kurve_initial[2] * math.pi
        beta_ = kurve_initial[3]
        velo_ = kurve_initial[4]

        if i == 7:
            x_pre.append(x_)
            y_pre.append(y_)

        pre_x.append(x_)
        pre_y.append(y_)
        pre_pfai.append(pfai_)
        pre_beta.append(beta_)
        pre_velo.append(velo_)

    # fistly we get the map features from decoder_input
    dec_input = tf.expand_dims(input_decoder[:, 0], 1)

    for ind in range(1, 7):

        predictions, hidden, cell, hidden_1, cell_1 = lstmlayer(dec_input, hidden, cell, hidden_1, cell_1, 'test')
        predictions = tf.squeeze(predictions, 1)

        input_map = map_data[:,ind]

        next_inp = []

        for k in range(16):

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

            if k == 7:
                x_pre.append(x_)
                y_pre.append(y_)

            inp_map = input_map[k]
            pfai_norm = pfai_ / math.pi
            beta_norm = beta_
            vel_norm = vel_step
            id1 = int(inp_map[0])
            id2 = int(inp_map[1])
            id3 = int(inp_map[2])
            x_r = inp_map[3]
            y_r = inp_map[4]

            if k == 7:
                print(math.sqrt((x_-x_r)**2+(y_-y_r)**2))

            lane1 = map.laneletLayer[id1]
            lane2 = map.laneletLayer[id2]
            lane3 = map.laneletLayer[id3]

            bx = float(x_)
            by = float(y_)

            width, kappa, orien, rul_, kappa_1, ori_1, rul_1, fai_1, kappa_3, ori_3, rul_3, fai_3, kappa_5, ori_5, rul_5, fai_5 = geo_rechnen(lane1, lane2, lane3, id1, id2, id3, bx, by, mapgeometry)

            kappa = kappa / 0.1
            kappa_1 = kappa_1 / 0.1
            kappa_3 = kappa_3 / 0.1
            kappa_5 = kappa_5 / 0.1

            orien = orien
            ori_1 = ori_1 / math.pi
            ori_3 = ori_3 / math.pi
            ori_5 = ori_5 / math.pi

            fai_1 = fai_1 / math.pi
            fai_3 = fai_3 / math.pi
            fai_5 = fai_5 / math.pi

            input.append([ relx_norm, rely_norm, pfai_norm, beta_norm,
                           vel_norm, kappa, rul_, orien,
                           kappa_1, rul_1, ori_1, fai_1,
                           kappa_3, rul_3, ori_3, fai_3,
                           kappa_5, rul_5, ori_5, fai_5
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
    plt.scatter(x_real,y_real)
    plt.scatter(x_pre,y_pre)
    plt.plot(x_real, y_real)
    plt.plot(x_pre, y_pre)
    plt.show()

visualization()