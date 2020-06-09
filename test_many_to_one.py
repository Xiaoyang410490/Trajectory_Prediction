from dataload import *
from Model_many_to_one import *
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

        dec_input = tf.expand_dims(dec_in[:, 0], 1)

        predictions, hidden, cell, hidden_1, cell_1 = lstmlayer(dec_input, hidden, cell, hidden_1, cell_1,'train')

        predictions = tf.squeeze(predictions,1)

        loss += tf.keras.losses.MSLE(target[:, 0], predictions)

    #optimization_step
    variables = lstmlayer.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss


def train_set():

    steps_per_epoch = len(x_train) // Batch_size

    #for faster experiments we first set the epoch with 2.
    for epoch in range(15):

        dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train))
        dataset_train = dataset_train.batch(Batch_size, drop_remainder=True)

        total_loss = 0

        for i in range(steps_per_epoch):

            step_loss = 0

            input_batch, target_batch = next(iter(dataset_train))

            for t in range(20,59):

                input_encoder = input_batch[:,t - 20:t]
                input_decoder = input_batch[:,t:t+1]
                target_decoder = target_batch[:,t:t+1]

                loss = train_step(input_encoder, target_decoder, input_decoder)
                step_loss += loss

            if i == (steps_per_epoch-1):
                checkpoint.save(file_prefix=checkpoint_prefix)

            step_loss = step_loss/39
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

        for t in range(20, 59):

            input_encoder = input_valid[:,t - 20:t]
            input_decoder = input_valid[:,t:t+1]
            target_decoder = target_valid[:,t:t+1]

            hidden = lstmlayer.initialize_hidden_state()
            cell = lstmlayer.initialize_cell_state()
            hidden_1 = lstmlayer.initialize_hidden_state()
            cell_1 = lstmlayer.initialize_cell_state()

            enc_out, hidden, cell, hidden_1, cell_1 = lstmlayer(input_encoder, hidden, cell, hidden_1, cell_1,'test')

            dec_input = tf.expand_dims(input_decoder[:,0], 1)

            predictions, hidden, cell, hidden_1, cell_1 = lstmlayer(dec_input, hidden, cell, hidden_1, cell_1,'test')

            predictions = tf.squeeze(predictions, 1)

            loss = tf.keras.losses.MSLE(target_decoder[:, 0], predictions)

            kurve_loss += loss

        kurve_loss = kurve_loss/39
        total_loss += kurve_loss

    total_loss = total_loss/Nummer
    print(total_loss)


def show():

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    input_valid, target_valid = next(iter(dataset_valid))

    t = 30
    dt = 0.2
    lr = 1.25
    l = 2.5

    input_encoder = input_valid[:, t - 20:t]
    input_decoder = input_valid[:, t:t + 15]

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

        # gather the real position data
        if i == 7:
            for time in range(15):
                vector = kurve[time]
                vec_x = vector[0]
                vec_y = vector[1]
                x_real.append(vec_x * 10)
                y_real.append(vec_y * 10)

        # initial data of the position x,y and pfai,beta
        kurve_initial = kurve[0]
        x_ = kurve_initial[0] * 10
        y_ = kurve_initial[1] * 10
        pfai_ = kurve_initial[4] * 2 * math.pi + math.pi
        beta_ = kurve_initial[5]
        velo_ = kurve_initial[6]

        pre_x.append(x_)
        pre_y.append(y_)
        pre_pfai.append(pfai_)
        pre_beta.append(beta_)
        pre_velo.append(velo_)

    # The input of current state is from decoder_input
    dec_input = tf.expand_dims(input_decoder[:, 0], 1)

    for ind in range(15):

        hidden = lstmlayer.initialize_hidden_state()
        cell = lstmlayer.initialize_cell_state()
        hidden_1 = lstmlayer.initialize_hidden_state()
        cell_1 = lstmlayer.initialize_cell_state()

        enc_out, hidden, cell, hidden_1, cell_1 = lstmlayer(input_encoder, hidden, cell, hidden_1, cell_1, 'test')
        predictions, hidden, cell, hidden_1, cell_1 = lstmlayer(dec_input, hidden, cell, hidden_1, cell_1, 'test')
        predictions = tf.squeeze(predictions, 1)

        if ind < 14:
            input_rel = input_decoder[:,ind+1]
        else:
            input_rel = input_decoder[:,ind]

        next_inp = []

        for k in range(16):

            out_step = predictions[k]
            ang_step = out_step[0]/100
            acc_step = out_step[1]

            input = []

            #The prediction of position,pfai,beta will be used in the next time step
            x_ = pre_x[k]
            y_ = pre_y[k]
            pfai_ = pre_pfai[k]
            beta_ = pre_beta[k]
            vel_step = pre_velo[k] + acc_step * dt

            x_ = x_ + vel_step * math.cos(pfai_ + beta_) * dt
            y_ = y_ + vel_step * math.sin(pfai_ + beta_) * dt
            beta_ = math.atan(lr * math.tan(ang_step) / l)
            pfai_ = pfai_ + vel_step * math.sin(beta_) * dt / lr

            pre_x[k] = x_
            pre_y[k] = y_
            pre_pfai[k] = pfai_
            pre_beta[k] = beta_
            pre_velo[k] = vel_step

            if k == 7:
                x_pre.append(x_)
                y_pre.append(y_)

            inp_real = input_rel[k]
            x_norm = x_/10
            y_norm = y_/10
            relx_norm = inp_real[2]
            rely_norm = inp_real[3]
            pfai_norm = (pfai_-math.pi)/(2*math.pi)
            kru_ = inp_real[7]
            rul_ = inp_real[8]
            ori_ = inp_real[9]
            kru_1 = inp_real[10]
            rul_1 = inp_real[11]
            ori_1 = inp_real[12]
            kru_3 = inp_real[13]
            rul_3 = inp_real[14]
            ori_3 = inp_real[15]

            input.append([x_norm, y_norm, relx_norm, rely_norm,
                          pfai_norm, beta_, vel_step, kru_,
                          rul_, ori_, kru_1, rul_1,
                          ori_1, kru_3, rul_3, ori_3])
            next_inp.append(input)

        deco_in = tf.convert_to_tensor(next_inp[0], tf.float64)
        for r in range(1, 16):
            inp = tf.convert_to_tensor(next_inp[r], tf.float64)
            deco_in = tf.concat([deco_in, inp], 0)

        dec_input = tf.expand_dims(deco_in, 1)
        input_encoder = input_encoder[:,1:20]
        input_encoder = tf.concat([input_encoder,dec_input],1)

    #visualization process
    x_real = np.array(x_real)
    y_real = np.array(y_real)
    x_pre = np.array(x_pre)
    y_pre = np.array(y_pre)
    plt.scatter(x_real,y_real,s=1)
    plt.scatter(x_pre,y_pre)
    plt.show()


#preparation of dataset for train and validation
x_train,y_train,x_valid,y_valid = train_valid_get()
Batch_size = 16
units = 32
out_size = 2
dataset_valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
dataset_valid = dataset_valid.batch(Batch_size,drop_remainder=True)

#set the encoder and decoder and do the initialization
lstmlayer = multi_lstm(units, Batch_size)

#set the optimizer,commonly we use the adam optimizer
optimizer = tf.keras.optimizers.Adam()

#set the checkpoint
checkpoint_dir = './many_to_one_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 lstmlayer=lstmlayer)

train_set()
test_set()
show()
