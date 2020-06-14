from dataload import *
from Model_multi_gru import *
import os
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

def train_step(input,target,dec_in):

    loss = 0

    with tf.GradientTape() as tape:

        enc_hidden = encoder.initialize_hidden_state()
        enc_output, enc_hidden = encoder(input, enc_hidden)

        dec_hidden = enc_hidden
        dec_hidden_1 = decoder.initialize_hidden_state_1()

        for t in range(15):

            dec_input = tf.expand_dims(dec_in[:, t], 1)

            predictions, dec_hidden, dec_hidden_1 = decoder(dec_input, dec_hidden, dec_hidden_1, enc_output, 'train')

            predictions = tf.squeeze(predictions,1)

            loss += tf.keras.losses.MSLE(target[:, t], predictions)

    batch_loss = loss/15

    #optimization_step
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def train_set():

    steps_per_epoch = len(x_train) // Batch_size

    #for faster experiments we first set the epoch with 2.
    for epoch in range(25):

        dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train))
        dataset_train = dataset_train.batch(Batch_size, drop_remainder=True)

        total_loss = 0

        for i in range(steps_per_epoch):

            step_loss = 0

            input_batch, target_batch = next(iter(dataset_train))

            for t in range(20, 45):

                input_encoder = input_batch[:,t - 20:t]
                input_decoder = input_batch[:,t:t + 15]
                target_decoder = target_batch[:,t:t + 15]

                loss = train_step(input_encoder, target_decoder, input_decoder)
                step_loss += loss

            checkpoint.save(file_prefix=checkpoint_prefix)

            step_loss = step_loss/25
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

        for t in range(20, 45):

            input_encoder = input_valid[:,t - 20:t]
            input_decoder = input_valid[:,t:t + 15]
            target_decoder = target_valid[:,t:t + 15]

            hidden = encoder.initialize_hidden_state()

            enc_out, enc_hidden = encoder(input_encoder, hidden)

            dec_hidden = enc_hidden
            dec_hidden_1 = decoder.initialize_hidden_state_1()

            loss = 0

            for ind in range(15):

                dec_input = tf.expand_dims(input_decoder[:, ind], 1)

                predictions, dec_hidden,dec_hidden_1 = decoder(dec_input, dec_hidden, dec_hidden_1, enc_out, 'test')

                predictions = tf.squeeze(predictions, 1)

                loss += tf.keras.losses.MSLE(target_decoder[:, ind], predictions)

            loss = loss / 15
            kurve_loss += loss

        kurve_loss = kurve_loss/25
        total_loss += kurve_loss

    total_loss = total_loss/Nummer
    print(total_loss)

#the main function

#preparation of dataset for train and validation
x_train,y_train,x_valid,y_valid = train_valid_get()
Batch_size = 16
units = 128
out_size = 2
dataset_valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
dataset_valid = dataset_valid.batch(Batch_size,drop_remainder=True)
#set the encoder and decoder and do the initialization
encoder = Encoder(units, Batch_size)
decoder = Decoder(out_size, units, Batch_size)

#set the optimizer,commonly we use the adam optimizer
optimizer = tf.keras.optimizers.Adam()

#set the checkpoint
checkpoint_dir = './multi_gru_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

#run the training process

#run the test process


def show():

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    input_valid, target_valid = next(iter(dataset_valid))

    t = 40
    dt = 0.2
    lr = 1.25
    l = 2.5
    #get the data we need
    input_encoder = input_valid[:,t - 20:t]
    input_decoder = input_valid[:,t:t + 15]

    #seq2seq model
    hidden = encoder.initialize_hidden_state()

    enc_out, enc_hidden= encoder(input_encoder, hidden)

    dec_hidden = enc_hidden
    dec_hidden_1 = decoder.initialize_hidden_state_1()

    x_real = []
    y_real = []
    x_pre = []
    y_pre = []

    pre_x = []
    pre_y = []
    pre_pfai = []
    pre_beta = []
    pre_velo = []

    #draw the real curve
    for i in range(16):

        kurve = np.array(input_decoder[i])

        #gather the real position data
        if i ==4:
            for time in range(15):
                vector = kurve[time]
                vec_x = vector[0]
                vec_y = vector[1]
                x_real.append(vec_x)
                y_real.append(vec_y)

        #initial data of the position x,y and pfai,beta
        kurve_initial = kurve[0]
        x_ = kurve_initial[0]
        y_ = kurve_initial[1]

        if i==4:
            x_pre.append(x_)
            y_pre.append(y_)

        pfai_ = kurve_initial[2]
        beta_ = kurve_initial[3]
        velo_ = kurve_initial[7]

        pre_x.append(x_)
        pre_y.append(y_)
        pre_pfai.append(pfai_)
        pre_beta.append(beta_)
        pre_velo.append(velo_)

    #The input of current state is from decoder_input
    dec_input = tf.expand_dims(input_decoder[:, 0], 1)

    for ind in range(1, 15):

        predictions, dec_hidden,dec_hidden_1 = decoder(dec_input, dec_hidden, dec_hidden_1, enc_out,'test')
        print(predictions)
        predictions = tf.squeeze(predictions, 1)
        input_rel = input_decoder[:, ind]

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
            pre_velo[k]= vel_step

            if k == 4:
                x_pre.append(x_)
                y_pre.append(y_)

            inp_real = input_rel[k]
            kru_ = inp_real[4]
            dis_ = inp_real[5]
            ang_ = inp_real[6]

            input.append([x_, y_, pfai_, beta_, kru_, dis_, ang_, vel_step])
            next_inp.append(input)

        deco_in = tf.convert_to_tensor(next_inp[0],tf.float64)

        for r in range(1,16):
            inp = tf.convert_to_tensor(next_inp[r],tf.float64)
            deco_in = tf.concat([deco_in,inp],0)

        dec_input = tf.expand_dims(deco_in, 1)

    #visualization process
    x_real = np.array(x_real)
    y_real = np.array(y_real)
    x_pre = np.array(x_pre)
    y_pre = np.array(y_pre)
    plt.scatter(x_real,y_real,s=1)
    plt.scatter(x_pre,y_pre)
    plt.show()

def show_real():
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    input_valid, target_valid = next(iter(dataset_valid))

    t = 70
    dt = 0.04
    lr = 1.25
    l = 2.5
    # get the data we need
    input_encoder = input_valid[:, t - 30:t]
    input_decoder = input_valid[:, t:t + 15]

    # seq2seq model
    hidden = encoder.initialize_hidden_state()

    enc_out, enc_hidden = encoder(input_encoder, hidden)

    dec_hidden = enc_hidden

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
        if i == 6:
            for time in range(15):
                vector = kurve[time]
                vec_x = vector[0]
                vec_y = vector[1]
                x_real.append(vec_x)
                y_real.append(vec_y)

        # initial data of the position x,y and pfai,beta
        kurve_initial = kurve[0]
        x_ = kurve_initial[0]
        y_ = kurve_initial[1]
        pfai_ = kurve_initial[2]
        beta_ = kurve_initial[3]
        velo_ = kurve_initial[7]

        pre_x.append(x_)
        pre_y.append(y_)
        pre_pfai.append(pfai_)
        pre_beta.append(beta_)
        pre_velo.append(velo_)

    for ind in range(15):

        dec_input = tf.expand_dims(input_decoder[:, ind], 1)

        predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_out, 'test')
        print(predictions)
        predictions = tf.squeeze(predictions, 1)

        for k in range(16):

            out_step = predictions[k]
            ang_step = out_step[0] / 100
            acc_step = out_step[1]

            # The prediction of position,pfai,beta will be used in the next time step
            x_ = pre_x[k]
            y_ = pre_y[k]
            pfai_ = pre_pfai[k]
            beta_ = pre_beta[k]

            vel_step =pre_velo[k]  + acc_step*dt

            x_ = x_ + vel_step * math.cos(pfai_ + beta_) * dt
            y_ = y_ + vel_step * math.sin(pfai_ + beta_) * dt
            beta_ = math.atan(lr * math.tan(ang_step) / l)
            pfai_ = pfai_ + vel_step * math.sin(beta_) * dt / lr

            pre_x[k] = x_
            pre_y[k] = y_
            pre_pfai[k] = pfai_
            pre_beta[k] = beta_
            pre_velo[k] = vel_step

            if k == 6:
                x_pre.append(x_)
                y_pre.append(y_)

    # visualization process
    x_real = np.array(x_real)
    y_real = np.array(y_real)
    x_pre = np.array(x_pre)
    y_pre = np.array(y_pre)
    plt.scatter(x_real, y_real, s=1)
    plt.scatter(x_pre, y_pre)
    plt.show()

show()