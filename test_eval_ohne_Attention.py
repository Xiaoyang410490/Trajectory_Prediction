from dataload import *
from Model_new_ohne_Attention import *
import os
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

def train_step(input,target,dec_in):

    loss = 0

    with tf.GradientTape() as tape:

        enc_hidden = encoder.initialize_hidden_state()
        enc_cell = encoder.initialize_cell_state()

        enc_output, enc_hidden, enc_cell = encoder(input, enc_hidden, enc_cell)

        dec_hidden = enc_hidden
        dec_cell = enc_cell
        dec_hidden_1 = decoder.initialize_hidden_state_1()
        dec_cell_1 = decoder.initialize_cell_state_1()

        for t in range(15):

            dec_input = tf.expand_dims(dec_in[:, t], 1)

            predictions, dec_hidden, dec_cell, dec_hidden_1, dec_cell_1 = decoder(dec_input, dec_hidden, dec_cell, dec_hidden_1, dec_cell_1, mode ='train')

            predictions = tf.squeeze(predictions,1)

            loss += tf.keras.losses.MAE(target[:, t], predictions)

    batch_loss = loss / 15

    #optimization
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def train_set():
    steps_per_epoch = len(x_train) // Batch_size

    #for faster experiments we first set the epoch with 2.
    for epoch in range(8):
        dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_train)
        dataset_train = dataset_train.batch(Batch_size, drop_remainder=True)

        total_loss = 0

        for i in range(steps_per_epoch):

            step_loss = 0

            input_batch, target_batch = next(iter(dataset_train))

            for t in range(30, 135):

                input_encoder = input_batch[:,t - 30:t]
                input_decoder = input_batch[:,t:t + 15]
                target_decoder = target_batch[:,t:t + 15]

                loss = train_step(input_encoder, target_decoder, input_decoder)
                step_loss += loss

            checkpoint.save(file_prefix=checkpoint_prefix)

            step_loss = step_loss/105
            total_loss += step_loss

        total_loss = total_loss/steps_per_epoch
        print(total_loss)


def test_set():
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    dataset_valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).shuffle(BUFFER_valid)
    dataset_valid = dataset_valid.batch(Batch_size, drop_remainder=True)

    Nummer = len(x_valid)//Batch_size

    total_loss = 0

    for i in range(Nummer):

        kurve_loss = 0

        input_valid, target_valid = next(iter(dataset_valid))

        for t in range(30, 135):

            input_encoder = input_valid[:,t - 30:t]
            input_decoder = input_valid[:,t:t + 15]
            target_decoder = target_valid[:,t:t + 15]

            hidden = encoder.initialize_hidden_state()
            cell = encoder.initialize_cell_state()

            hidden_1 = decoder.initialize_hidden_state_1()
            cell_1 = decoder.initialize_cell_state_1()

            enc_out, enc_hidden, enc_cell = encoder(input_encoder, hidden, cell)

            dec_hidden = enc_hidden
            dec_cell = enc_cell

            loss = 0

            for ind in range(15):

                dec_input = tf.expand_dims(input_decoder[:, ind], 1)

                predictions, dec_hidden, dec_cell, hidden_1, cell_1 = decoder(dec_input, dec_hidden, dec_cell, hidden_1, cell_1, 'test')

                predictions = tf.squeeze(predictions, 1)

                loss += tf.keras.losses.MAE(target_decoder[:, ind], predictions)

            loss = loss / 15
            kurve_loss += loss

        kurve_loss = kurve_loss/105
        total_loss += kurve_loss

    total_loss = total_loss/Nummer
    print(total_loss)

#the main function

#preparation of dataset for train and validation
x_train,y_train,x_valid,y_valid = train_valid_get3()
BUFFER_train = len(x_train)
BUFFER_valid = len(x_valid)
Batch_size = 16
steps_per_epoch = len(x_train)//Batch_size
units = 64
out_size = 2

#set the encoder and decoder and do the initialization
encoder = Encoder_lstm(units, Batch_size)
decoder = Decoder_lstm(out_size, units, units, Batch_size)

#set the optimizer,commonly we use the adam optimizer
optimizer = tf.keras.optimizers.Adam()

#set the checkpoint
checkpoint_dir = './ohne_attention_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

#run the training process
train_set()
#run the test process
test_set()


def show():

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    for k in range(2):
         input_valid, target_valid = next(iter(dataset_valid))

    t = 100
    dt = 0.04
    lr = 1.25
    l = 2.5

    input_encoder = input_valid[:,t - 30:t]
    input_decoder = input_valid[:,t:t + 15]

    #seq2seq model
    hidden = encoder.initialize_hidden_state()
    cell = encoder.initialize_cell_state()
    hidden_1 = decoder.initialize_hidden_state_1()
    cell_1 = decoder.initialize_cell_state_1()

    enc_out, enc_hidden, enc_cell = encoder(input_encoder, hidden, cell)

    dec_hidden = enc_hidden
    dec_cell = enc_cell

    #draw the real curve
    kurve_0 = np.array(input_decoder[0])
    kurve_1 = np.array(input_decoder[1])

    x_real = []
    y_real = []

    for i in range(15):
        vector = kurve_1[i]
        vec_x = (vector[0]-0.05)*30
        vec_y = (vector[1]-0.05)*30
        x_real.append(vec_x)
        y_real.append(vec_y)

    x_pre = []
    y_pre = []

    #initialization
    kurve_initial = kurve_0[0]
    x_ =  (kurve_initial[0] - 0.05) * 30
    y_ =  (kurve_initial[1] - 0.05) * 30
    pfai_ =  (kurve_initial[2] - 0.03) * math.pi
    beta_ = kurve_initial[3]

    kurve_initial_1 = kurve_1[0]
    x_1 = (kurve_initial_1[0] - 0.05) * 30
    y_1 = (kurve_initial_1[1] - 0.05) * 30
    pfai_1 = (kurve_initial_1[2] - 0.03) * math.pi
    beta_1 = kurve_initial_1[3]

    x_pre.append(x_1)
    y_pre.append(y_1)

    dec_input = tf.expand_dims(input_decoder[:, 0], 1)

    for ind in range(1, 15):

        predictions, dec_hidden, dec_cell, hidden_1, cell_1 = decoder(dec_input, dec_hidden, dec_cell, hidden_1, cell_1, 'test')
        predictions = tf.squeeze(predictions, 1)

        out_step0 = predictions[0]
        ang_step0 = out_step0[0]
        vel_step0 = out_step0[1]

        out_step1 = predictions[1]
        ang_step1 = out_step1[0]
        vel_step1 = out_step1[1]

        input_0 = []
        input_1 = []

        x_ = x_ + vel_step0 * math.cos(pfai_ + beta_)*dt
        y_ = y_ + vel_step0 * math.sin(pfai_ + beta_)*dt
        beta_ = math.atan(lr * math.tan(ang_step0) / l)
        pfai_ = pfai_ + vel_step0 * math.sin(beta_) * dt / lr

        x_1 = x_1 + vel_step1 * math.cos(pfai_1 + beta_1) * dt
        y_1 = y_1 + vel_step1 * math.sin(pfai_1 + beta_1) * dt
        beta_1 = math.atan(lr * math.tan(ang_step1) / l)
        pfai_1 = pfai_1 + vel_step1 * math.sin(beta_1) * dt / lr

        x_pre.append(x_1)
        y_pre.append(y_1)

        input_rel = input_decoder[:,ind]
        inp_real0 = input_rel[0]
        inp_real1 = input_rel[1]
        kru_0 = inp_real0[5]
        kru_1 = inp_real1[5]

        x_norm = x_ / 30 + 0.05
        y_norm = y_ / 30 + 0.05
        pfai_norm = pfai_ / math.pi + 0.03
        input_0.append([x_norm,y_norm,pfai_norm,beta_,kru_0,vel_step0])
        input_0 = tf.convert_to_tensor(input_0,tf.float64)

        x_norm_1 = x_1 / 30 + 0.05
        y_norm_1 = y_1 / 30 + 0.05
        pfai_norm_1 = pfai_1 / math.pi + 0.03
        input_1.append([x_norm_1, y_norm_1, pfai_norm_1, beta_1, kru_1, vel_step1])
        input_1 = tf.convert_to_tensor(input_1,tf.float64)

        deco_in = tf.concat([input_0,input_1],0)
        dec_input = tf.expand_dims(deco_in, 1)

    #visualization
    x_real = np.array(x_real)
    y_real = np.array(y_real)
    x_pre = np.array(x_pre)
    y_pre = np.array(y_pre)
    plt.scatter(x_real,y_real,s=1)
    plt.scatter(x_pre,y_pre)
    plt.show()
