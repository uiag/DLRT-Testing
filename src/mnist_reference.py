from networks.dense_dlrt_nets import ReferenceNet
from networks.utils import create_csv_logger_cb

import tensorflow as tf
from tensorflow import keras

import numpy as np
from optparse import OptionParser
from os import path, makedirs

from random import randint, sample

def train(load_model=0, dim_layer=256, epochs=250, batch_size=256, name="dense_weights", input_dim=784, output_dim=10, optimizerName="Adam",
          learning_rate=1e-3, datasetSize=1, noise=0, dataset="mnist", regularizer=None, regularizer_amount=[0, 0], val_size=10000, val_full=True):
    # specify training
    
    filename = name
    folder_name = filename + '/latest_model'
    folder_name_best = filename + '/best_model'

    # check if dir exists
    if not path.exists(folder_name):
        makedirs(folder_name)
    if not path.exists(folder_name_best):
        makedirs(folder_name_best)
    print("save model as: " + filename)

    dlra_layer_dim = dim_layer
    model = ReferenceNet(input_dim=input_dim, output_dim=output_dim, layer_dim=dlra_layer_dim, regularizer=regularizer, regularizer_amount=regularizer_amount)
    model.build_model()
    
    
    # changed optimizer
    if optimizerName == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizerName == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        print("Optimizer not supported!")
        return
    
    
    # Choose loss
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # Choose metrics (to monitor training, but not to optimize on)
    loss_metric = tf.keras.metrics.Mean()
    acc_metric = tf.keras.metrics.Accuracy()

    # Build dataset
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, (-1, input_dim))
        x_test = np.reshape(x_test, (-1, input_dim))
    
    elif dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = np.reshape(x_train, (-1, input_dim))
        x_test = np.reshape(x_test, (-1, input_dim))
        
    
    # Reserve 10,000 samples for validation.
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    if val_full:
        x_train = x_train[:-val_size]
        y_train = y_train[:-val_size]
    
    
    #changed smaller dataset
    if datasetSize != 1:
        train_len = int(len(x_train)*datasetSize)
        val_len = int(len(x_val)*datasetSize)
        test_len = int(len(x_test)*datasetSize)
        x_train = x_train[:train_len]
        y_train = y_train[:train_len]
        x_val = x_val[:val_len]
        y_val = y_val[:val_len]
        x_test = x_test[:test_len]
        y_test = y_test[:test_len]
    
    print("\nSize Dataset: " + str(len(x_train)) + ", " + str(len(x_val)) + ", " + str(len(x_test)) + "\n")
    
    #changed label noise
    if noise != 0:
        print("\n Noise: " + str(noise) + "\n")
        ls = sample(range(len(x_train)), int(noise*len(x_train)))
        for i in ls:
            y_train[i] = randint(0, 9)
            
    
    (x_train, y_train) = normalize_img(x_train, y_train)
    (x_val, y_val) = normalize_img(x_val, y_val)
    (x_test, y_test) = normalize_img(x_test, y_test)
    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Create logger
    log_file, file_name = create_csv_logger_cb(folder_name=filename)

    # print headline of output file
    log_string = "loss_train;acc_train;loss_val;acc_val;loss_test;acc_test;rank1;rank2;rank3\n"
    with open(file_name, "a") as log:
        log.write(log_string)

    # load weights
    if load_model == 1:
        model.load(folder_name=folder_name)

    best_acc = 0
    best_loss = 10
    # Iterate over epochs. (Training loop)
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))
        # Iterate over the batches of the dataset.

        for step, batch_train in enumerate(train_dataset):
            # 1 evaluation
            with tf.GradientTape() as tape:
                out = model(batch_train[0], training=True)
                # softmax activation for classification
                out = tf.keras.activations.softmax(out)
                # Compute reconstruction loss
                loss = loss_fn(batch_train[1], out)
                loss += sum(model.losses)  # Add KLD regularization loss
            grads = tape.gradient(loss, model.trainable_weights)

            # Gradient update for K and L
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Network monotoring and verbosity
            loss_metric.update_state(loss)
            prediction = tf.math.argmax(out, 1)
            acc_metric.update_state(prediction, batch_train[1])

            loss_value = loss_metric.result().numpy()
            acc_value = acc_metric.result().numpy()
            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss_value))
                print("Accuracy: " + str(acc_value))

            # Reset metrics
            loss_metric.reset_state()
            acc_metric.reset_state()

        # Compute vallidation loss and accuracy

        # Validate model
        out = model(x_val, training=False)
        out = tf.keras.activations.softmax(out)
        loss = loss_fn(y_val, out)
        loss_metric.update_state(loss)
        loss_val = loss_metric.result().numpy()

        prediction = tf.math.argmax(out, 1)
        acc_metric.update_state(prediction, y_val)
        acc_val = acc_metric.result().numpy()
        print("Val Accuracy: " + str(acc_val))

        # save current model if it's the best
        if acc_val >= best_acc and loss_val <= best_loss:
            best_acc = acc_val
            best_loss = loss_val
            print("new best model with accuracy: " + str(best_acc) + " and loss " + str(best_loss))

        model.save(folder_name=folder_name_best)
        model.save(folder_name=folder_name)

        # Reset metrics
        loss_metric.reset_state()
        acc_metric.reset_state()

        # Test model
        out = model(x_test, training=False)
        out = tf.keras.activations.softmax(out)
        loss = loss_fn(y_test, out)
        loss_metric.update_state(loss)
        loss_test = loss_metric.result().numpy()

        prediction = tf.math.argmax(out, 1)
        acc_metric.update_state(prediction, y_test)
        acc_test = acc_metric.result().numpy()
        log_string = "Loss: " + str(loss_test) + "| Accuracy" + str(acc_test) + "\n"
        print("Test :" + log_string)
        # Reset metrics
        loss_metric.reset_state()
        acc_metric.reset_state()

        # Log Data of current epoch
        log_string = str(loss_value) + ";" + str(acc_value) + ";" + str(
            loss_val) + ";" + str(acc_val) + ";" + str(
            loss_test) + ";" + str(acc_test) + ";" + str(
            int(dlra_layer_dim)) + ";" + str(int(dlra_layer_dim)) + ";" + str(
            int(dlra_layer_dim)) + ";" + str(
            int(dlra_layer_dim)) + "\n"
        with open(file_name, "a") as log:
            log.write(log_string)
        print("Epoch Data :" + log_string)

    return 0


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

"""
if __name__ == '__main__':

    print("---------- Start Network Training Suite ------------")
    print("Parsing options")
    # --- parse options ---
    parser = OptionParser()

    parser.add_option("-l", "--load_model", dest="load_model", default=1)

    (options, args) = parser.parse_args()
    options.load_model = int(options.load_model)

    train()
"""