# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 00:13:31 2023

@author: jakob
"""

import mnist_DLRT_fr
import mnist_reference
import datetime
import os
import tensorflow as tf

DLRT_FR = False		# run DLRT with one fixed rank
DLRT_MR = False		# run DLRT for multiple ranks (only possible for DLRT_FR=False)(DLRT_FR=False and DLRT_MR=False -> reference network)
RUNS = 1		# specify the number of runs
LOAD_MODEL = 0		# 1 for loading a model (not advised)
RANKS = [10, 11]	# specify the first and last rank (only possible for DLRT)
DIMENSIONS = [5000, 5001]	# specify the first and last dimension
EPOCHS = 250		# specify the amount of epochs
BATCH_SIZE = 256	# specify the batch size
LEARNING_RATE = 1e-3	# specify the learning rate
DATASET_SIZE = 0.1	# specify the dataset size (test, train and validation set size will be multiplied by this number)
NOISE = 0.2		# specify the amount of noise (random labels)
REGULARIZER = None	# specify the regularizer (possible: None, "L1", "L2", "L1L2")
REGULARIZER_AMOUNT = [0.005, 0.001]	# specify the regularization parameters
OPTIMIZER = "Adam"	# specify the optimizer
DATASET = "mnist"	# specify the dataset
INPUT_DIM = 784		# specify the input dimension
OUTPUT_DIM = 10		# specify the output dimension

VAL_SIZE=10000		# absolute size of validation set (prior to adjusting dataset size)
VAL_FULL=True		# if true validation set will be removed from test set. Otherwise it will be copied from the test set

MEMORY = 2048		# used GPU memory

def main():
    """
    if tf.test.gpu_device_name():
        print("GPU found. Using GPU")
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        print("Disabled GPU. Using CPU")
    
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=MEMORY)])

    if not DLRT_FR:
        for k in range(1, RUNS+1):
            totalStartTime = datetime.datetime.now()
            for i in range(DIMENSIONS[0], DIMENSIONS[1]):
                name = "TestDim" + str(DIMENSIONS[0]) + "-" + str(DIMENSIONS[1]) + "ep" + str(EPOCHS) + "batch" + str(BATCH_SIZE) + "lr" + str(LEARNING_RATE) + "Noise" + str(NOISE*100) + "Dataset" + str(DATASET_SIZE*100)
                if OPTIMIZER != "Adam":
                    name = OPTIMIZER + "TestDim" + str(DIMENSIONS[0]) + "-" + str(DIMENSIONS[1]) + "ep" + str(EPOCHS) + "batch" + str(BATCH_SIZE) + "lr" + str(LEARNING_RATE) + "Noise" + str(NOISE*100) + "Dataset" + str(DATASET_SIZE*100)
                if REGULARIZER:
                    if REGULARIZER == "L1L2":
                        name += "Reg" + REGULARIZER + "-" + str(REGULARIZER_AMOUNT[0]*100) + ";" + str(REGULARIZER_AMOUNT[1]*100)
                    else:
                        name += "Reg" + REGULARIZER + "-" + str(REGULARIZER_AMOUNT[0]*100)
                name += "Run" + str(k)
                print("\nDurchgang " + str(k) + " Dimension " + str(i) + "\n")
                startTime = datetime.datetime.now()
                mnist_reference.train(load_model=LOAD_MODEL, dim_layer=i, epochs=EPOCHS, batch_size=BATCH_SIZE, name=name, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, optimizerName=OPTIMIZER, learning_rate=LEARNING_RATE, datasetSize=DATASET_SIZE, noise=NOISE, dataset=DATASET, regularizer=REGULARIZER, regularizer_amount=REGULARIZER_AMOUNT, val_size=VAL_SIZE, val_full=VAL_FULL)
                endTime = datetime.datetime.now()
                timeDelta = endTime-startTime
                print("\nDauer: ", timeDelta)
                print("\n")
            totalEndTime = datetime.datetime.now()
            print("\nTotal Dauer: ", totalEndTime-totalStartTime)
            print("\n")
            print("Uhrzeit: ", totalEndTime)
            print("\n")
                
    elif not DLRT_MR:
        for k in range(1, RUNS+1):
            totalStartTime = datetime.datetime.now()
            for i in range(DIMENSIONS[0], DIMENSIONS[1]):
                name = "DLRT(Rank" + str(RANKS[0]) + ")TestDim" + str(DIMENSIONS[0]) + "-" + str(DIMENSIONS[1]) + "ep" + str(EPOCHS) + "batch" + str(BATCH_SIZE) + "lr" + str(LEARNING_RATE) + "Noise" + str(NOISE*100) + "Dataset" + str(DATASET_SIZE*100)
                if OPTIMIZER != "Adam":
                    name = OPTIMIZER + "TestDim" + str(DIMENSIONS[0]) + "-" + str(DIMENSIONS[1]) + "ep" + str(EPOCHS) + "batch" + str(BATCH_SIZE) + "lr" + str(LEARNING_RATE) + "Noise" + str(NOISE*100) + "Dataset" + str(DATASET_SIZE*100)
                name += "Run" + str(k)
                print("\nDurchgang " + str(k) + " Dimension " + str(i) + "\n")
                startTime = datetime.datetime.now()
                mnist_DLRT_fr.train(start_rank=RANKS[0], load_model=LOAD_MODEL, dim_layer=i, epochs=EPOCHS, batch_size=BATCH_SIZE, name=name, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, optimizerName=OPTIMIZER, learning_rate=LEARNING_RATE, datasetSize=DATASET_SIZE, noise=NOISE, dataset=DATASET, val_size=VAL_SIZE, val_full=VAL_FULL)
                endTime = datetime.datetime.now()
                timeDelta = endTime-startTime
                print("\nDauer: ", timeDelta)
                print("\n")
            totalEndTime = datetime.datetime.now()
            print("\nTotal Dauer: ", totalEndTime-totalStartTime)
            print("\n")
            print("Uhrzeit: ", totalEndTime)
            print("\n")
    
    else:
        for k in range(1, RUNS+1):
            totalStartTime = datetime.datetime.now()
            for i in range(RANKS[0], RANKS[1]):
                name = "DLRT(Rank" + str(RANKS[0]) + "-" + str(RANKS[1]) + ")TestDim" + str(DIMENSIONS[0]) + "ep" + str(EPOCHS) + "batch" + str(BATCH_SIZE) + "lr" + str(LEARNING_RATE) + "Noise" + str(NOISE*100) + "Dataset" + str(DATASET_SIZE*100)
                if OPTIMIZER != "Adam":
                    name = OPTIMIZER + "TestDim" + str(DIMENSIONS[0]) + "-" + str(DIMENSIONS[1]) + "ep" + str(EPOCHS) + "batch" + str(BATCH_SIZE) + "lr" + str(LEARNING_RATE) + "Noise" + str(NOISE*100) + "Dataset" + str(DATASET_SIZE*100)
                name += "Run" + str(k)
                print("\nDurchgang " + str(k) + " Rank " + str(i) + "\n")
                startTime = datetime.datetime.now()
                mnist_DLRT_fr.train(start_rank=i, load_model=LOAD_MODEL, dim_layer=DIMENSIONS[0], epochs=EPOCHS, batch_size=BATCH_SIZE, name=name, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, optimizerName=OPTIMIZER, learning_rate=LEARNING_RATE, datasetSize=DATASET_SIZE, noise=NOISE, dataset=DATASET, val_size=VAL_SIZE, val_full=VAL_FULL)
                endTime = datetime.datetime.now()
                timeDelta = endTime-startTime
                print("\nDauer: ", timeDelta)
                print("\n")
            totalEndTime = datetime.datetime.now()
            print("\nTotal Dauer: ", totalEndTime-totalStartTime)
            print("\n")
            print("Uhrzeit: ", totalEndTime)
            print("\n")
    
if __name__ == "__main__":
    main()
