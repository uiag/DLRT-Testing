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
from cudatoolkit import cuda

DLRT_FR = False
DLRT_MR = False
RUNS = 1
LOAD_MODEL = 0
RANKS = [10, 10]
DIMENSIONS = [10, 21]
EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
DATASET_SIZE = 1
NOISE = 0.2
REGULARIZER = None
REGULARIZER_AMOUNT = [0.001, 0]
OPTIMIZER = "Adam"
DATASET = "mnist"
INPUT_DIM = 784
OUTPUT_DIM = 10
TIME_DELTA = 5

def main():
    if tf.test.gpu_device_name():
        print("GPU found. Using GPU")
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        print("Disabled GPU. Using CPU")
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    if not DLRT_FR:
        evaluation = [False, False]
        timeDeltas = []
        startEvTime = datetime.datetime.now() - datetime.timedelta(minutes=TIME_DELTA+1)
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
                if startEvTime + datetime.timedelta(minutes=TIME_DELTA) < datetime.datetime.now():
                    print("\n\n\n\n\n\nsaadffdddddddddddd\n\n\n\n\n")
                    evaluation = [True, False]
                    startEvTime = datetime.datetime.now()
                    cuda.cuInit(0)
                    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                startTime = datetime.datetime.now()
                mnist_reference.train(load_model=LOAD_MODEL, dim_layer=i, epochs=EPOCHS, batch_size=BATCH_SIZE, name=name, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, optimizerName=OPTIMIZER, learning_rate=LEARNING_RATE, datasetSize=DATASET_SIZE, noise=NOISE, dataset=DATASET, regularizer=REGULARIZER, regularizer_amount=REGULARIZER_AMOUNT)
                endTime = datetime.datetime.now()
                timeDelta = endTime-startTime
                timeDeltas.append(timeDelta)
                print("\nDauer: ", timeDelta)
                print("\n")
                if evaluation[1] == True:
                    evaluation = [False, False]
                    startEvTime = datetime.datetime.now()
                    if timeDeltas[-1] > timeDeltas[-2]:
                        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                    else:
                        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                if evaluation[0] == True:
                    evaluation = [False, True]
                    startEvTime = datetime.datetime.now()
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            totalEndTime = datetime.datetime.now()
            print("\nTotal Dauer: ", totalEndTime-totalStartTime)
            print("\n")
            print("Uhrzeit: ", totalEndTime)
            print("\n")
                
    elif not DLRT_MR:
        evaluation = [False, False]
        timeDeltas = []
        startEvTime = datetime.datetime.now() - datetime.timedelta(minutes=TIME_DELTA+1)
        for k in range(1, RUNS+1):
            totalStartTime = datetime.datetime.now()
            for i in range(DIMENSIONS[0], DIMENSIONS[1]):
                name = "DLRT(Rank" + str(RANKS[0]) + ")TestDim" + str(DIMENSIONS[0]) + "-" + str(DIMENSIONS[1]) + "ep" + str(EPOCHS) + "batch" + str(BATCH_SIZE) + "lr" + str(LEARNING_RATE) + "Noise" + str(NOISE*100) + "Dataset" + str(DATASET_SIZE*100)
                if OPTIMIZER != "Adam":
                    name = OPTIMIZER + "TestDim" + str(DIMENSIONS[0]) + "-" + str(DIMENSIONS[1]) + "ep" + str(EPOCHS) + "batch" + str(BATCH_SIZE) + "lr" + str(LEARNING_RATE) + "Noise" + str(NOISE*100) + "Dataset" + str(DATASET_SIZE*100)
                name += "Run" + str(k)
                print("\nDurchgang " + str(k) + " Dimension " + str(i) + "\n")
                if startEvTime + datetime.timedelta(minutes=TIME_DELTA) < datetime.datetime.now():
                    evaluation = [True, False]
                    startEvTime = datetime.datetime.now()
                    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                if evaluation[1] == True:
                    evaluation = [False, False]
                    startEvTime = datetime.datetime.now()
                    if timeDeltas[-1] > timeDeltas[-2]:
                        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                    else:
                        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                if evaluation[0] == True:
                    evaluation = [False, True]
                    startEvTime = datetime.datetime.now()
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                startTime = datetime.datetime.now()
                mnist_DLRT_fr.train(start_rank=RANKS[0], load_model=LOAD_MODEL, dim_layer=i, epochs=EPOCHS, batch_size=BATCH_SIZE, name=name, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, optimizerName=OPTIMIZER, learning_rate=LEARNING_RATE, datasetSize=DATASET_SIZE, noise=NOISE, dataset=DATASET)
                endTime = datetime.datetime.now()
                timeDelta = endTime-startTime
                timeDeltas.append(timeDelta)
                print("\nDauer: ", timeDelta)
                print("\n")
            totalEndTime = datetime.datetime.now()
            print("\nTotal Dauer: ", totalEndTime-totalStartTime)
            print("\n")
            print("Uhrzeit: ", totalEndTime)
            print("\n")
    
    else:
        evaluation = [False, False]
        timeDeltas = []
        startEvTime = datetime.datetime.now() - datetime.timedelta(minutes=TIME_DELTA+1)
        for k in range(1, RUNS+1):
            totalStartTime = datetime.datetime.now()
            for i in range(RANKS[0], RANKS[1]):
                name = "DLRT(Rank" + str(RANKS[0]) + "-" + str(RANKS[1]) + ")TestDim" + str(DIMENSIONS[0]) + "ep" + str(EPOCHS) + "batch" + str(BATCH_SIZE) + "lr" + str(LEARNING_RATE) + "Noise" + str(NOISE*100) + "Dataset" + str(DATASET_SIZE*100)
                if OPTIMIZER != "Adam":
                    name = OPTIMIZER + "TestDim" + str(DIMENSIONS[0]) + "-" + str(DIMENSIONS[1]) + "ep" + str(EPOCHS) + "batch" + str(BATCH_SIZE) + "lr" + str(LEARNING_RATE) + "Noise" + str(NOISE*100) + "Dataset" + str(DATASET_SIZE*100)
                name += "Run" + str(k)
                print("\nDurchgang " + str(k) + " Rank " + str(i) + "\n")
                if startEvTime + datetime.timedelta(minutes=TIME_DELTA) < datetime.datetime.now():
                    evaluation = [True, False]
                    startEvTime = datetime.datetime.now()
                    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                if evaluation[1] == True:
                    evaluation = [False, False]
                    startEvTime = datetime.datetime.now()
                    if timeDeltas[-1] > timeDeltas[-2]:
                        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                    else:
                        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                if evaluation[0] == True:
                    evaluation = [False, True]
                    startEvTime = datetime.datetime.now()
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                startTime = datetime.datetime.now()
                mnist_DLRT_fr.train(start_rank=i, load_model=LOAD_MODEL, dim_layer=DIMENSIONS[0], epochs=EPOCHS, batch_size=BATCH_SIZE, name=name, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, optimizerName=OPTIMIZER, learning_rate=LEARNING_RATE, datasetSize=DATASET_SIZE, noise=NOISE, dataset=DATASET)
                endTime = datetime.datetime.now()
                timeDelta = endTime-startTime
                timeDeltas.append(timeDelta)
                print("\nDauer: ", timeDelta)
                print("\n")
            totalEndTime = datetime.datetime.now()
            print("\nTotal Dauer: ", totalEndTime-totalStartTime)
            print("\n")
            print("Uhrzeit: ", totalEndTime)
            print("\n")
    
if __name__ == "__main__":
    main()