import tensorflow as tf
from sys import argv
from .src.training import training
from .test.testData import testData
from .test.predictions import predictions
import multiprocessing

def run_tensorflow():
    training(tf,model,train_dataset_fp)
    testData(tf,test_fp,model)
    predictions(tf,model)


train_dataset_fp = argv[1]
test_fp = argv[2]


tf.enable_eager_execution()

model = tf.keras.Sequential([#jugar con estos numeros si los resultados no dan bien...
      tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),
      tf.keras.layers.Dense(10, activation="relu"),
      tf.keras.layers.Dense(3)
    ])

p = multiprocessing.Process(target=run_tensorflow)
p.start()
p.join()