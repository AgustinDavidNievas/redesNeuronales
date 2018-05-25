import tensorflow as tf
from sys import argv
from .src.training import training
from .test.testData import testData
from .test.predictions import predictions

train_dataset_fp = argv[1]
test_fp = argv[2]

tf.enable_eager_execution()

model = tf.keras.Sequential([#see this numbers...
      tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),
      tf.keras.layers.Dense(10, activation="relu"),
      tf.keras.layers.Dense(3)
    ])


training(tf,model,train_dataset_fp)
testData(tf,test_fp,model)
predictions(tf,model)
