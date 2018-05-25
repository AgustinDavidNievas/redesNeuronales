#import os
#import tensorflow as tf
import tensorflow.contrib.eager as tfe
from ..src.parseData import parse_csv

#tf.enable_eager_execution()

#test_url = "http://download.tensorflow.org/data/iris_test.csv"

#test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
#                                  origin=test_url)

def testData(tf,test_fp,model):

    test_dataset = tf.data.TextLineDataset(test_fp)
    test_dataset = test_dataset.skip(1)             # skip header row
    test_dataset = test_dataset.map(parse_csv)      # parse each row with the funcition created earlier
    test_dataset = test_dataset.shuffle(1000)       # randomize
    test_dataset = test_dataset.batch(32)           # use the same batch size as the training set
    
    #print("Local copy of the datatest file: {}".format(test_fp))
    
    """
    model = tf.keras.Sequential([#see this numbers...
      tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),  # input shape required
      tf.keras.layers.Dense(10, activation="relu"),
      tf.keras.layers.Dense(3)
    ])
    """
    
    test_accuracy = tfe.metrics.Accuracy()
    
    for (x, y) in tfe.Iterator(test_dataset):
        prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)
    
    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
