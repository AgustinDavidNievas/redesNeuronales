#import os
#import tensorflow as tf
import tensorflow.contrib.eager as tfe
from ..src.parseData import parse_csv



def testData(tf,test_fp,model):

    test_dataset = tf.data.TextLineDataset(test_fp)
    test_dataset = test_dataset.skip(1)             
    test_dataset = test_dataset.map(parse_csv)      
    test_dataset = test_dataset.shuffle(1000)       
    test_dataset = test_dataset.batch(32)           
    

    test_accuracy = tfe.metrics.Accuracy()
    
    for (x, y) in tfe.Iterator(test_dataset):
        prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)
    
    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
