#import tensorflow as tf

#tf.enable_eager_execution()

def predictions(tf,model):
    class_ids = ["Iris setosa", "Iris versicolor", "Iris virginica"]
    
    predict_dataset = tf.convert_to_tensor([
        [5.1, 3.3, 1.7, 0.5,],
        [5.9, 3.0, 4.2, 1.5,],
        [6.9, 3.1, 5.4, 2.1]
    ])
    
    """
    model = tf.keras.Sequential([#see this numbers...
      tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),  # input shape required
      tf.keras.layers.Dense(10, activation="relu"),
      tf.keras.layers.Dense(3)
    ])
    """
    
    predictions = model(predict_dataset)
    
    for i, logits in enumerate(predictions):
        class_idx = tf.argmax(logits).numpy()
        name = class_ids[class_idx]
        print("Example {} prediction: {}".format(i, name))
