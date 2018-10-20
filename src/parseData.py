import tensorflow as tf

def parse_csv(line):
    
    example_defaults = [[0.], [0.], [0.], [0.], [0]]
    parsed_line = tf.decode_csv(line, example_defaults)
    
    features = tf.reshape(parsed_line[:-1], shape=(4,))
    
    label = tf.reshape(parsed_line[-1], shape=())
    
    return features, label