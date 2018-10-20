import tensorflow.contrib.eager as tfe
from .parseData import *

def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets):
    with tfe.GradientTape(persistent=True) as tape:
        loss_value = loss(model, inputs, targets)
        return tape.gradient(loss_value, model.variables)


def training(tf,model,train_dataset_fp):
    
    train_dataset = tf.data.TextLineDataset(train_dataset_fp)
    train_dataset = train_dataset.skip(1)             
    train_dataset = train_dataset.map(parse_csv)      
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(32)
    
    features, label = tfe.Iterator(train_dataset).next()
    print("example features:", features[0])
    print("example label:", label[0])
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    
    
    train_loss_results = []
    train_accuracy_results = []
    
    num_epochs = 201
    
    for epoch in range(num_epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()
    
        
        for x, y in tfe.Iterator(train_dataset):
            
            grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.variables),
                                        global_step=tf.train.get_or_create_global_step())
        
            
            epoch_loss_avg(loss(model, x, y))  
            
            epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)
        
            
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())
        
        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))
