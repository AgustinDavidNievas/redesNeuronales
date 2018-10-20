#import tensorflow as tf

#tf.enable_eager_execution()

def predictions(tf,model):
    class_ids = ["Bajo de peso", "Peso Ideal", "Sobrepeso"]
            
    predict_dataset = tf.convert_to_tensor([
        [24.0, 1.0, 150.0,36.0,],#Bajo de peso
        [41.0, 0.0, 171.0, 81.0,],#Sobrepeso
        [62.0, 1.0, 170.0, 60.0],#Peso Ideal
        [19.0, 1.0, 120.0,50.0]#Sobrepeso
    ])

    
    """
    #Nota: los numeros aca arriba tienen que ser float, segun la documentacion por la forma en que usa la gpu para hacer cuentas
    no se lleva bien con los int, es mejor usar float.
    
    PD: aunque en los sets hay ints, hice que los tome como float X.0 por esta misma razon
    ])
    """
    
    predictions = model(predict_dataset)
    
    for i, logits in enumerate(predictions):
        class_idx = tf.argmax(logits).numpy()
        name = class_ids[class_idx]
        print("Example {} prediction: {}".format(i, name))
