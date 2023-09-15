# '''eq Pagani'''

# equazioni, ecc

# Datagen();

# LED();


#'''LED class'''



class LED:
    def __init__(self):
        pass
    
    def build_autoencoder(dim_latent, dense, conv, pool, activation='relu'):
        
        #README: pool not wanted -> set 0
        
        conv = [(16,3),(32,5)]
        dense = [32, 64, 128]
        
        [dense, conv, pool]
        [64, (16,3), "avg"]
        #extra assert
        
        
        assert len(pool) == len(conv)
        
        E.add_layer(input)
        
        for layerC,layerP in conv,pool:
            E.add_layer(layerC)
            E.add_layer(layerP)
        
        E.add_layer(max_global_pooling)
        
        for layer in dense:
            E.add_layer(layer)
        
        E.add_layer(output)
        
        
        
        
        E = keras.encoder()
        D = keras.decoder()
        return E,D
    
    def build_RNN():
        RNN = keras.RNN
        
    def train(what_to_save='all'):
        train E,D
        
        net = E+RNN+D
        
        train net (E,D fixed)
        
        #maybe fine tune all
        
        save(...)
        
    def predict():
        pass
    
    def save_network(net, tipo_net):
    def load_network():
        for tipo in tipo_net
        if tipo == "E"
            E = np.load()
    
# class AdaLED(LED):
#     def train
#     def predict
#     def RNN




autoencoder(tfkl.Dense(units=64,))


