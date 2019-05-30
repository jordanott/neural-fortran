# read my_net.txt

import numpy as np
import json
import h5py

def safe_layers(layers):
    good_layers = [b"dense"]
    safe = []
    for l in layers:
        for g in good_layers:
            if g in l:
                safe.append(l)
    return safe

def read_h5file(filename, newmodel):

    #check and open file
    f = h5py.File(filename,'r')

    weights_group_key=list(f.keys())[0]

    # activation function information in model_config
    model_config = f.attrs['model_config'].decode('utf-8') # Decode using the utf-8 encoding
    model_config = model_config.replace('true','True')
    model_config = model_config.replace('null','None')
    model_config = eval(model_config)

    layers=list(f['model_weights'].attrs['layer_names'])
    layers = safe_layers(layers)
    print("names of layers in h5 file: %s \n" % layers)

    # attributes needed for .txt file
    # number of layers + 1(Fortran includes input layer),
    #   dimensions, biases, weights, and activations
    num_layers = len(layers)+1
    dimensions = []
    bias = {}
    weights = {}
    activations = []

    for num,l in enumerate(layers):
        layer_info_keys=list(f[weights_group_key][l][l].keys())

        #layer_info_keys should have 'bias:0' and 'kernel:0'
        for key in layer_info_keys:
            if "bias" in key:
                bias.update({num:np.array(f[weights_group_key][l][l][key])})

            elif "kernel" in key:
                weights.update({num:np.array(f[weights_group_key][l][l][key])})

                if num == 0:
                    dimensions.append(str(np.array(f[weights_group_key][l][l][key]).shape[0]))
                    dimensions.append(str(np.array(f[weights_group_key][l][l][key]).shape[1]))
                else:
                    dimensions.append(str(np.array(f[weights_group_key][l][l][key]).shape[1]))
        # print model_config['config'].keys()
        activations.append(model_config['config']['layers'][num]['config']['activation'])


    newmodel.write(str(num_layers) + "\n")

    newmodel.write("\t".join(dimensions) + "\n")
    if bias:
        for x in range(len(layers)):
            bias_str="\t".join(list(map(str,bias[x].tolist())))
            newmodel.write(bias_str + "\n")
    if weights:
        for x in range(len(layers)):
            weights_str="\t".join(list(map(str,weights[x].T.flatten())))
            newmodel.write(weights_str + "\n")
    if activations:
        for a in activations:
            newmodel.write(a + "\n")



# print("\n\n\n")
kerasfile = 'keras10_io.h5' #input("enter .h5 file from keras to convert: ")
newfile = 'new_keras10_io.txt' #input("enter name for new textfile: ")
newmodel = open(newfile,"w")
read_h5file(kerasfile,newmodel)

newmodel.close()
