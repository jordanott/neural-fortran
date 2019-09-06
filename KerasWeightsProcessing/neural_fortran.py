import h5py
import json
import argparse
import numpy as np

def get_available_layers(layers, available_layers=[b"dense"]):
    parsed_layers = []
    for l in layers:
        for g in available_layers:
            if g in l:
                parsed_layers.append(l)
    return parsed_layers

def h5_to_txt(weights_file_name, output_file_name=''):

    #check and open file
    with h5py.File(weights_file_name,'r') as weights_file:

        weights_group_key=list(weights_file.keys())[0]

        # activation function information in model_config
        model_config = weights_file.attrs['model_config'].decode('utf-8') # Decode using the utf-8 encoding
        model_config = model_config.replace('true','True')
        model_config = model_config.replace('false','False')

        model_config = model_config.replace('null','None')
        model_config = eval(model_config)

        layers = list(weights_file['model_weights'].attrs['layer_names'])
        layers = get_available_layers(layers)
        print("names of layers in h5 file: %s \n" % layers)

        # attributes needed for .txt file
        # number of layers + 1(Fortran includes input layer),
        #   dimensions, biases, weights, and activations
        num_layers = len(layers)+1

        dimensions = []
        bias = {}
        weights = {}
        activations = []

        print('Processing the following {} layers: \n{}\n'.format(len(layers),layers))
        if 'Input' in model_config['config']['layers'][0]['class_name']:
            model_config = model_config['config']['layers'][1:]
        else:
            model_config = model_config['config']['layers']

        for num,l in enumerate(layers):
            layer_info_keys=list(weights_file[weights_group_key][l][l].keys())

            #layer_info_keys should have 'bias:0' and 'kernel:0'
            for key in layer_info_keys:
                if "bias" in key:
                    bias.update({num:np.array(weights_file[weights_group_key][l][l][key])})

                elif "kernel" in key:
                    weights.update({num:np.array(weights_file[weights_group_key][l][l][key])})
                    if num == 0:
                        dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[0]))
                        dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[1]))
                    else:
                        dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[1]))

            if 'Dense' in model_config[num]['class_name']:
                activations.append(model_config[num]['config']['activation'])
            else:
                print('Skipping bad layer: \'{}\'\n'.format(model_config[num]['class_name']))

    if not output_file_name:
        # if not specified will use path of weights_file with txt extension
        output_file_name = weights_file_name.replace('.h5', '.txt')

    with open(output_file_name,"w") as output_file:
        output_file.write(str(num_layers) + '\n')

        output_file.write("\t".join(dimensions) + '\n')
        if bias:
            for x in range(len(layers)):
                bias_str="\t".join(list(map(str,bias[x].tolist())))
                output_file.write(bias_str + '\n')
        if weights:
            for x in range(len(layers)):
                weights_str="\t".join(list(map(str,weights[x].T.flatten())))
                output_file.write(weights_str + '\n')
        if activations:
            for a in activations:
                if a == 'softmax':
                    print('WARNING: Softmax activation not allowed... Replacing with Linear activation')
                    a = 'linear'
                output_file.write(a + "\n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_file", default='examples/mnist_example_input_layer.h5', type=str)
    parser.add_argument('--output_file', default='', type=str)
    args =  parser.parse_args()

    h5_to_txt(
        weights_file_name=args.weights_file,
        output_file_name=args.output_file
    )
