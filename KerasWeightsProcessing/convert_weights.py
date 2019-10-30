import h5py
import json
import warnings
import argparse
import numpy as np

ACTIVATIONS = ['relu', 'linear', 'leakyrelu']
SUPPORTED_LAYERS = ['Dense', 'Dropout'] + ACTIVATIONS

def h5_to_txt(weights_file_name, output_file_name=''):

    bias        = []
    weights     = []
    dimensions  = []
    activations = []
    layer_types = []

    #check and open file
    with h5py.File(weights_file_name,'r') as weights_file:

        # weights of model
        model_weights = weights_file['model_weights']
        keras_version = weights_file.attrs['keras_version']

        # Decode using the utf-8 encoding; change values for eval
        model_config = weights_file.attrs['model_config'].decode('utf-8')
        model_config = model_config.replace('true','True')
        model_config = model_config.replace('false','False')
        model_config = model_config.replace('null','None')
        # convert to dictionary
        model_config = eval(model_config)

        num_layers = len(model_config['config']['layers'])

        # check what type of keras model sequential or functional
        if model_config['class_name'] == 'Model':
            layer_config = model_config['config']['layers'][1:]
        else:
            layer_config = model_config['config']['layers']
            num_layers += 1 # sequential model doesn't include an input layer

        for idx,layer in enumerate(layer_config):
            name = layer['config']['name']
            class_name = layer['class_name']
            layer_types.append(class_name)

            if class_name not in SUPPORTED_LAYERS:
                warnings.warn('Unsupported layer found! Skipping...')
                continue

            elif class_name == 'Dense':
                # get weights and biases out of dictionary
                layer_bias    = np.array(
                    model_weights[name][name]['bias:0']
                )
                layer_weights = np.array(
                    model_weights[name][name]['kernel:0']
                )

                # store bias values
                bias.append(layer_bias)
                # store weight value
                weights.append(layer_weights)

                # store first dimension for the input layer
                if idx == 0:
                    dimensions.append(
                        str(layer_weights.shape[0])
                    )
                # store dimension of hidden dim
                dimensions.append(
                    str(layer_weights.shape[1])
                )

                activation = layer['config']['activation']

                if activation not in ACTIVATIONS:
                    warnings.warn('Unsupported activation found! Replacing with Linear.')
                    activations.append('linear')
                else:
                    activations.append(activation)

            elif class_name == 'Dropout':
                pass

            elif class_name.lower() in ACTIVATIONS:
                # replace previous dense layer with the advanced activation function (LeakyReLU)
                activations[-1] = class_name.lower() + '\t' + str(layer['config']['alpha'])

    if not output_file_name:
        # if not specified will use path of weights_file with txt extension
        output_file_name = weights_file_name.replace('.h5', '.txt')

    with open(output_file_name,"w") as output_file:
        output_file.write(str(num_layers) + '\n')

        output_file.write("\t".join(dimensions) + '\n')

        for b in bias:
            bias_str = '\t'.join(
                '{:0.6e}'.format(num) for num in b.tolist()
            )
            output_file.write(bias_str + '\n')

        for w in weights:
            weights_str = '\t'.join(
                '{:0.6e}'.format(num) for num in w.T.flatten()
            )
            output_file.write(weights_str + '\n')

        for a in activations:
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
