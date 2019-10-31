import h5py
import json
import warnings
import argparse
import numpy as np

ACTIVATIONS = ['relu', 'linear', 'leakyrelu']
SUPPORTED_LAYERS = ['dense', 'dropout'] + ACTIVATIONS

def h5_to_txt(weights_file_name, output_file_name=''):

    info_str    = '{name}\t{info}\n'
    bias        = []
    weights     = []
    layer_info  = []

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

        # store first dimension for the input layer
        layer_info.append(
            info_str.format(
                name = 'input',
                info = model_config['config']['layers'][0]['config']['batch_input_shape'][1]
            )
        )

        # check what type of keras model sequential or functional
        if model_config['class_name'] == 'Model':
            layer_config = model_config['config']['layers'][1:]
        else:
            layer_config = model_config['config']['layers']

        for idx,layer in enumerate(layer_config):
            name = layer['config']['name']
            class_name = layer['class_name'].lower()

            print class_name

            if class_name not in SUPPORTED_LAYERS:
                warnings.warn('Unsupported layer found! Skipping...')
                continue

            elif class_name == 'dense':
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

                activation = layer['config']['activation']

                if activation not in ACTIVATIONS:
                    warnings.warn('Unsupported activation found! Replacing with Linear.')
                    activation = 'linear'

                # store dimension of hidden dim
                layer_info.append(
                    info_str.format(
                        name = class_name,
                        info = layer_weights.shape[1]
                    )
                )
                # add information about the activation
                layer_info.append(
                    info_str.format(
                        name = activation,
                        info = 0
                    )
                )

            elif class_name == 'dropout':
                layer_info.append(
                    info_str.format(
                        name = class_name,
                        info = layer['config']['rate']
                    )
                )

            elif class_name in ACTIVATIONS:
                print layer_info[-1]
                # replace previous dense layer with the advanced activation function (LeakyReLU)
                layer_info[-1] = info_str.format(
                    name = class_name,
                    info = layer['config']['alpha']
                )

    if not output_file_name:
        # if not specified will use path of weights_file with txt extension
        output_file_name = weights_file_name.replace('.h5', '.txt')

    with open(output_file_name,"w") as output_file:
        output_file.write(str(len(layer_info)) + '\n')

        output_file.write(
            ''.join(layer_info)
        )

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_file", default='examples/mnist_example_input_layer.h5', type=str)
    parser.add_argument('--output_file', default='', type=str)
    args =  parser.parse_args()

    h5_to_txt(
        weights_file_name=args.weights_file,
        output_file_name=args.output_file
    )
