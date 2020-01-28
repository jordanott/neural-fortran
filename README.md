# Keras integration with Neural Fortran

![](https://github.com/jordanott/neural-fortran/blob/master/Figures/logo.png?raw=true)

This library is derived from [Milan Curcic's](https://github.com/jordanott/neural-fortran) original work  
I've beefed up the generalizability and added many features

## Additions
* An extendable layer type
  * The original library was only capable of a dense layer
    * Forward and backward operations occurred outside the layer (in the network module)
  * Ability to implement arbitrary layers
    * Simply extend the `layer_type` and specify these functions:
      * `forward`
      * `backward`
* Training
  * Backprop takes place inside the extended `layer_type`
  * Ability to training arbitrary cost functions
* Implemented layers
  * Dense
  * Dropout
  * Batch Normalization
* Ensembles
  * Read in a directory of network configs
  * Create a network for each config
  * Run in parallel using `$OMP PARALLEL` directives
  * Average results of all predictions in ensemble
* A bridge between Keras and Fortran
  * Convert model trained in Keras (`h5` file) to Neural Fortran
    * Any of the above layers are allowed
    * Sequential or Functional API
  * Convert Neural Fortran model back to Keras

---

## Getting started

Get the code:

```
git clone https://github.com/jordanott/neural-fortran
```

Dependencies:

* Fortran 2018-compatible compiler
* OpenCoarrays (optional, for parallel execution, gfortran only)
* BLAS, MKL (optional)

### Build
* Tests and examples will be built in the `bin/` directory
* To use a different compiler modify `FC=mpif90 cmake .. -DSERIAL=1`

```
sh build_steps.sh
```

## Examples

### Loading a model trained in Keras

```
python convert_weights.py --weights_file path/to/keras_model.h5 --output_file path/to/model_config.txt
```

This would create the `model_config.txt` file with the following:
```
9                         --> How many total layers (includes input and activations)
input	5                 --> 5 inputs
dense	3                 --> Hidden layer 1 has 3 nodes
leakyrelu	0.3       --> Hidden layer 1 activation LeakyReLU with alpha = 0.3
dense	4                 --> Hidden layer 2 has 4 nodes
leakyrelu	0.3       --> Hidden layer 2 activation LeakyReLU with alpha = 0.3
dense	3                 --> Hidden layer 3 has 3 nodes
leakyrelu	0.3       --> Hidden layer 3 activation LeakyReLU with alpha = 0.3
dense	2                 --> 2 outputs in the last layer
linear	0                 --> Linear activation with no alpha
0.5                       --> Learning rate
<BIASES>
.
.
.
<DENSE LAYER WEIGHTS>
.
.
.
<BATCH NORMALIZATION PARAMETERS>
```

### Creating a network

Architecture descriptions are specified in a config text file:
```
9                         --> How many total layers (includes input and activations)
input	5                 --> 5 inputs
dense	3                 --> Hidden layer 1 has 3 nodes
leakyrelu	0.3       --> Hidden layer 1 activation LeakyReLU with alpha = 0.3
dense	4                 --> Hidden layer 2 has 4 nodes
leakyrelu	0.3       --> Hidden layer 2 activation LeakyReLU with alpha = 0.3
dense	3                 --> Hidden layer 3 has 3 nodes
leakyrelu	0.3       --> Hidden layer 3 activation LeakyReLU with alpha = 0.3
dense	2                 --> 2 outputs in the last layer
linear	0                 --> Linear activation with no alpha
0.5                       --> Learning rate
```

Then the network configuration can be loaded into FORTRAN:
```fortran
use mod_network, only: network_type
type(network_type) :: net

call net % load('model_config.txt')
```

### Saving and loading from file

To save a network to a file, do:

```fortran
call net % save('model_config.txt')
```

Loading from file works the same way:

```fortran
call net % load('model_config.txt')
```

### Building in parallel mode

If you use gfortran and want to build neural-fortran in parallel mode,
you must first install [OpenCoarrays](https://github.com/sourceryinstitute/OpenCoarrays).
Once installed, use the compiler wrappers `caf` and `cafrun` to build and execute
in parallel, respectively:

```
FC=caf cmake ..
make
cafrun -n 4 bin/example_mnist # run MNIST example on 4 cores
```

### Building with a different compiler

If you want to build with a different compiler, such as Intel Fortran,
specify `FC` when issuing `cmake`:

```
FC=ifort cmake ..
```

### Building with BLAS or MKL

To use an external BLAS or MKL library for `matmul` calls,
run cmake like this:

```
cmake .. -DBLAS=-lblas
```

where the value of `-DBLAS` should point to the desired BLAS implementation,
which has to be available in the linking path.
This option is currently available only with gfortran.

### Building in double or quad precision

By default, neural-fortran is built in single precision mode
(32-bit floating point numbers). Alternatively, you can configure to build
in 64 or 128-bit floating point mode:

```
cmake .. -DREAL=64
```

or

```
cmake .. -DREAL=128
```

### Building in debug mode

To build with debugging flags enabled, type:

```
cmake .. -DCMAKE_BUILD_TYPE=debug
```
