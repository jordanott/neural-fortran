# Keras integration with Neural Fortran



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
* Implemented layers
  * Dense
  * Dropout
  * Batch Normalization
* Ensembles
  * Read in a directory of network configs
  * Create a network for each config
  * Average results of all predictions in ensemble
* A bridge between Keras and Fortran
  * Convert model trained in Keras (`h5` file) to Neural Fortran
    * Any of the above layers are allowed 
    * Sequential or Functional API
  * Convert Neural Fortran model back to Keras

---
## neural-fortran

* [Getting started](https://github.com/jordanott/neural-fortran#getting-started)
  - [Building in serial mode](https://github.com/jordanott/neural-fortran#building-in-serial-mode)
  - [Building in parallel mode](https://github.com/jordanott/neural-fortran#building-in-parallel-mode)
  - [Building with a different compiler](https://github.com/jordanott/neural-fortran#building-with-a-different-compiler)
  - [Building with BLAS or MKL](https://github.com/jordanott/neural-fortran#building-with-blas-or-mkl)
  - [Building in double or quad precision](https://github.com/jordanott/neural-fortran#building-in-double-or-quad-precision)
  - [Building in debug mode](https://github.com/jordanott/neural-fortran#building-in-debug-mode)
* [Examples](https://github.com/jordanott/neural-fortran#examples)
  - [Creating a network](https://github.com/jordanott/neural-fortran#creating-a-network)
  - [Training the network](https://github.com/jordanott/neural-fortran#training-the-network)
  - [Saving and loading from file](https://github.com/jordanott/neural-fortran#saving-and-loading-from-file)


## Getting started

Get the code:

```
git clone https://github.com/jordanott/neural-fortran
```

Dependencies:

* Fortran 2018-compatible compiler
* OpenCoarrays (optional, for parallel execution, gfortran only)
* BLAS, MKL (optional)

### Building in serial mode

```
cd neural-fortran
mkdir build
cd build
cmake .. -DSERIAL=1
make
```

Tests and examples will be built in the `bin/` directory.

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

## Examples

### Creating a network

Creating a network with 3 layers,
one input, one hidden, and one output layer,
with 3, 5, and 2 neurons each:

```fortran
use mod_network, only: network_type
type(network_type) :: net
net = network_type([3, 5, 2])
```

### Setting the activation function

By default, the network will be initialized with the sigmoid activation
function for all layers. You can specify a different activation function:

```fortran
net = network_type([3, 5, 2], activation='tanh')
```

or set it after the fact:

```fortran
net = network_type([3, 5, 2])
call net % set_activation('tanh')
```

It's possible to set different activation functions for each layer.
For example, this snippet will create a network with a Gaussian
activation functions for all layers except the output layer,
and a RELU function for the output layer:

```fortran
net = network_type([3, 5, 2], activation='gaussian')
call net % layers(3) % set_activation('relu')
```

Available activation function options are: `gaussian`, `relu`, `sigmoid`,
`step`, and `tanh`.
See [mod_activation.f90](https://github.com/jordanott/neural-fortran/blob/master/src/lib/mod_activation.F90)
for specifics.

### Saving and loading from file

To save a network to a file, do:

```fortran
call net % save('my_net.txt')
```

Loading from file works the same way:

```fortran
call net % load('my_net.txt')
```

### Synchronizing networks in parallel mode

When running in parallel mode, you may need to synchronize the weights
and biases between images. You can do it like this:

```fortran
call net % sync(1)
```

The argument to `net % sync()` refers to the source image from which to
broadcast. It can be any positive number not greater than `num_images()`.
