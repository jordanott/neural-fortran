# To Do


### Short Term
- [x] Parse H5 file to text file
- [x] Add activation functions from `load`
- [x] Test script with sample data
  * Match outputs from Fortran to Keras
- [x] New test script
  * Read inputs from text files
  * return output from network

### New Layers
- [ ] Dropout
- [ ] Batchnorm

### New Activation Functions
- [x] Linear

### Long Term
- [ ] Fortran H5 library to parse models directly from Keras file
- [ ] Online training within the climate model

# SPCAM3 Instructions

Where the neural network was implemented before:
  * [cloudbrain_keras_dense.F90](https://gitlab.com/mspritch/spcam3.0-neural-net/blob/nn_fbp_engy_ess/models/atm/cam/src/physics/cam1/cloudbrain_keras_dense.F90)
    * Branch: `nn_fbp_engy_ess`

Where the neural network subroutine is called:
  * [tphysbc_internallythreaded.F90](https://gitlab.com/mspritch/spcam3.0-neural-net/blob/nn_fbp_engy_ess/models/atm/cam/src/physics/cam1/tphysbc_internallythreaded.F90#L1954)

SPCAM isn't something you can run on your laptop. It has very specific compile instructions and has to be run on a cluster. So we won't be able to easily test things

### Steps
- [ ] take out [VBP](https://gitlab.com/mspritch/spcam3.0-neural-net/blob/nn_fbp_engy_ess/models/atm/cam/src/physics/cam1/cloudbrain_keras_dense.F90#L87)
  * This variable is not needed
  * In old versions it was an input to the network
- [ ] This is where the input is [normalized](https://gitlab.com/mspritch/spcam3.0-neural-net/blob/nn_fbp_engy_ess/models/atm/cam/src/physics/cam1/cloudbrain_keras_dense.F90#L131)
  * This needs to be kept
- [ ] Everything from [here](https://gitlab.com/mspritch/spcam3.0-neural-net/blob/nn_fbp_engy_ess/models/atm/cam/src/physics/cam1/cloudbrain_keras_dense.F90#L142) down should be replaced with our network
  * Where ever there is layer specific `for` loops should be taken out
- [ ] Command line argument of weights filename
