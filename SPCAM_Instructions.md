# SPCAM Instructions

### Modifications to SPCAM

1. Change `openmp` to `qopenmp`
  * `models/atm/cam/bld/Makefile.stampede`
  * `models/utils/esmf/build/linux_intel/base_variables`

2. Add to [Makefile.stampede](https://gitlab.com/mspritch/spcam3.0-neural-net/blob/nn_fbp_engy_ess/models/atm/cam/bld/Makefile.stampede)
```
NEURAL_MOD := -I$HOME/neural-fortran/build/include/
NEURAL_A := -L$HOME/neural-fortran/build/lib/
NEURAL_O := -L$HOME/neural-fortran/build/CMakeFiles/neural.dir/
```

3. Add `$(NEURAL_O) $(NEURAL_A) -lneural` to [line](https://gitlab.com/mspritch/spcam3.0-neural-net/blob/nn_fbp_engy_ess/models/atm/cam/bld/Makefile.stampede#L167)

4. Add `$(NEURAL_MOD)` to [line](https://gitlab.com/mspritch/spcam3.0-neural-net/blob/nn_fbp_engy_ess/models/atm/cam/bld/Makefile.stampede#L330)


### Neural-Fortran
```
cd $HOME
git clone https://github.com/jordanott/neural-fortran.git
cd neural-fortran
```

Set desired compiler in `build_steps.sh`
  * For example, using mpif90: `FC=mpif90 cmake .. -DSERIAL=1`

Then compile 
`sh build_steps.sh`


### Build SPCAM


