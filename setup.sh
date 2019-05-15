rm -rf build
mkdir build
cd build
cmake .. -DSERIAL=1
make

cp -r ../data .
