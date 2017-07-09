
mkdir -p deps
cd deps

echo "Building Redis"
git clone https://github.com/antirez/redis.git
cd redis
make
cd ..

echo "Building Tensorflow"
git clone --recurse-submodules https://github.com/tensorflow/tensorflow.git
cd tensorflow
./configure
bazel build -c opt //tensorflow:libtensorflow_cc.so

cd ..

cd ..
echo "Done"

