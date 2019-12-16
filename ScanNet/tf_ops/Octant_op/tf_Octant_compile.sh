#/bin/bash
/home/jzn/local_install/cuda-8/bin/nvcc Octant.cu -o Octant_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.4
g++ -std=c++11 main.cpp Octant_g.cu.o -o tf_Octant_so.so -shared -fPIC -I /home/jzn/anaconda3/lib/python3.6/site-packages/tensorflow/include -I /home/jzn/local_install/cuda-8/include -I /home/jzn/anaconda3/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /home/jzn/local_install/cuda-8/lib64/ -L/home/jzn/anaconda3/lib/python3.6/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
