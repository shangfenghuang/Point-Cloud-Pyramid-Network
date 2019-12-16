# TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /hom -I /home/jzn/local_install/cuda-8/include -lcudart -L /home/jzn/local_install/cuda-8/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/jzn/anaconda3/lib/python3.6/site-packages/tensorflow/include -I /home/jzn/local_install/cuda-8/include -I /home/jzn/anaconda3/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /home/jzn/local_install/cuda-8/lib64/ -L/home/jzn/anaconda3/lib/python3.6/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
