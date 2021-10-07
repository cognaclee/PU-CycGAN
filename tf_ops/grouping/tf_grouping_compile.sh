#!/usr/bin/env bash
#/bin/bash

TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

/usr/local/cuda/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I $TF_LIB/include  -I /usr/local/cuda/include -I $TF_LIB/include/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ -L $TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1

