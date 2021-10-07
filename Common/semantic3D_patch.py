# -*- coding: utf-8 -*-
# @Time        : 01/08/2021 10:00 PM
# @Description :
# @Author      : rui wang
# @Email       : 42017015@mail.dlut.edu.cn
import os
import h5py
import numpy as np
import vispy.scene
from vispy.scene import visuals
import random
import time
from glob import glob
import pc_util

def knn_partition(input_xyz, patch_point_nums, patch_num_ratio):
    farthest_sampler = pc_util.FarthestSampler()
    seed1_num = int(input_xyz.shape[0] / patch_point_nums * patch_num_ratio)
    # FPS sampling
    seed = farthest_sampler(input_xyz, seed1_num)
    dist = np.linalg.norm(x=seed, ord=2, axis=1)
    maxdis = max(dist)

    seed_list = seed[dist < 0.95 * maxdis]
    patches = pc_util.extract_knn_patch(seed_list, input_xyz, patch_point_nums)
    return patches


def draw_patches(patches):
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    for patch_idx in range(patches.shape[0]):
        color_list = [0.4, 0.6, 0.8, 0.1]
        scatter = visuals.Markers()
        scatter.set_data(patches[patch_idx], edge_color=(random.choice(color_list), random.choice(color_list),
                                                         random.choice(color_list)),
                         face_color=(1, 1, 1, 0.5), size=3)
        view.add(scatter)
    # or try 'arcball'
    view.camera = 'turntable'
    axis = visuals.XYZAxis(parent=view.scene)
    vispy.app.run()


def print_h5(h5file):
    for key in h5file.keys():
        print(h5file[key].name, h5file[key].shape)


def semantic3D_partition(path):
    files = glob(path+'/*.'+'h5')
    for h5_file in files:
        h5f = h5py.File(file_name, 'r')
        for i in range(3,9):
            one_class = h5f[str(i)]
            sparse_patches = knn_partition(one_class, 1024, 3)
            #draw_patches(sparse_patches)


def split_semantic3d_by_label(directory, name):
    labels_file = open(os.path.join(directory, name + '.labels'))
    labels = labels_file.readlines()
    points_file = open(os.path.join(directory, name + '.txt'))

    for i in range(9):
        exec(f'file_{i} = open(os.path.join(directory, name + "-" + str({i}) + ".xyz"), "w")')
        exec(f'buff_{i} = ""')

    BUFFER_SIZE = 10000
    for i, label in enumerate(labels):
        label = label.split()[0]
        point = points_file.readline()
        exec(f'buff_{label} += (" ".join(point.split()[0:3]) + "\\n")')
        if i % BUFFER_SIZE == 0 or i == len(labels) - 1:
            print(i)
            for j in range(9):
                exec(f'file_{j}.write(buff_{j})')
                exec(f'buff_{j} = ""')
    points_file.close()
    for i in range(9):
        exec(f'file_{i}.close')




def pu1k_factory(pu1k_train_file_path, h5file):
    pu1k = h5py.File(pu1k_train_file_path, 'r')
    h5file.create_dataset(name='pu1k', shape=[69000, 1024, 3], data=pu1k['poisson_1024'].value)


def file1_factory():
    h5f = h5py.File('kitti_256-pu1k_1024.h5', 'r+')

    # kitti_factory_from_xyz('raw/kitti-pick/train', h5f)
    # pu1k_factory('/home/era/workspace/project/pu-gan/data/pu1k/train/pu1k_poisson_256_poisson_1024_pc_2500'
        #         '_patch50_addpugan.h5', h5f)
    # repair(h5f)
    print_h5(h5f)
    h5f.close()


def semantic3d_factory(path, h5file):
    semantic3d = h5file.create_dataset(name='kitti', shape=[0, 1024, 3], maxshape=[None, 1024, 3])
    for file in os.listdir(path):
        if not file.endswith('.xyz'):
            continue
        points = np.loadtxt(os.path.join(path, file))
        if len(points) < 1024:
            continue
        patches = knn_partition(points, 1024, 3)
        print(patches.shape)
        # draw_patches(patches)
        cur_size = semantic3d.shape[0]
        semantic3d.resize([patches.shape[0]+cur_size, 1024, 3])
        semantic3d[cur_size:patches.shape[0]+cur_size] = patches
    pass


def file2_factory():
    h5f = h5py.File('semantic3d_1024.h5', 'w')
    semantic3d_factory('raw/semantic3d/tmp', h5f)
    h5f.close()
    pass


def repair(h5file):
    keys = []
    for key in h5file.keys():
        keys.append(key)
        print(h5file[key].name, h5file[key].shape)
    len1 = h5file[keys[0]].shape[0]
    len2 = h5file[keys[1]].shape[0]
    data1 = h5file[keys[0]]
    if len1 > len2:
        data1 = data1[0:len2, ...]
    elif len1 < len2:
        tmp = data1[0:len2-len1, ...]
        data1.resize([len2, data1.shape[1], data1.shape[2]])
        data1[len1:len2] = tmp
    print_h5(h5file)
    h5file.close
    

def semantic3D_to_hdf5(file_name):
    start_time = time.time()
    points = np.loadtxt(file_name)[:,0:3]
    duration = time.time() - start_time
    print('time duration1=',duration)
    
    start_time = time.time()
    nums= np.shape(points)[0]
    h5f = h5py.File('semantic3d.h5', 'w')
    semantic3d = h5f.create_dataset(name='semantic3d', shape=[nums, 3])
    semantic3d = points
    #h5f['semantic3d'] = points
    h5f.close()
    duration = time.time() - start_time
    print('time duration2=',duration)

    start_time = time.time()
    h5f = h5py.File('semantic3d.h5', 'r')
    points = h5f['semantic3d']
    h5f.close()
    duration = time.time() - start_time
    print('time duration3=',duration)


if __name__ == '__main__':
    # file1_factory()
    # file2_factory()
    #h5f = h5py.File('semantic3d_1024.h5', 'r')
    # print_h5(h5f)
    # split_semantic3d_by_label('raw/semantic3d', 'bildstein_station1_xyz_intensity_rgb')
    path = '/root/data/data/semantic3D/'
    semantic3D_partition(path)
    pass
