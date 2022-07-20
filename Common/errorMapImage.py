import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt 
import math
from glob import glob
import os
import time
from numba import jit

def points2setDis(pred, gt,gridsize=[40,40,40]):
    nA = pred.shape[0]
    nB = gt.shape[0]
    min_px, max_px= np.min(pred[:,0]),np.max(pred[:,0])
    min_py, max_py= np.min(pred[:,1]),np.max(pred[:,1])
    min_pz, max_pz= np.min(pred[:,2]),np.max(pred[:,2])

    min_gx, max_gx= np.min(gt[:,0]),np.max(gt[:,0])
    min_gy, max_gy= np.min(gt[:,1]),np.max(gt[:,1])
    min_gz, max_gz= np.min(gt[:,2]),np.max(gt[:,2])

    min_x = min_px if min_px<min_gx else min_gx
    min_y = min_py if min_py<min_gy else min_gy
    min_z = min_pz if min_pz<min_gz else min_gz
    max_x = max_px if max_px>max_gx else max_gx
    max_y = max_py if max_py>max_gy else max_gy
    max_z = max_pz if max_pz>max_gz else max_gz

    range_x = max_x-min_x
    range_y = max_y-min_y
    range_z = max_z-min_z

    steps = [range_x/gridsize[0],range_y/gridsize[1],range_z/gridsize[2]]
    blockp = [[]]*gridsize[0]*gridsize[1]*gridsize[2]
    blockg = [[]]*gridsize[0]*gridsize[1]*gridsize[2]
    for i in range(nA):
        indX = math.floor((pred[i,0]-min_x)/steps[0])
        indY = math.floor((pred[i,1]-min_y)/steps[1])
        indZ = math.floor((pred[i,2]-min_z)/steps[2])
        if indX == gridsize[0]:
            indX -= 1
        if indY == gridsize[1]:
            indY -= 1
        if indZ == gridsize[2]:
            indZ -= 1
        blockp[indZ+indY*gridsize[2]+indX*gridsize[1]*gridsize[2]].append(i)

    for i in range(nB):
        indX = math.floor((gt[i,0]-min_x)/steps[0])
        indY = math.floor((gt[i,1]-min_y)/steps[1])
        indZ = math.floor((gt[i,2]-min_z)/steps[2])
        if indX == gridsize[0]:
            indX -= 1
        if indY == gridsize[1]:
            indY -= 1
        if indZ == gridsize[2]:
            indZ -= 1
        blockg[indZ+indY*gridsize[2]+indX*gridsize[1]*gridsize[2]].append(i)

    first = np.zeros(nA)
    second = np.zeros(nB)
    for i in range(nA):
        cmin = np.inf
        indX = math.floor((pred[i,0]-min_x)/steps[0])
        indY = math.floor((pred[i,1]-min_y)/steps[1])
        indZ = math.floor((pred[i,2]-min_z)/steps[2])
        if indX == gridsize[0]:
            indX -= 1
        if indY == gridsize[1]:
            indY -= 1
        if indZ == gridsize[2]:
            indZ -= 1
        neib = blockg[indZ+indY*gridsize[2]+indX*gridsize[1]*gridsize[2]]
        if indX<gridsize[0]-1:
            neib += blockg[indZ+indY*gridsize[2]+(indX+1)*gridsize[1]*gridsize[2]]
        if indX>0:
            neib += blockg[indZ+indY*gridsize[2]+(indX-1)*gridsize[1]*gridsize[2]]
        if indY<gridsize[1]-1:
            neib += blockg[indZ+(indY+1)*gridsize[2]+indX*gridsize[1]*gridsize[2]]
        if indY>0:
            neib += blockg[indZ+(indY-1)*gridsize[2]+indX*gridsize[1]*gridsize[2]]
        if indZ<gridsize[2]-1:
            neib += blockg[(indZ+1)+indY*gridsize[2]+indX*gridsize[1]*gridsize[2]]
        if indZ>0:
            neib += blockg[(indZ-1)+indY*gridsize[2]+indX*gridsize[1]*gridsize[2]]
        for j in neib:
            d = np.linalg.norm(pred[i,:]-gt[j,:])
            if d<cmin:
                cmin = d
        first[i] = cmin
    
    for i in range(nB):
        cmin = np.inf
        indX = math.floor((gt[i,0]-min_x)/steps[0])
        indY = math.floor((gt[i,1]-min_y)/steps[1])
        indZ = math.floor((gt[i,2]-min_z)/steps[2])
        if indX == gridsize[0]:
            indX -= 1
        if indY == gridsize[1]:
            indY -= 1
        if indZ == gridsize[2]:
            indZ -= 1
        neib = blockp[indZ+indY*gridsize[2]+indX*gridsize[1]*gridsize[2]]
        if indX<gridsize[0]-1:
            neib += blockp[indZ+indY*gridsize[2]+(indX+1)*gridsize[1]*gridsize[2]]
        if indX>0:
            neib += blockp[indZ+indY*gridsize[2]+(indX-1)*gridsize[1]*gridsize[2]]
        if indY<gridsize[1]-1:
            neib += blockp[indZ+(indY+1)*gridsize[2]+indX*gridsize[1]*gridsize[2]]
        if indY>0:
            neib += blockp[indZ+(indY-1)*gridsize[2]+indX*gridsize[1]*gridsize[2]]
        if indZ<gridsize[2]-1:
            neib += blockp[(indZ+1)+indY*gridsize[2]+indX*gridsize[1]*gridsize[2]]
        if indZ>0:
            neib += blockp[(indZ-1)+indY*gridsize[2]+indX*gridsize[1]*gridsize[2]]
        for j in neib:
            d = np.linalg.norm(pred[j,:]-gt[i,:])
            if d<cmin:
                cmin = d
        second[i] = cmin

    '''
    for i in range(nA):
        cmin = np.inf
        for j in range(nB):
            d = np.linalg.norm(pred[i,:]-gt[j,:])
            if d<cmin:
                cmin = d
        first[i] = cmin

    for j in range(nB):
        cmin = np.inf
        for i in range(nA):
            d = np.linalg.norm(pred[i,:]-gt[j,:])
            if d<cmin:
                cmin = d
        second[j]=cmin
    '''

    return first,second


def shapenet2Image(pre_dir, gt_dir, img_save_dir,color_error=False):
    samples = glob(pre_dir+'/*.'+'xyz')
    start = len(pre_dir)+1
    theta_Y = -math.pi*90/180
    Rot_Y = np.zeros((3,3))
    Rot_Y[1,1] = 1
    Rot_Y[0,0] = math.cos(theta_Y)
    Rot_Y[0,2] = -math.sin(theta_Y)
    Rot_Y[2,0] = math.sin(theta_Y)
    Rot_Y[2,2] = math.cos(theta_Y)

    theta_Z = -math.pi*90/180
    Rot_Z = np.zeros((3,3))
    Rot_Z[2,2] = 1
    Rot_Z[0,0] = math.cos(theta_Z)
    Rot_Z[0,1] = math.sin(theta_Z)
    Rot_Z[1,0] = -math.sin(theta_Z)
    Rot_Z[1,1] = math.cos(theta_Z)

    theta_X = -math.pi*90/180
    Rot_X = np.zeros((3,3))
    Rot_X[0,0] = 1
    Rot_X[1,1] = math.cos(theta_X)
    Rot_X[1,2] = math.sin(theta_X)
    Rot_X[2,1] = -math.sin(theta_X)
    Rot_X[2,2] = math.cos(theta_X)

    theta_Z1 = -math.pi*10/180
    Rot_Z1 = np.zeros((3,3))
    Rot_Z1[2,2] = 1
    Rot_Z1[0,0] = math.cos(theta_Z1)
    Rot_Z1[0,1] = math.sin(theta_Z1)
    Rot_Z1[1,0] = -math.sin(theta_Z1)
    Rot_Z1[1,1] = math.cos(theta_Z1)

    #for point_path in samples:
    for i in range(len(samples)):
        point_path = samples[i]
        gtname = gt_dir + point_path[start:]
        pred = np.loadtxt(point_path).astype(np.float32)
        gt = np.loadtxt(gtname).astype(np.float32)

        pcd = o3d.geometry.PointCloud()
        #mid_pc = np.matmul(np.matmul(gt, Rot_Y), Rot_Z)  
        mid_pc = np.matmul(np.matmul(gt, Rot_Y), Rot_Z)  
        pcd.points = o3d.utility.Vector3dVector(np.matmul(np.matmul(mid_pc,Rot_Z1), -Rot_Y))
        if color_error:
            first,second = points2setDis2(pred, gt)
            gt2p_min=np.min(second)
            gt2p_max=np.max(second)
            gt2p_range=gt2p_max-gt2p_min

            hd_dis = max(np.max(first),gt2p_max)
            cd_dis = np.mean(first)+np.mean(second)
            print(point_path[start:-4],':  (hd,cd)=',hd_dis,cd_dis)
            colorIndex = (second-gt2p_min)/gt2p_range
            gt_nums = gt.shape[0]
            rgb = np.zeros((gt_nums,3))
            for i  in range(gt_nums):
                if colorIndex[i]<0.5:
                    rgb[i,2]=2*(0.5-colorIndex[i])
                    rgb[i,1]=2*colorIndex[i]
                else:
                    rgb[i,0]=2*(colorIndex[i]-0.5)
                    rgb[i,1]=1-rgb[i,0]
            pcd.colors = o3d.utility.Vector3dVector(rgb)
        #else:
            #pcd.paint_uniform_color([0, 0, 0.0])
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible = True)#visible = False 
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        #vis.capture_screen_image(img_save_dir+point_path[start:-3]+'png')
        vis.run()
        vis.capture_screen_image(img_save_dir+point_path[start:-3]+'png')
        img = vis.capture_screen_float_buffer(True)
        plt.imshow(np.asarray(img))
        plt.axis('off')
        #plt.show()
        plt.savefig(img_save_dir+point_path[start:-3]+'eps') 
        vis.destroy_window()



@jit(parallel=True,nogil=True)
def points2setDis2(pred, gt):
    nA = pred.shape[0]
    nB = gt.shape[0]
    dist=np.ones((nA,nB))
    for i in range(nA):
        for j in range(nB):
            dist[i,j] = np.linalg.norm(pred[i,:]-gt[j,:])
    first = np.min(dist,axis=1)
    second = np.min(dist,axis=0)
    return first,second


def sparseAndGt2Image(sparse_dir, gt_dir,sparse_save_dir,gt_save_dir):
    samples = glob(sparse_dir+'/*.'+'xyz')
    start = len(sparse_dir)+1
    color = [210/255.0, 180/255.0, 140/255.0]#
    for point_path in samples:
        sparse = np.loadtxt(point_path).astype(np.float32)
        gtname = gt_dir + point_path[start:]
        gt = np.loadtxt(gtname).astype(np.float32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gt)
        pcd.paint_uniform_color(color)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible = True)#visible = False  
        vis.get_render_option().point_size = 5
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        vis.capture_screen_image(gt_save_dir+point_path[start:-3]+'png')
        img = vis.capture_screen_float_buffer(True)
        plt.imshow(np.asarray(img))
        plt.axis('off')
        plt.savefig(gt_save_dir+point_path[start:-3]+'eps')

        pcd.points = o3d.utility.Vector3dVector(sparse)
        vis.update_geometry(pcd)
        pcd.paint_uniform_color(color)
        vis.get_render_option().point_size = 5
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        vis.capture_screen_image(sparse_save_dir+point_path[start:-3]+'png')
        img = vis.capture_screen_float_buffer(True)
        plt.imshow(np.asarray(img))
        plt.axis('off')
        plt.savefig(sparse_save_dir+point_path[start:-3]+'eps')
        vis.destroy_window()
        del ctr
        del vis

def gt_sparse_dense2Image(gt_dir,other_dir,root_save_dir):
    samples = glob(gt_dir+'/*.'+'xyz')
    start = len(gt_dir)
    color = [210/255.0, 180/255.0, 140/255.0]#褐棕色
    gt_save_dir = os.path.join(root_save_dir,gt_dir.split('/')[-1])
    if not os.path.exists(gt_save_dir):
        os.makedirs(gt_save_dir)
    save_dirs = []
    error_dirs = []
    for path in other_dir:
        root_path = root_save_dir+path.split('/')[-1]
        save_path = root_path+'/output'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_dirs.append(save_path)
        save_path = root_path+'/errorMap'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        error_dirs.append(save_path)

    dir_nums = len(other_dir)
    cmap = plt.cm.get_cmap('jet')
    colors = cmap(np.arange(cmap.N))[:,0:3]
    for k in range(0,len(samples)):
        point_path = samples[k]
        gt = np.loadtxt(point_path).astype(np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gt)
        pcd.paint_uniform_color(color)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible = True)#visible = False  
        vis.get_render_option().point_size = 5
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        vis.capture_screen_image(gt_save_dir+point_path[start:-3]+'png')
        img = vis.capture_screen_float_buffer(True)
        plt.imshow(np.asarray(img))
        plt.axis('off')
        plt.savefig(gt_save_dir+point_path[start:-3]+'eps')

        for i in range(dir_nums):
            name = other_dir[i] + point_path[start:]
            points = np.loadtxt(name).astype(np.float32)
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color(color)
            vis.update_geometry(pcd)
            vis.get_render_option().point_size = 5
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)
            vis.poll_events()
            vis.update_renderer()
            vis.run()
            vis.capture_screen_image(save_dirs[i]+point_path[start:-3]+'png')
            img = vis.capture_screen_float_buffer(True)
            plt.imshow(np.asarray(img))
            plt.axis('off')
            plt.savefig(save_dirs[i]+point_path[start:-3]+'eps')
        
        pcd.points = o3d.utility.Vector3dVector(gt)
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        for n in range(1,dir_nums):
            name = other_dir[n] + point_path[start:]
            points = np.loadtxt(name).astype(np.float32)

            first,second = points2setDis2(points, gt)
            gt2p_min=np.min(second)
            gt2p_max=np.max(second)
            gt2p_range=gt2p_max-gt2p_min

            hd_dis = max(np.max(first),gt2p_max)
            cd_dis = np.mean(first)+np.mean(second)
            print(other_dir[n]+point_path[start:-4],':  (hd,cd)=',hd_dis,cd_dis)
            #colorIndex = (second-gt2p_min)/gt2p_range
            colorIndex = 255*(second-gt2p_min)/gt2p_range
            gt_nums = gt.shape[0]
            rgb = np.zeros((gt_nums,3))
            for i  in range(gt_nums):
                '''if colorIndex[i]<0.5:
                    rgb[i,2]=2*(0.5-colorIndex[i])
                    rgb[i,1]=2*colorIndex[i]
                else:
                    rgb[i,0]=2*(colorIndex[i]-0.5)
                    rgb[i,1]=1-rgb[i,0]'''
                fIdx = math.floor(colorIndex[i])
                cIdx = math.ceil(colorIndex[i])
                rgb[i,:]=(1.0+fIdx-colorIndex[i])*colors[fIdx,:]+(colorIndex[i]-fIdx)*colors[cIdx,:]
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            vis.get_render_option().point_size = 5
            vis.update_geometry(pcd)
            vis.update_renderer()
            vis.run()
            vis.capture_screen_image(error_dirs[n]+point_path[start:-3]+'png')
            img = vis.capture_screen_float_buffer(True)
            plt.imshow(np.asarray(img))
            plt.axis('off')
            plt.savefig(error_dirs[n]+point_path[start:-3]+'eps')
        vis.destroy_window()
        del ctr
        del vis

def gt2sparseError(sparse_dir, gt_dir):
    samples = glob(sparse_dir+'/*.'+'xyz')
    start = len(sparse_dir)
    for point_path in samples:
        gtname = gt_dir + point_path[start:]
        points = np.loadtxt(point_path).astype(np.float32)
        gt = np.loadtxt(gtname).astype(np.float32)
        first,second = points2setDis2(points, gt)
        gt2p_min=np.min(second)
        gt2p_max=np.max(second)
        gt2p_range=gt2p_max-gt2p_min

        hd_dis = max(np.max(first),gt2p_max)
        cd_dis = np.mean(first)+np.mean(second)
        print(point_path[start:-4],':  (hd,cd)=',hd_dis,cd_dis)
        colorIndex = (second-gt2p_min)/gt2p_range
        '''gt_nums = gt.shape[0]
        rgb = np.zeros((gt_nums,3))
        for i  in range(gt_nums):
            if colorIndex[i]<0.5:
                rgb[i,2]=2*(0.5-colorIndex[i])
                rgb[i,1]=2*colorIndex[i]
            else:
                rgb[i,0]=2*(colorIndex[i]-0.5)
                rgb[i,1]=1-rgb[i,0]'''
        
        #viewer(gt, colorIndex)
        
if __name__ == "__main__":
    start =time.time()
    
    path1 = 'D:/data/PC_Experiment_Results/standard/random'
    #path1 = 'D:/data/PC_Experiment_Results/standard/pugan'
    path2 = 'D:/data/PC_Experiment_Results/standard/poisson'
    
    #gt2sparseError(path1, path2)
    #img_save_dir = 'D:/data/PC_Experiment_Results/standard/shotImage/pugan/'
    #shapenet2Image(path1, path2, img_save_dir,True)

    #gt_save_dir = 'D:/data/PC_Experiment_Results/standard/shotImage/gt/'
    #sparse_save_dir = 'D:/data/PC_Experiment_Results/standard/shotImage/sparse/'
    #sparseAndGt2Image(path1, path2,sparse_save_dir,gt_save_dir)

    other_dir = ['D:/data/PC_Experiment_Results/standard/random']
    other_dir.append('D:/data/PC_Experiment_Results/standard/r02')
    other_dir.append('D:/data/PC_Experiment_Results/standard/pugan')
    root_save_dir = 'D:/data/PC_Experiment_Results/standard/image/'
    gt_sparse_dense2Image(path2,other_dir,root_save_dir)


    endT = time.time()

    print('Running time: %s Seconds'%(endT-start))

