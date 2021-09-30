#-*- coding:utf-8 -*-
""" line promgram.

Author: Zezeng Li
Date: March 2021
"""

from pulp import LpProblem,LpMaximize,LpVariable,LpContinuous,lpSum
import numpy as np
import pc_util
import math
#from Common import pc_util
from pc_util import *
from scipy.optimize import linprog

def H_Star_Solution(fake_point, true_point, coef_K):
    fakeNum=fake_point.shape[0]
    trueNum=true_point.shape[0]
    transportCost=np.zeros((fakeNum,trueNum))
    
    for i in range(trueNum):
        for j in range(fakeNum):
            diff=fake_point[i]-true_point[j]
            square=diff*diff
            transportCost[i][j]=2*coef_K*np.sum(square)
    
    Ingredients1=['Y_'+str(i) for i in range(trueNum)]
    Ingredients2=['X_'+str(i) for i in range(fakeNum)]
    Ingredients=Ingredients1+Ingredients2

    costCoef = [1/trueNum for i in range(trueNum)]+[-1/fakeNum for j in range(fakeNum)]

    costs=dict(zip(Ingredients,costCoef))

    prob = LpProblem("H_Star_Solution", LpMaximize)

    ingredient_vars = LpVariable.dicts("Ingr",Ingredients,0,1,LpContinuous)

    prob += lpSum([costs[i]*ingredient_vars[i] for i in Ingredients])

    for i in range(trueNum):
        trueIndex='Y_'+str(i) 
        for j in range(fakeNum):
            fakeIndex='X_'+str(j)
            prob += lpSum([ingredient_vars[trueIndex],-ingredient_vars[fakeIndex]]) <= transportCost[i][j]
    
    try:
        prob.solve(PULP_CBC_CMD(msg =False))

        HStar_real = [0]*trueNum
        Ord=0
        for i in Ingredients1:
            HStar_real[Ord] = ingredient_vars[i].value()
            Ord +=1
    
        HStar_fake = [0]*fakeNum
        Ord=0
        for i in Ingredients2:
            HStar_fake[Ord] = ingredient_vars[i].value()
            Ord +=1
    
        HStar_real = np.array(HStar_real)
        HStar_fake = np.array(HStar_fake)
    except:
        HStar_real = np.ones(trueNum, dtype=float)
        HStar_fake = np.zeros(trueNum, dtype=float)
    return HStar_fake, HStar_real

def H_Star_Solution_scipy(fakeImg, trueImg, coef_K):
    fakeNum=fakeImg.shape[0]
    trueNum=trueImg.shape[0]
    # 1.Set the variable coefficient of the objective function
    Coef = [-1/trueNum for i in range(trueNum)]+[1/fakeNum for j in range(fakeNum)]
    Coef=np.array(Coef)

    A_Ineq=np.zeros((fakeNum*trueNum,fakeNum+trueNum))
    B_Ineq=np.zeros(fakeNum*trueNum)
    for i in range(trueNum):
        for j in range(fakeNum):
            ind = i*fakeNum+j
            A_Ineq[ind,i]=1
            A_Ineq[ind,i+j+1]=-1
            diff=(fakeImg[i]-trueImg[j])
            square=diff*diff
            B_Ineq[ind]=0.5*coef_K*np.sum(square)
    bounds=[(0,1) for i in range(trueNum+fakeNum)]

    res=linprog(Coef,A_Ineq,B_Ineq,bounds=bounds)
    #print(res)
    HStar_real = res.x[0:trueNum]
    HStar_fake = res.x[trueNum:]
    '''
    if res.success==False:
        HStar_real = np.ones(trueNum, dtype=float)
        HStar_fake = np.zeros(trueNum, dtype=float)'''
    return HStar_real, HStar_fake

def main(true_point_path,fake_point_path,point_nums,patch_nums):

    #1.读取真实数据
    real_data = pc_util.load(true_point_path)
    print("real_data load done!")

    sum_points = point_nums * patch_nums
    real_nums = real_data.shape[0]
    real_patchs = math.floor(min(sum_points, real_nums) / point_nums)
    real_points = np.zeros((real_patchs,point_nums,3),dtype=float)
    for i in range(real_patchs):
        start_index=i*point_nums
        real_points[i]=real_data[start_index:start_index+point_nums,:]


    # 2.读取虚假数据
    fake_data = pc_util.load(fake_point_path)
    print("fake_data load done!")
    
    fake_nums = fake_data.shape[0]
    fake_patchs = math.floor(min(sum_points, fake_nums) / point_nums)
    fake_points = np.zeros((fake_patchs, point_nums, 3), dtype=float)
    for i in range(real_patchs):
        start_index=i*point_nums
        fake_points[i]=fake_data[start_index:start_index+point_nums,:]


    #assert len(fake_points) == len(fake_points)
    
    #3.求解线性优化问�?
    coef_K=1
    HStar_real, HStar_fake = H_Star_Solution(fake_points, real_points, coef_K)

    print("HStar_real=",HStar_real)
    print("HStar_fake=",HStar_fake)
    print("HStar_fake.type=",type(HStar_fake))
    print("HStar_fake.mean=",HStar_fake.mean())




if __name__ == '__main__':
    
    true_point_path='/root/data/code/data/test/cow.off'
    fake_point_path='/root/data/code/data/test/duck.off'
    point_nums = 500
    patch_nums = 10
    main(true_point_path,fake_point_path,point_nums,patch_nums)

