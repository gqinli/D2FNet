# --*-- coding:utf-8 --*--
import math
import cv2
import os
import math
import numpy as np
import scipy.io as scio

from utils.rgbd_util import *
from utils.getCameraParam import *

'''
must use 'COLOR_BGR2GRAY' here, or you will get a different gray-value with what MATLAB gets.
'''
def getImage(root='demo'):
    D = cv2.imread(root, cv2.COLOR_BGR2GRAY)/100
    # D = scio.loadmat('../data/NLPR/smoothedDepth/1_02-02-40_Depth.mat')['smoothedDepth']/1000
    # ../data/NLPR/smoothedDepth/1_02-02-40_Depth.mat scio.loadmat('../data/NLPR/rawDepth/1_02-02-40_Depth.mat')['rawDepth']
    RD = D
    return D, RD


'''
C: Camera matrix
D: Depth image, the unit of each element in it is "meter"
RD: Raw depth image, the unit of each element in it is "meter"
'''
def getHHA(C, D, RD):
    missingMask = (RD == 0);
    pc, N, yDir, h, pcRot, NRot = processDepthImage(D * 100, missingMask, C);

    tmp = np.multiply(N, yDir)
    acosValue = np.minimum(1,np.maximum(-1,np.sum(tmp, axis=2)))
    angle = np.array([math.degrees(math.acos(x)) for x in acosValue.flatten()])
    angle = np.reshape(angle, h.shape)

    '''
    Must convert nan to 180 as the MATLAB program actually does. 
    Or we will get a HHA image whose border region is different
    with that of MATLAB program's output.
    '''
    angle[np.isnan(angle)] = 180        


    pc[:,:,2] = np.maximum(pc[:,:,2], 100)
    I = np.zeros(pc.shape)

    # opencv-python save the picture in BGR order.
    I[:,:,2] = 31000/pc[:,:,2]
    I[:,:,1] = h
    I[:,:,0] = (angle + 128-90)

    # print(np.isnan(angle))

    '''
    np.uint8 seems to use 'floor', but in matlab, it seems to use 'round'.
    So I convert it to integer myself.
    '''
    I = np.rint(I)

    # np.uint8: 256->1, but in MATLAB, uint8: 256->255
    I[I>255] = 255
    HHA = I.astype(np.uint8)
    return HHA

if __name__ == "__main__":
    depth_Data = os.listdir('../data/NJU2000/depth/')
    for i in depth_Data:
        root =  os.path.join('../data/NJU2000/depth', i)
        D, RD = getImage(root)
        camera_matrix = getCameraParam('color')
        print('max gray value: ', np.max(D))        # make sure that the image is in 'meter'
        # hha = getHHA(camera_matrix, D, RD)
        hha_complete = getHHA(camera_matrix, D, D)
        # cv2.imwrite('demo/raw_smooth.png', hha)
        cv2.imwrite(os.path.join('../data/NJU2000/hha', i), hha_complete)
    
    
    ''' multi-peocessing example '''
    '''
    from multiprocessing import Pool
    
    def generate_hha(i):
        # generate hha for the i-th image
        return
    
    processNum = 16
    pool = Pool(processNum)

    for i in range(img_num):
        print(i)
        pool.apply_async(generate_hha, args=(i,))
        pool.close()
        pool.join()
    ''' 
