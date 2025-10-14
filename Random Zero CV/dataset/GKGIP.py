import numpy as np
import pandas as pd
import math
import numpy.matlib
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import copy


SM_miRNA_M = np.loadtxt(r"new_matrix1029.txt", dtype=int)


#计算SM高斯轮廓核相似性
def Gaussian():
    row=155
    sum=0
    SM1=np.matlib.zeros((row,row))
    for i in range(0,row):
        a=np.linalg.norm(SM_miRNA_M[i,])*np.linalg.norm(SM_miRNA_M[i,])
        sum=sum+a
    ps=row/sum
    for i in range(0,row):
        for j in range(0,row):
            SM1[i,j]=math.exp(-ps*np.linalg.norm(SM_miRNA_M[i,]-SM_miRNA_M[j,])*np.linalg.norm(SM_miRNA_M[i,]-SM_miRNA_M[j,]))
    GSM = SM1
    return GSM


#计算mirna高斯轮廓核相似性
def Gaussian1():
    column=129
    sum=0
    miRNA1=np.matlib.zeros((column,column))
    for i in range(0,column):
        a=np.linalg.norm(SM_miRNA_M[:,i])*np.linalg.norm(SM_miRNA_M[:,i])
        sum=sum+a
    ps=column/sum
    for i in range(0,column):
        for j in range(0,column):
            miRNA1[i,j]=math.exp(-ps*np.linalg.norm(SM_miRNA_M[:,i]-SM_miRNA_M[:,j])*np.linalg.norm(SM_miRNA_M[:,i]-SM_miRNA_M[:,j]))


    GmiRNA = miRNA1
    return GmiRNA


def main():
    GKSS=Gaussian()
    GKSM=Gaussian1()
    # lnc=min_max_normalize(GKSS)
    # mi = min_max_normalize(GKSM)
    np.savetxt(r'GKGIP_snoRNA1029.txt', GKSS, delimiter='\t', fmt='%.9f')
    np.savetxt(r'GKGIP_drug1029.txt',  GKSM, delimiter='\t', fmt='%.9f')
    print(GKSS)
    print(GKSM)

if __name__ == "__main__":
        main()
