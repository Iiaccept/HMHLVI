from sklearn.metrics import matthews_corrcoef
import numpy as np
import copy
import math
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score
from sklearn.metrics import average_precision_score
from numpy.core import multiarray
from torch.nn.parameter import Parameter
import random
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from model import *
from kl_loss import kl_loss
from utils import f1_score_binary,precision_binary,recall_binary,accuracy_binary
import scipy.sparse as sp
from hypergraph_utils import *
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error
#以下两句用来忽略版本错误信息
import warnings
warnings.filterwarnings("ignore")
#设置device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 设置随机数种子
seed = 48
# random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
torch.backends.cudnn.deterministic = True  # 确保CUDA卷积操作的确定性
torch.backends.cudnn.benchmark = False  #禁用卷积算法选择，确保结果可重复

def laplacian_norm(adj):
    adj += np.eye(adj.shape[0])   # add self-loop
    degree = np.array(adj.sum(1))
    D = []
    for i in range(len(degree)):
        if degree[i] != 0:
            de = np.power(degree[i], -0.5)
            D.append(de)
        else:
            D.append(0)
    degree = np.diag(np.array(D))
    norm_A = degree.dot(adj).dot(degree)

    return norm_A

def cross_validation_5fold(k_folds):
    fold = int(totalassociation / k_folds) #1538
    auc = 0
    aupr = 0
    rec = 0
    pre = 0
    f1 = 0
    acc = 0
    mcc = 0
    tprs=[]
    fprs=[]
    aucs=[]
    precisions=[]
    recalls = []
    auprs = []
    loss_lists=[]
    accuracy_lists=[]
    mcc_lists = []
    # 五折交叉验证开始
    for f in range(1, k_folds + 1):
        print('%d fold:' %(f))
        if f == k_folds:
            testset = shuffle_data[((f - 1) * fold): totalassociation + 1]
        else:
            testset = shuffle_data[((f - 1) * fold): f * fold]    
      
        auc1, aupr1, recall1, precision1, f11, acc1, mcc1, loss_list, accuracy_list, mcc_list, all_fpr,all_tpr,all_auc, fpr, tpr=train(testset,epochs)
        tprs.append(tpr)
        fprs.append(fpr)

        precisions.append(precision1)
        recalls.append(recall1)
        aucs.append(auc1)
        auprs.append(aupr1)
        loss_lists.append(loss_list)
        accuracy_lists.append(accuracy_list)
        mcc_lists.append(mcc_list)
        auc = auc + auc1
        aupr = aupr + aupr1
        rec = rec + recall1
        pre = pre + precision1
        f1 = f1 + f11
        acc =acc + acc1
        mcc =mcc+ mcc1
    auc2 = auc/k_folds
    aupr2= aupr/k_folds
    pre2= pre / k_folds
    rec2= rec / k_folds
    f1_2= f1 / k_folds
    acc2 = acc / k_folds
    mcc2 = mcc / k_folds
    print("cv_mean:")
    print('auc: {:.4f}, aupr: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1_score: {:.4f}, acc: {:.4f}, mcc: {:.4f}'
          .format(auc2, aupr2, pre2, rec2, f1_2, acc2, mcc2))

    metric = ["{:.4f}".format(v) for v in [auc2, aupr2, pre2, rec2, f1_2, acc2, mcc2]]

    return metric, aucs, precisions, recalls, auprs, loss_lists, accuracy_lists, mcc_lists,all_fpr,all_tpr,all_auc, fpr, tpr


def train(testset,epochs):
    all_f = np.random.permutation(np.size(Index_zeroRow))
    test_p = list(testset)
    test_f = all_f[0:len(test_p)]
    difference_set_f = list(set(all_f).difference(set(test_f)))
    train_f = difference_set_f

    X = copy.deepcopy(MD)
    Xn = copy.deepcopy(X)
    zero_index = []
    for ii in range(len(train_f)):
        zero_index.append([Index_zeroRow[train_f[ii]], Index_zeroCol[train_f[ii]]])

    true_list = multiarray.zeros((len(test_p) + len(test_f), 1))
    for ii in range(len(test_p)):
        Xn[Index_PositiveRow[testset[ii]], Index_PositiveCol[testset[ii]]] = 0
        true_list[ii, 0] = 1
    train_mask = np.ones(shape=Xn.shape)
    for ii in range(len(test_p)):
        train_mask[Index_PositiveRow[testset[ii]], Index_PositiveCol[testset[ii]]] = 0
        train_mask[Index_zeroRow[test_f[ii]], Index_zeroCol[test_f[ii]]] = 0
    train_mask_tensor = torch.from_numpy(train_mask).to(torch.bool)
    train_mask_tensor = train_mask_tensor.to(device)
    label = true_list
    
    model = Gai_HGNN()
    optimizer2 = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    def duo_hg():
        #DG是药物高斯核相似性，MG是snoRNA高斯核相似性
        HHMG = construct_H_with_KNN(MG)
        HMG = generate_G_from_H(HHMG)
        HMG = HMG.double()

        HHDG = construct_H_with_KNN(DG)
        HDG = generate_G_from_H(HHDG)
        HDG = HDG.double()

        A = copy.deepcopy(Xn)

        AT = A.T

        HHMD = construct_H_with_KNN(A)
        HMD = generate_G_from_H(HHMD)
        HMD = HMD.double()

        HHDM = construct_H_with_KNN(A.T)
        HDM = generate_G_from_H(HHDM)
        HDM = HDM.double()

        #MM是snoRNA序列相似性，DD是药物相似性
        HHMM = construct_H_with_KNN(MM)
        HMM = generate_G_from_H(HHMM)
        HMM = HMM.double()

        HHDD = construct_H_with_KNN(DD)
        HDD = generate_G_from_H(HHDD)
        HDD = HDD.double()

        A = torch.from_numpy(A)

        AT = torch.from_numpy(AT)

        #用于VAE的两个块矩阵编码器

        VAE_1 = np.hstack((HMD,A ))
        VAE_1 = torch.tensor(VAE_1)
        VAE_1 = VAE_1.to(device)

        VAE_2 = np.hstack((HDM,A.T))
        VAE_2 = torch.tensor(VAE_2)
        VAE_2 = VAE_2.to(device)



        # 生物属性异构网络
        # 拼接snoRNA序列相似性和药物结构相似性
        lnc_shape = HMM.shape[0]  # 高阶snoRNA序列相似性
        dis_shape = HDD.shape[0]  # 高阶药物结构相似性

        lnc_dis_mi = np.hstack((HMM, np.zeros((lnc_shape, dis_shape))))
        dis_lnc_mi = np.hstack((np.zeros((dis_shape, lnc_shape)), HDD))
        matrix_A = np.vstack((lnc_dis_mi, dis_lnc_mi))
        matrix_A = laplacian_norm(matrix_A)
        matrix_A = torch.tensor(matrix_A)
        matrix_A = matrix_A.to(device)


        # 网络拓扑异构网络,HMG是snoRNA高斯核相似性，HDG是药物高斯核相似性
        lnc_dis_sim = np.hstack((HMG, np.zeros((lnc_shape, dis_shape))))
        dis_lnc_sim = np.hstack((np.zeros((dis_shape, lnc_shape)), HDG))
        matrix_B = np.vstack((lnc_dis_sim, dis_lnc_sim))
        matrix_B = laplacian_norm(matrix_B)
        matrix_B = torch.tensor(matrix_B)
        matrix_B = matrix_B.to(device)

        HMG = HMG.to(device)
        HDG = HDG.to(device)
        HMD = HMD.to(device)
        HDM = HDM.to(device)
        HMM = HMM.to(device)
        HDD = HDD.to(device)
        A = A.to(device)
        AT = AT.to(device)
        mir_feat = torch.eye(155)
        mir_feat = mir_feat.to(device)
        dis_feat = torch.eye(129)
        dis_feat = dis_feat.to(device)
        heterogeneous = torch.eye(284)
        heterogeneous = heterogeneous.to(device)


        mir_feat, dis_feat, heterogeneous = Variable(mir_feat), Variable(dis_feat), Variable(heterogeneous)

        return AT, A, HMG, HDG, mir_feat, dis_feat, HMD, HDM, HMM, HDD, matrix_A, matrix_B, heterogeneous, VAE_1, VAE_2



    AT,A,HMG,HDG,mir_feat,dis_feat,HMD,HDM,HMM,HDD,matrix_A, matrix_B, heterogeneous, VAE_1, VAE_2=duo_hg()


    pos_weight = float(A.shape[0] * A.shape[1] - A.sum()) / A.sum()

    loss_kl = kl_loss(129, 155)

    accuracy_list = []
    mcc_list = []
    loss_list=[]
    for epoch in tqdm(range(epochs), desc='epochs'):    
        model.train()

        # 总，输入两个异构网络+共享编码+两个VAE
        reconstruction1, result, reconstructionG, reconstructionMD, reconstructionMMDD, result_h, recover, mir_feature_1, mir_feature_2, mir_feature_5, dis_feature_1, dis_feature_2, dis_feature_5, shuxing_feature, jiegou_feature, hsum = model(
            matrix_A, matrix_B, heterogeneous, VAE_1, VAE_2)

        outputs = recover.t().cpu().detach().numpy()
        print(outputs)
        test_predict = create_resultlist(outputs, testset, Index_PositiveRow, Index_PositiveCol, Index_zeroRow,Index_zeroCol, len(test_p), zero_length, test_f)
        MA = torch.masked_select(A, train_mask_tensor)
        reG = torch.masked_select(reconstructionG.t(),train_mask_tensor)
        reMD = torch.masked_select(reconstructionMD.t(), train_mask_tensor)
        reMMDD = torch.masked_select(reconstructionMMDD.t(), train_mask_tensor)
        ret = torch.masked_select(result.t(), train_mask_tensor)
        re1 = torch.masked_select(reconstruction1.t(), train_mask_tensor)
        rec = torch.masked_select(recover.t(), train_mask_tensor)
        loss_k = loss_kl(model.z_node_log_std, model.z_node_mean, model.z_edge_log_std, model.z_edge_mean)

        loss_2 = loss_k + F.binary_cross_entropy_with_logits(re1.t(), MA,pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(ret.t(), MA,pos_weight=pos_weight)
        loss_1 =F.binary_cross_entropy_with_logits(reG.t(), MA,pos_weight=pos_weight) +F.binary_cross_entropy_with_logits(reMMDD.t(), MA,pos_weight=pos_weight)+ F.binary_cross_entropy_with_logits(reMD.t(), MA,pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(rec.t(), MA,pos_weight=pos_weight)

        loss = loss_1 +  loss_2


        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()
        auc_val = roc_auc_score(label, test_predict)
        # rmse = np.sqrt(mean_squared_error(label,test_predict))
        # rp = np.corrcoef(label, test_predict)[0, 1]
        # print("rmse",rmse)
        # print("rp",rp)
        # fpr, tpr, _ = roc_curve(label, test_predict)
        aupr_val = average_precision_score(label, test_predict)

        print('Epoch: {:04d},loss: {:.5f},auc_val: {:.5f},aupr_val: {:.5f}'
              .format(epoch+1,loss.data.item(),auc_val,aupr_val))
        loss_list.append(loss.data.item())
        max_f1_score, threshold = f1_score_binary(torch.from_numpy(label).float(),torch.from_numpy(test_predict).float())
        print("max_f1_score",max_f1_score)
        precision = precision_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("precision:", precision)
        recall = recall_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("recall:", recall)


        test_predict = np.array(test_predict)  # Ensure it's a NumPy array
        # Convert test_predict to a tensor to match the type with threshold
        test_predict_tensor = torch.from_numpy(test_predict)

        # Accuracy calculation using tensors
        binary_predictions = (test_predict_tensor >= threshold).float()
        accuracy = (binary_predictions == torch.from_numpy(label).float()).float().mean().item()
        # Append metrics to lists
        accuracy_list.append(accuracy)
        # MCC calculation
        mcc = matthews_corrcoef(torch.from_numpy(label).numpy(), binary_predictions.numpy())
        mcc_list.append(mcc)
        print("Accuracy:", accuracy)
        print("MCC:", mcc)

    fpr, tpr = [], []
    print("train end!")
    
    auc1 = auc_val
    aupr1 = aupr_val
    recall1 = recall
    precision1 = precision
    f11 = max_f1_score
    acc1 = accuracy
    mcc1 = mcc

    # 为了画图
    fpr, tpr, thresholds = roc_curve(label, test_predict)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_auc.append(roc_auc)

    return auc1,aupr1,recall1,precision1,f11,acc1, mcc1,loss_list,accuracy_list,mcc_list,all_fpr,all_tpr,all_auc, fpr, tpr

#主函数
if __name__== '__main__':
    #读数据
    MD = np.loadtxt("dataset/new_matrix.txt")
    MM = np.loadtxt("dataset/snoBIANJI.txt")
    DD = np.loadtxt("dataset/drug.txt")
    DG = np.loadtxt("dataset/GKGIP_drug.txt")
    MG = np.loadtxt("dataset/GKGIP_snoRNA.txt")
    [row, col] = np.shape(MD)  #获取MD数组的形状，即行数和列数
    #识别出MD数组中值为0和值为1的元素位置，代表不同的类别或状态（如正样本和负样本）。
    indexn = np.argwhere(MD == 0)  #找出MD数组中所有值为0的元素的位置，返回一个数组indexn，其中包含这些元素的行和列索引。
    Index_zeroRow = indexn[:, 0]  #从indexn中提取所有行索引
    Index_zeroCol = indexn[:, 1]  #提取所有列索引
    indexp = np.argwhere(MD == 1)  #找出MD数组中所有值为1的元素的位置，返回一个数组indexp，其中包含这些元素的行和列索引。
    Index_PositiveRow = indexp[:, 0]  #提取所有行索引
    Index_PositiveCol = indexp[:, 1]  #提取所有列索引
    zero_length = np.size(Index_zeroRow) #计算值为0的元素的数量（通过行索引的大小）。
    totalassociation = np.size(Index_PositiveRow) #计算值为1的元素的数量，即正样本的总数。
    shuffle_data = np.random.permutation(totalassociation)   #对totalassociation（值为1的元素数量）进行随机排列。打乱正样本的顺序
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    all_fpr, all_tpr, all_auc = [], [], []
    all_precision, all_recall, all_aupr = [], [], []
    k_folds=5
    #定义模型超参数
    lr= 0.002
    p= 0.3
    # k= 50
    weight_decay=0.02
    temperature=0.2
    epochs=100
    dropout=0.5
    result, aucs, precisions, recalls, auprs, loss_lists, accuracy_lists, mcc_lists,all_fpr,all_tpr,all_auc, fpr, tpr = cross_validation_5fold(k_folds)
