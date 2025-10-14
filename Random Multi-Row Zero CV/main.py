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
import DataEnhancement
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from model3 import *
from kl_loss import kl_loss
from utils import f1_score_binary,precision_binary,recall_binary,accuracy_binary
import scipy.sparse as sp
import plot_auc_curves
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
        # auc1, aupr1, recall1, precision1, f11, acc1, mcc1, loss_list, accuracy_list, mcc_list, all_recall, all_precision, all_aupr, recall, precision = train(
        #     testset, epochs)
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
    # metric.append(auc2)
    # metric.append(aupr2)
    # metric.append(pre2)
    # metric.append(rec2)
    # metric.append(f1_2)
    # metric.append(acc2)
    # metric.append(mcc2)


    return metric, aucs, precisions, recalls, auprs, loss_lists, accuracy_lists, mcc_lists,all_fpr,all_tpr,all_auc, fpr, tpr
    # return metric, aucs, precisions, recalls, auprs, loss_lists, accuracy_lists, mcc_lists, all_recall, all_precision, all_aupr, recall, precision
def train(testset,epochs):
    X = copy.deepcopy(MD)

    Xn = copy.deepcopy(X)
    test_length = len(testset)
    print(test_length)
    for ii in range(test_length):
        Xn[prolist[testset[ii]], :] = 0


    train_mask = np.ones(shape=Xn.shape)
    for ii in range(test_length):
        train_mask[prolist[testset[ii]], :] = 0
    train_mask_tensor = torch.from_numpy(train_mask).to(torch.bool)
    train_mask_tensor = train_mask_tensor.to(device)
    
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
        # VAE_1 = A
        VAE_1 = np.hstack((HMD,A ))
        VAE_1 = torch.tensor(VAE_1)
        VAE_1 = VAE_1.to(device)
        # VAE_2 = A.T
        VAE_2 = np.hstack((HDM,A.T))
        VAE_2 = torch.tensor(VAE_2)
        VAE_2 = VAE_2.to(device)
        # print(VAE_2.shape[0])
        # print(VAE_2.shape[1])


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
        #在生物属性异构网络里面加入关联矩阵
        # lnc_dis_mi = np.hstack((HMM, A))
        # dis_lnc_mi = np.hstack((A.T, HDD))
        # matrix_A = np.vstack((lnc_dis_mi, dis_lnc_mi))
        # matrix_A = torch.tensor(matrix_A)
        # matrix_A = matrix_A.to(device)

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

        # Generating feature matrix
        # np.random.seed(48)
        # features = np.random.normal(loc=0, scale=1, size=(matrix_A.shape[0], 177))
        # node_feature = row_normalize(features)
        # node_feature = torch.tensor(node_feature)
        # node_feature = node_feature.to(device)
        # # Adversarial nodes
        # np.random.seed(48)
        # id = np.arange(node_feature.shape[0])
        # id = np.random.permutation(id)
        # shuf_feature = node_feature[id]
        # shuf_feature = torch.tensor(shuf_feature)
        # shuf_feature = shuf_feature.to(device)
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
        #最初的
        # reconstruction1,result,reconstructionG,reconstructionMD,reconstructionMMDD,result_h,recover,mir_feature_1,mir_feature_2,mir_feature_3,dis_feature_1,dis_feature_2 ,dis_feature_3= model(AT,A,HMG,HDG,mir_feat,dis_feat,HMD,HDM,HMM,HDD)

        #修改，输入两个异构网络=去除共享
        # reconstruction1, result, reconstructionG, reconstructionMD,  result_h, recover, mir_feature_1, mir_feature_2,  dis_feature_1, dis_feature_2, shuxing_feature, jiegou_feature = model(
        #         matrix_A, matrix_B, heterogeneous, VAE_1, VAE_2)
        # 修改，输入两个异构网络+共享编码

        # reconstruction1,result,reconstructionG,reconstructionMD,reconstructionMMDD,result_h,recover,mir_feature_1,mir_feature_2,mir_feature_5,dis_feature_1,dis_feature_2 ,dis_feature_5, shuxing_feature, jiegou_feature, hsum= model(
        #     AT, A, matrix_A, matrix_B, heterogeneous)

        #消融去除VAE
        # reconstructionG, reconstructionMD, reconstructionMMDD, result_h, recover, mir_feature_1, mir_feature_2, mir_feature_5, dis_feature_1, dis_feature_2, dis_feature_5, shuxing_feature, jiegou_feature, hsum = model(
        #     matrix_A, matrix_B, heterogeneous, VAE_1, VAE_2)
        #消融-只保留属性
        # reconstruction1, result, reconstructionG, result_h, recover, mir_feature_1,  dis_feature_1,  shuxing_feature = model(
        #     matrix_A, matrix_B, heterogeneous, VAE_1, VAE_2)
        #消融-只保留结构
        # reconstruction1, result, reconstructionMD, result_h, recover, mir_feature_2,  dis_feature_2,  jiegou_feature = model(
        #     matrix_A, matrix_B, heterogeneous, VAE_1, VAE_2)

        # 总，输入两个异构网络+共享编码+两个VAE
        reconstruction1, result, reconstructionG, reconstructionMD, reconstructionMMDD, result_h, recover, mir_feature_1, mir_feature_2, mir_feature_5, dis_feature_1, dis_feature_2, dis_feature_5, shuxing_feature, jiegou_feature, hsum = model(
            matrix_A, matrix_B, heterogeneous, VAE_1, VAE_2)

        outputs = recover.t().cpu().detach().numpy()
        test_predict = create_resultmatrix_row(outputs, testset, prolist)
        test_predict = test_predict.reshape(-1, 1)

        # label
        label = create_resultmatrix_row(MD, testset, prolist)
        label = label.reshape(-1, 1)


        MA = torch.masked_select(A, train_mask_tensor)
        reG = torch.masked_select(reconstructionG.t(),train_mask_tensor)
        reMD = torch.masked_select(reconstructionMD.t(), train_mask_tensor)
        reMMDD = torch.masked_select(reconstructionMMDD.t(), train_mask_tensor)
        ret = torch.masked_select(result.t(), train_mask_tensor)
        re1 = torch.masked_select(reconstruction1.t(), train_mask_tensor)
        rec = torch.masked_select(recover.t(), train_mask_tensor)
        loss_k = loss_kl(model.z_node_log_std, model.z_node_mean, model.z_edge_log_std, model.z_edge_mean)

        #原对比损失
        # loss_c_m =  compute_contrastive_loss(mir_feature_1,mir_feature_5,temperature) + compute_contrastive_loss(mir_feature_2,mir_feature_5,temperature)
        # loss_c_d = compute_contrastive_loss(dis_feature_1, dis_feature_5,temperature) + compute_contrastive_loss(dis_feature_2, dis_feature_5,temperature)
        # loss_3 = loss_c_d + loss_c_m

        #两个异构网络+共享对比损失
        # loss_c_m =  compute_contrastive_loss(mir_feature_1,mir_feature_5,temperature) + compute_contrastive_loss(mir_feature_1,mir_feature_5,temperature)
        # loss_c_d = compute_contrastive_loss(dis_feature_1, dis_feature_5,temperature) + compute_contrastive_loss(dis_feature_1, dis_feature_5,temperature)
        # loss_3 = loss_c_m

        # 三个异构网络的对比损失
        # loss_c_m =  compute_contrastive_loss(shuxing_feature, hsum,temperature) + compute_contrastive_loss(jiegou_feature, hsum, temperature)
        # # loss_c_d = compute_contrastive_loss(dis_feature_1, dis_feature_5,temperature) + compute_contrastive_loss(dis_feature_1, dis_feature_5,temperature)
        # loss_3 = loss_c_m

        loss_2 = loss_k + F.binary_cross_entropy_with_logits(re1.t(), MA,pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(ret.t(), MA,pos_weight=pos_weight)
        loss_1 =F.binary_cross_entropy_with_logits(reG.t(), MA,pos_weight=pos_weight) +F.binary_cross_entropy_with_logits(reMMDD.t(), MA,pos_weight=pos_weight)+ F.binary_cross_entropy_with_logits(reMD.t(), MA,pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(rec.t(), MA,pos_weight=pos_weight)

        # loss = loss_1 + p*loss_2 + (1-p)*loss_3

        #只有两个特异编码器
        # loss_1 = F.binary_cross_entropy_with_logits(reMD.t(), MA,pos_weight=pos_weight)  + F.binary_cross_entropy_with_logits(rec.t(), MA,pos_weight=pos_weight)
        # 注意力机制融合
        # loss_1 = F.binary_cross_entropy_with_logits(reG.t(), MA,pos_weight=pos_weight)  +F.binary_cross_entropy_with_logits(rec.t(), MA,pos_weight=pos_weight)

        loss = loss_1 +  loss_2

        # loss = loss_1

        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()
        auc_val = roc_auc_score(label, test_predict)
        rmse = np.sqrt(mean_squared_error(label,test_predict))
        rp = np.corrcoef(label, test_predict)[0, 1]
        print("rmse",rmse)
        print("rp",rp)
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
    # # 为了画图
    # precision, recall, thresholds = precision_recall_curve(label, test_predict)
    #
    # # Interpolate the recall-precision values for consistent comparison across folds
    # interp_recall = np.linspace(0, 1, 100)  # Uniformly spaced recall values
    # interp_precision = np.interp(interp_recall, recall[::-1],
    #                              precision[::-1])  # Reverse recall and precision for interpolation
    # interp_precision[0] = 1.0  # Ensure the starting precision is 1.0 for recall = 0
    #
    # # Store interpolated values
    # all_precision.append(precision)
    # all_recall.append(recall)
    # #
    # # # Calculate AUPR for this fold
    # aupr = auc(recall, precision)
    # all_aupr.append(aupr)
    # # for i in range(len(recall)):
    # #     if recall[i] == 1:
    # #         precision[i] = 0
    # # aupr = auc(recall, precision)
    #
    # return auc1, aupr1, recall1, precision1, f11, acc1, mcc1, loss_list, accuracy_list, mcc_list, all_recall, all_precision, all_aupr, recall, precision

#主函数
if __name__== '__main__':
    #读数据
    #关联矩阵
    # MD = np.loadtxt("data/md_delete.txt")
    #
    # #miRNA相似性矩阵
    # MM = np.loadtxt("data/mm_delete.txt")
    # # MM= read_csv('E:/li/zr_RNA_dis_pre/datasets/m_ss.csv')
    # #疾病相似性矩阵
    # DD = np.loadtxt("data/dd_delete.txt")
    #
    # #疾病基因矩阵
    # DG = np.loadtxt("data/dg_delete.txt")
    #
    # #基因疾病矩阵
    # MG = np.loadtxt("data/mg_delete.txt")

    # 关联矩阵
    # MD = np.loadtxt("dataset4(923,104)/association.txt")
    #
    # # miRNA相似性矩阵
    # MM = np.loadtxt("dataset4(923,104)/GKGIP_circRNA.txt")
    # # MM= read_csv('E:/li/zr_RNA_dis_pre/datasets/m_ss.csv')
    # # 疾病相似性矩阵
    # DD = np.loadtxt("dataset4(923,104)/GKGIP_disease.txt")
    #
    # # 疾病基因矩阵
    # DG = np.loadtxt("dataset4(923,104)/LKGIP_disease.txt")
    #
    # # 基因疾病矩阵
    # MG = np.loadtxt("dataset4(923,104)/LKGIP_circRNA.txt")
#*****************************************************************************************************
    # 关联矩阵
    MD = np.loadtxt("snoRNA(155,129)/new_matrix.txt")
    # miRNA相似性矩阵
    MM = np.loadtxt("snoRNA(155,129)/snoBIANJI.txt")
    # MM= read_csv('E:/li/zr_RNA_dis_pre/datasets/m_ss.csv')
    # 疾病相似性矩阵
    DD = np.loadtxt("snoRNA(155,129)/drug.txt")
    # 疾病基因矩阵
    DG = np.loadtxt("snoRNA(155,129)/GKGIP_drug.txt")
    # 基因疾病矩阵
    MG = np.loadtxt("snoRNA(155,129)/GKGIP_snoRNA.txt")


    [row, col] = np.shape(MD)  # 获取MD数组的形状，即行数和列数
    prolist = np.array(list(range(row)))
    # 识别出MD数组中值为0和值为1的元素位置，代表不同的类别或状态（如正样本和负样本）。
    indexn = np.argwhere(MD == 0)  # 找出MD数组中所有值为0的元素的位置，返回一个数组indexn，其中包含这些元素的行和列索引。
    Index_zeroRow = indexn[:, 0]  # 从indexn中提取所有行索引
    Index_zeroCol = indexn[:, 1]  # 提取所有列索引
    indexp = np.argwhere(MD == 1)  # 找出MD数组中所有值为1的元素的位置，返回一个数组indexp，其中包含这些元素的行和列索引。
    Index_PositiveRow = indexp[:, 0]  # 提取所有行索引
    Index_PositiveCol = indexp[:, 1]  # 提取所有列索引
    zero_length = np.size(Index_zeroRow)  # 计算值为0的元素的数量（通过行索引的大小）。
    totalassociation = np.size(prolist)  # 计算值为1的元素的数量，即正样本的总数。
    shuffle_data = np.random.permutation(totalassociation)  # 对totalassociation（值为1的元素数量）进行随机排列。打乱正样本的顺序

    # Set the seed at the beginning of your main script
    # MD = DataEnhancement.SVD(MD)
    # MM = DataEnhancement.SVD(MM)
    # DD = DataEnhancement.SVD(DD)
    # DG = DataEnhancement.SVD(DG)
    # MG = DataEnhancement.SVD(MG)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    all_fpr, all_tpr, all_auc = [], [], []
    all_precision, all_recall, all_aupr = [], [], []
    k_folds=10
    #定义模型超参数
    lr= 0.002
    p= 0.3
    # k= 50
    weight_decay=0.02
    temperature=0.2
    epochs=100
    dropout=0.5
    result, aucs, precisions, recalls, auprs, loss_lists, accuracy_lists, mcc_lists,all_fpr,all_tpr,all_auc, fpr, tpr = cross_validation_5fold(k_folds)
    # result, aucs, precisions, recalls, auprs, loss_lists, accuracy_lists, mcc_lists, all_recall, all_precision, all_aupr, recall, precision = cross_validation_5fold(
    #     k_folds)
    #保存结果
    # import os
    # result_dir = "./results"
    # if not os.path.exists(result_dir):
    #         os.makedirs(result_dir)
    #
    # from datetime import datetime
    # # 获取当前日期和时间
    # current_datetime = datetime.now()
    # # 将当前日期转为字符串显示，格式为 YYYY-MM-DD
    # current_date_str = current_datetime.strftime('%Y_%m_%d')
    #
    # #保存结果到excel文件
    # name="CLHGNNMDA"
    # import pandas as pd
    # print(result)
    #
    # df = pd.DataFrame([result], columns=['auc', 'aupr', 'precision', 'recall', 'f1_score', 'accuracy', 'mcc'])
    # filename1=result_dir+f'/{name}_优化后性能指标记录_{current_date_str}.xlsx'
    # df.to_excel(filename1, index=False)


    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))

    for i in range(len(all_fpr)):
        plt.plot(all_fpr[i], all_tpr[i], label=f'ROC fold {i + 1} (AUC = {all_auc[i]:.4f})', linestyle='-',
                 linewidth=2)
    # 先找到最小的长度
    min_length = min(len(fpr) for fpr in all_fpr)

    # 对每个子数组进行截断或插值，使它们具有相同的长度
    all_fpr_fixed = [np.interp(np.linspace(0, 1, min_length), fpr, fpr) for fpr in all_fpr]
    all_tpr_fixed = [np.interp(np.linspace(0, 1, min_length), tpr, tpr) for tpr in all_tpr]
    # 然后计算平均值
    mean_fpr = np.mean(all_fpr_fixed, axis=0)
    mean_tpr = np.mean(all_tpr_fixed, axis=0)
    mean_auc = np.mean(all_auc)
    # np.savetxt(r'mean_fpr.txt', mean_fpr, delimiter='\t', fmt='%.9f')
    # np.savetxt(r'mean_tpr.txt', mean_tpr, delimiter='\t', fmt='%.9f')

    plt.plot(fpr, tpr, label=f'Mean ROC (AUC = {mean_auc:.4f})', linestyle='-')
    # Plot the diagonal chance line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for 10-fold Random Multi-Row Zero Cross-Validation')
    plt.legend(loc='lower right')
    # 保存图像到文件
    plt.savefig('roc_10fold.png')
    plt.show()

    # 绘制Precision-Recall曲线
    # plt.figure(figsize=(8, 6))
    #
    # for i in range(len(all_precision)):
    #     plt.plot(all_recall[i], all_precision[i], label=f'PR fold {i + 1} (AUPR = {all_aupr[i]:.4f})',
    #              linestyle='-')
    #
    # # 先找到最小的长度
    # min_length = min(len(recall) for recall in all_recall)
    #
    # # 对每个子数组进行截断或插值，使它们具有相同的长度
    # all_recall_fixed = [np.interp(np.linspace(0, 1, min_length), recall, recall) for recall in all_recall]
    # all_precision_fixed = [np.interp(np.linspace(0, 1, min_length), precision, precision) for precision in
    #                        all_precision]
    #
    # # 然后计算平均值
    # mean_recall = np.mean(all_recall_fixed, axis=0)
    # mean_precision = np.mean(all_precision_fixed, axis=0)
    # mean_aupr = np.mean(all_aupr)
    #
    # plt.plot(recall, precision, label=f'Mean PR (AUPR = {mean_aupr:.4f})', linestyle='-')
    # plt.plot([0, 1], [1, 0], linestyle='--', color='gray')
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('PR Curve for 5-Fold CV')
    # plt.legend(loc='lower left')
    # # 保存图像到文件
    # plt.savefig('pr_curve_5fold.png')
    # plt.show()