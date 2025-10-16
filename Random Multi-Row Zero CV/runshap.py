from sklearn.metrics import matthews_corrcoef
import numpy as np
import copy
import math
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import DataEnhancement
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from model3 import *
from kl_loss import kl_loss
from utils import f1_score_binary, precision_binary, recall_binary, accuracy_binary
import scipy.sparse as sp
import plot_auc_curves
from hypergraph_utils import *
import shap
import torch
from tqdm import tqdm
import os

# 尝试导入不同版本的SHAP
try:
    # 新版SHAP (>=0.44.0)
    from shap import Explainer
    from shap import maskers

    SHAP_NEW_API = True
    print("Using new SHAP API")
except ImportError:
    try:
        # 旧版SHAP (<0.44.0)
        from shap import KernelExplainer

        SHAP_NEW_API = False
        print("Using old SHAP API")
    except ImportError:
        print("SHAP not installed. Please install with: pip install shap")
        exit(1)

# 以下两句用来忽略版本错误信息
import warnings

warnings.filterwarnings("ignore")

# 设置device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置随机数种子
seed = 48
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
torch.backends.cudnn.deterministic = True  # 确保CUDA卷积操作的确定性
torch.backends.cudnn.benchmark = False  # 禁用卷积算法选择，确保结果可重复


def laplacian_norm(adj):
    adj += np.eye(adj.shape[0])  # add self-loop
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


def construct_data_matrices(MD_data, MM_data, DD_data, DG_data, MG_data):
    """根据输入数据构建超图和异构网络"""
    HHMG = construct_H_with_KNN(MG_data)
    HMG = generate_G_from_H(HHMG).double()

    HHDG = construct_H_with_KNN(DG_data)
    HDG = generate_G_from_H(HHDG).double()

    HHMD = construct_H_with_KNN(MD_data)
    HMD = generate_G_from_H(HHMD).double()

    HHDM = construct_H_with_KNN(MD_data.T)
    HDM = generate_G_from_H(HHDM).double()

    HHMM = construct_H_with_KNN(MM_data)
    HMM = generate_G_from_H(HHMM).double()

    HHDD = construct_H_with_KNN(DD_data)
    HDD = generate_G_from_H(HHDD).double()

    # 构建矩阵
    lnc_shape = HMM.shape[0]
    dis_shape = HDD.shape[0]

    # 生物属性异构网络
    lnc_dis_mi = np.hstack((HMM, np.zeros((lnc_shape, dis_shape))))
    dis_lnc_mi = np.hstack((np.zeros((dis_shape, lnc_shape)), HDD))
    matrix_A = np.vstack((lnc_dis_mi, dis_lnc_mi))
    matrix_A = laplacian_norm(matrix_A)

    # 网络拓扑异构网络
    lnc_dis_sim = np.hstack((HMG, np.zeros((lnc_shape, dis_shape))))
    dis_lnc_sim = np.hstack((np.zeros((dis_shape, lnc_shape)), HDG))
    matrix_B = np.vstack((lnc_dis_sim, dis_lnc_sim))
    matrix_B = laplacian_norm(matrix_B)

    # VAE输入
    VAE_1 = np.hstack((HMD, MD_data))
    VAE_2 = np.hstack((HDM, MD_data.T))

    return matrix_A, matrix_B, HMD, HDM, HMM, HDD, HMG, HDG, VAE_1, VAE_2


def train(epochs):
    X = copy.deepcopy(MD)
    Xn = copy.deepcopy(X)

    model = Gai_HGNN()
    optimizer2 = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    def duo_hg():
        # 使用全局数据构建超图和异构网络
        matrix_A, matrix_B, HMD, HDM, HMM, HDD, HMG, HDG, VAE_1, VAE_2 = construct_data_matrices(
            MD, MM, DD, DG, MG
        )

        # 转换为张量并发送到设备
        matrix_A = torch.tensor(matrix_A).to(device)
        matrix_B = torch.tensor(matrix_B).to(device)
        HMG = torch.tensor(HMG).to(device)
        HDG = torch.tensor(HDG).to(device)
        HMD = torch.tensor(HMD).to(device)
        HDM = torch.tensor(HDM).to(device)
        HMM = torch.tensor(HMM).to(device)
        HDD = torch.tensor(HDD).to(device)
        A = torch.tensor(MD).to(device)
        AT = torch.tensor(MD.T).to(device)
        VAE_1 = torch.tensor(VAE_1).to(device)
        VAE_2 = torch.tensor(VAE_2).to(device)

        mir_feat = torch.eye(155).to(device)
        dis_feat = torch.eye(129).to(device)
        heterogeneous = torch.eye(284).to(device)

        return AT, A, HMG, HDG, mir_feat, dis_feat, HMD, HDM, HMM, HDD, matrix_A, matrix_B, heterogeneous, VAE_1, VAE_2

    AT, A, HMG, HDG, mir_feat, dis_feat, HMD, HDM, HMM, HDD, matrix_A, matrix_B, heterogeneous, VAE_1, VAE_2 = duo_hg()

    pos_weight = float(A.shape[0] * A.shape[1] - A.sum()) / A.sum()

    loss_kl = kl_loss(129, 155)
    for epoch in tqdm(range(epochs), desc='Training Epochs'):
        model.train()
        reconstruction1, result, reconstructionG, reconstructionMD, reconstructionMMDD, result_h, recover, mir_feature_1, mir_feature_2, mir_feature_5, dis_feature_1, dis_feature_2, dis_feature_5, shuxing_feature, jiegou_feature, hsum = model(
            matrix_A, matrix_B, heterogeneous, VAE_1, VAE_2)

        MA = A
        reG = reconstructionG.t()
        reMD = reconstructionMD.t()
        reMMDD = reconstructionMMDD.t()
        ret = result.t()
        re1 = reconstruction1.t()
        rec = recover.t()

        loss_k = loss_kl(model.z_node_log_std, model.z_node_mean, model.z_edge_log_std, model.z_edge_mean)
        loss_2 = loss_k + F.binary_cross_entropy_with_logits(re1, MA,
                                                             pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(
            ret, MA, pos_weight=pos_weight)
        loss_1 = F.binary_cross_entropy_with_logits(reG, MA,
                                                    pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(reMMDD,
                                                                                                                MA,
                                                                                                                pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(
            reMD, MA, pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(rec, MA, pos_weight=pos_weight)
        loss = loss_1 + loss_2

        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()

    # 最终预测
    model.eval()
    with torch.no_grad():
        _, _, _, _, _, _, recover, _, _, _, _, _, _, _, _, _ = model(
            matrix_A, matrix_B, heterogeneous, VAE_1, VAE_2)
        outputs = recover.t().cpu().detach().numpy()

    return model, outputs


def explain_with_shap(model, sample_index, background_size=30):
    """
    使用SHAP解释模型对特定样本的预测
    :param model: 训练好的模型
    :param sample_index: 要解释的样本索引 (drug_index, snoRNA_index)
    :param background_size: 背景数据集大小
    """
    model.eval()
    drug_idx, snoRNA_idx = sample_index

    # 创建背景数据集 - 随机选择背景样本
    background = []
    for _ in range(background_size):
        # 复制原始数据
        bg_MD = MD.copy()
        bg_MM = MM.copy()
        bg_DD = DD.copy()
        bg_DG = DG.copy()
        bg_MG = MG.copy()

        # 随机扰动部分数据
        perturbation_mask = np.random.choice([0, 1], size=MD.shape, p=[0.8, 0.2])
        bg_MD = np.where(perturbation_mask, np.random.rand(*MD.shape), bg_MD)

        # 添加到背景集
        background.append((bg_MD, bg_MM, bg_DD, bg_DG, bg_MG))

    # 定义预测函数
    def model_predict(input_data):
        """
        SHAP需要的预测函数
        :param input_data: 输入数据列表 [(MD, MM, DD, DG, MG), ...]
        :return: 预测结果数组
        """
        predictions = []
        model.eval()

        with torch.no_grad():
            for data in input_data:
                MD_data, MM_data, DD_data, DG_data, MG_data = data

                # 构建数据矩阵
                matrix_A, matrix_B, _, _, _, _, _, _, VAE_1, VAE_2 = construct_data_matrices(
                    MD_data, MM_data, DD_data, DG_data, MG_data
                )

                # 转换为张量并发送到设备
                matrix_A = torch.tensor(matrix_A).float().to(device)
                matrix_B = torch.tensor(matrix_B).float().to(device)
                VAE_1 = torch.tensor(VAE_1).float().to(device)
                VAE_2 = torch.tensor(VAE_2).float().to(device)
                heterogeneous = torch.eye(284).float().to(device)

                # 模型预测
                _, _, _, _, _, _, recover, _, _, _, _, _, _, _, _, _ = model(
                    matrix_A, matrix_B, heterogeneous, VAE_1, VAE_2)

                # 获取特定样本的预测值
                prediction = recover[drug_idx, snoRNA_idx].item()
                predictions.append(prediction)

        return np.array(predictions)

    # 创建要解释的样本 - 使用原始数据
    sample = [(MD, MM, DD, DG, MG)]

    # 使用不同版本的SHAP API
    if SHAP_NEW_API:
        # 新版SHAP (>=0.44.0)
        masker = maskers.Independent(background)
        explainer = Explainer(model_predict, masker)
        shap_values = explainer(sample)
        shap_values = shap_values.values
        expected_value = explainer.expected_value[0]
    else:
        # 旧版SHAP (<0.44.0)
        explainer = KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(sample, nsamples=100)
        expected_value = explainer.expected_value

    # 分析并可视化结果
    print(f"\nSHAP values for sample (Drug {drug_idx}, snoRNA {snoRNA_idx}):")

    # 获取特征名称
    feature_names = [
        "Drug-snoRNA Interaction Matrix",
        "snoRNA Sequence Similarity",
        "Drug Structure Similarity",
        "Drug-Gene Interactions",
        "snoRNA-Gene Interactions"
    ]

    # 创建SHAP摘要图
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"Feature Importance for Prediction (Drug {drug_idx}, snoRNA {snoRNA_idx})")
    plt.tight_layout()
    plt.savefig(f"shap_summary_{drug_idx}_{snoRNA_idx}.png", dpi=300)
    plt.close()

    # 创建SHAP力图
    plt.figure(figsize=(12, 6))

    # 处理不同版本的SHAP值格式
    if isinstance(shap_values, list):
        # 旧版格式
        shap_values_force = shap_values[0]
    else:
        # 新版格式
        shap_values_force = shap_values[0].values

    shap.force_plot(
        expected_value,
        shap_values_force,
        feature_names=feature_names,
        show=False,
        matplotlib=True
    )
    plt.title(f"SHAP Force Plot for Prediction (Drug {drug_idx}, snoRNA {snoRNA_idx})")
    plt.tight_layout()
    plt.savefig(f"shap_force_{drug_idx}_{snoRNA_idx}.png", dpi=300)
    plt.close()

    return shap_values


# 主函数
if __name__ == '__main__':
    # 读数据
    # 关联矩阵
    MD = np.loadtxt("snoRNA(155,129)/new_matrix.txt")
    # miRNA相似性矩阵
    MM = np.loadtxt("snoRNA(155,129)/snoBIANJI.txt")
    # 疾病相似性矩阵
    DD = np.loadtxt("snoRNA(155,129)/drug.txt")
    # 疾病基因矩阵
    DG = np.loadtxt("snoRNA(155,129)/GKGIP_drug.txt")
    # 基因疾病矩阵
    MG = np.loadtxt("snoRNA(155,129)/GKGIP_snoRNA.txt")

    # 定义模型超参数
    lr = 0.002
    p = 0.3
    weight_decay = 0.02
    temperature = 0.2
    epochs = 100
    dropout = 0.5

    # 训练模型并获取结果
    trained_model, result = train(epochs)

    # 保存模型
    torch.save(trained_model.state_dict(), 'gai_hgnn_model.pth')
    print("Model saved to 'gai_hgnn_model.pth'")

    # 原始关联矩阵中为0的位置
    zero_indices = np.where(MD == 0)
    predicted_scores = result[zero_indices]

    # 获取按分数排序后的前100个索引（从大到小排序）
    top_100_indices = np.argsort(predicted_scores)[::-1][:100]

    # 提取对应的行索引、列索引和分数
    top_100_rows = zero_indices[0][top_100_indices]
    top_100_cols = zero_indices[1][top_100_indices]
    top_100_scores = predicted_scores[top_100_indices]

    # 打印结果
    print("\nTop 100 predicted unknown associations:")
    for i in range(100):
        print(f"Drug {top_100_rows[i]}, snoRNA {top_100_cols[i]}: Score = {top_100_scores[i]:.6f}")

    # 选择前3个预测进行SHAP解释
    for i in range(min(3, len(top_100_rows))):  # 确保不超过实际数量
        drug_idx = top_100_rows[i]
        snoRNA_idx = top_100_cols[i]
        print(f"\nPerforming SHAP analysis for prediction #{i + 1}: Drug {drug_idx}, snoRNA {snoRNA_idx}")

        # 进行SHAP解释
        shap_values = explain_with_shap(trained_model, (drug_idx, snoRNA_idx), background_size=30)

        # 打印SHAP值
        print(f"SHAP values for each feature:")
        feature_names = [
            "Drug-snoRNA Interaction Matrix",
            "snoRNA Sequence Similarity",
            "Drug Structure Similarity",
            "Drug-Gene Interactions",
            "snoRNA-Gene Interactions"
        ]

        # 处理不同版本的SHAP值格式
        if isinstance(shap_values, list):
            # 旧版格式
            shap_vals = shap_values[0]
        else:
            # 新版格式
            shap_vals = shap_values[0].values

        for j, val in enumerate(shap_vals):
            print(f"{feature_names[j]}: {val:.4f}")

    print("\nSHAP analysis completed. Plots saved as 'shap_summary_*.png' and 'shap_force_*.png'")