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
#以下两句用来忽略版本错误信息
import warnings

warnings.filterwarnings("ignore")
#设置device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#从超图关联矩阵H计算G,param H: 超图关联矩阵H,param variable_weight: 超边的权重是否可变
def generate_G_from_H(H, variable_weight=False):
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G
#从超图关联矩阵H计算G,param H: 超图关联矩阵H,param variable_weight: 超边的权重是否可变
def _generate_G_from_H(H, variable_weight=False):
    H = np.array(H)
    n_edge = H.shape[1]
    W = np.ones(n_edge) # 超边的权重
    DV = np.sum(H * W, axis=1) # 节点的度
    DE = np.sum(H, axis=0) # 超边的度

    a = []
    for i in DE:
        if i == 0:
            a.append(0)
        else:
            a.append(np.power(i, -1))
    invDE = np.mat(np.diag(a))

    b = []
    for i in DV:
        if i == 0:
            b.append(0)
        else:
            b.append(np.power(i, -0.5))
    DV2 = np.mat(np.diag(b))

    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2

    G = torch.Tensor(G)
    return G 

#将H_list中的超边组合并,param H_list: 包含两个或两个以上超图关联矩阵的超边组
#return: 融合后的超图关联矩阵
def hyperedge_concat(*H_list):
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


#注意力机制融合三个图
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=56):
        super(Attention, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(1.5))  # 学习温度系数
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        z = z.float()
        w = self.project(z) / self.temperature
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta







class node_encoder(nn.Module):
    def __init__(self, num_in_node, num_hidden, dropout, act=torch.tanh):
        super(node_encoder, self).__init__()
        self.num_in_node = num_in_node
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = lambda x: x  # 直接在这里指定恒等激活函数
        # self.act = act
        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_node, self.num_hidden), dtype=torch.double))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.double))

    def forward(self, H):
        z1 = self.act(H.mm(self.W1) + 2*self.b1)
        return self.act(z1)
    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_node) + ' -> ' + str(self.num_hidden)



class decoder2(nn.Module):
    def __init__(self, dropout=0.8, act=torch.sigmoid):
        super(decoder2, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = lambda x: x  # 直接在这里指定恒等激活函数
    def forward(self, z_node, z_hyperedge):
        z_node_ = self.dropout(z_node)
        z_hyperedge_ = self.dropout(z_hyperedge)
        z = self.act(z_node_.mm(z_hyperedge_.t()))
        return self.act(z) 



class decoder1(nn.Module):
    def __init__(self, dropout=0.5, act=torch.sigmoid):
        super(decoder1, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = lambda x: x  # 直接在这里指定恒等激活函数

    def forward(self, z_node, z_hyperedge):
        z_node_ = z_node
        z_hyperedge_ = z_hyperedge
        z = self.act(z_node_.mm(z_hyperedge_.t()))
        return self.act(z)





class HGNN_conv1(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):  #
        super(HGNN_conv1, self).__init__()
        self.weight = Parameter(torch.DoubleTensor(in_ft, out_ft))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = Parameter(torch.DoubleTensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.linear_x_1 = nn.Linear(in_ft, out_ft).to(torch.double)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):  # x: torch.Tensor, G: torch.Tensor
        x = x.double()
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x) + x
        return x

class HGNN1(nn.Module):
    def __init__(self, in_ch, n_hid, n_class, n_node, emb_dim, n_hid_2=128,dropout=0.5): #这个函数的作用就是可传，可自定义
        super(HGNN1, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv1(in_ch, n_hid)
        self.hgc2 = HGNN_conv1(n_hid, n_class)

    def forward(self, x, G):
        G = G + torch.eye(G.shape[0]).to(device)
        x= self.hgc1(x, G)
        x = torch.tanh(x)
        x= self.hgc2(x, G)
        return x

class hyperedge_encoder(nn.Module):
     def __init__(self, num_in_edge, num_hidden, dropout, act=torch.tanh):
         super(hyperedge_encoder, self).__init__()
         self.num_in_edge = num_in_edge
         self.num_hidden = num_hidden
         # self.dropout = dropout
         self.act = act
         self.act = lambda x: x  # 直接在这里指定恒等激活函数
         self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_edge, self.num_hidden), dtype=torch.double))
         nn.init.xavier_uniform_(self.W1.data, gain=1.414)
         self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.double))

     def forward(self, H_T):
         z1 = self.act(torch.mm(H_T, self.W1) + self.b1)
         return z1
     def __repr__(self):
         return self.__class__.__name__ + '(' + str(self.num_in_edge) + ' -> ' + str(self.num_hidden)

# 修改后的模型，包含多头注意力
class Gai_HGNN(nn.Module):


    def __init__(self, num_in_node = 129, num_in_edge = 48, num_hidden1 = 256, num_out=128):
        super(Gai_HGNN, self).__init__()
        value = 284
        # 注意力机制
        self.attention = Attention(155)
        # 特异性编码器
        self.hgnn_encoder1 = HGNN1(value, value, num_out, value, value)  # 只传前三个参数
        self.hgnn_encoder2 = HGNN1(value, value, num_out, value, value)  # 只传前三个参数
        # # 共享编码器
        self.hgnn_encoder3 = HGNN1(value, value, num_out, value, value)  # 只传前三个参数
        self.node_encoders1 = node_encoder(value, num_hidden1, 0.3)
        self.hyperedge_encoders1 = hyperedge_encoder(value, num_hidden1, 0.3)
        self.decoder1 = decoder1()
        self.decoder2 = decoder2()
        self._enc_mu_node = node_encoder(num_hidden1, num_out, 0.3)
        self._enc_log_sigma_node = node_encoder(num_hidden1, num_out, 0.3)
        self._enc_mu_hedge = node_encoder(num_hidden1, num_out, 0.3)
        self._enc_log_sigma_hyedge = node_encoder(num_hidden1, num_out, 0.3)


    def sample_latent(self, z_node, z_hyperedge):
        self.z_node_mean = self._enc_mu_node(z_node)  # mu
        self.z_node_log_std = self._enc_log_sigma_node(z_node)
        self.z_node_std = torch.exp(self.z_node_log_std)  # sigma
        z_node_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_node_std.size())).double()
        self.z_node_std_ = z_node_std_.to(device)
        self.z_node_ = self.z_node_mean + self.z_node_std.mul(Variable(self.z_node_std_, requires_grad=True))
        self.z_edge_mean = self._enc_mu_hedge(z_hyperedge)
        self.z_edge_log_std = self._enc_log_sigma_hyedge(z_hyperedge)
        self.z_edge_std = torch.exp(self.z_edge_log_std)
        z_edge_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_edge_std.size())).double()
        self.z_edge_std_ = z_edge_std_.to(device)
        self.z_hyperedge_ = self.z_edge_mean + self.z_edge_std.mul(Variable(self.z_edge_std_, requires_grad=True))

        if self.training:
            return self.z_node_, self.z_hyperedge_
        else:
            return self.z_node_mean, self.z_edge_mean


    def forward(self,matrix_A, matrix_B,heterogeneous, VAE_1, VAE_2):

        value = 155
        z_node_encoder = self.node_encoders1(VAE_2)
        z_hyperedge_encoder = self.hyperedge_encoders1(VAE_1)

        self.z_node_s, self.z_hyperedge_s = self.sample_latent(z_node_encoder, z_hyperedge_encoder)
        z_node = self.z_node_s
        z_hyperedge = self.z_hyperedge_s
        #超图部分***************************************************
        #两个异构网络的特异性编码器，果然不一样
        shuxing_feature = self.hgnn_encoder1(heterogeneous, matrix_A)#177*128
        jiegou_feature = self.hgnn_encoder2(heterogeneous, matrix_B) #177*128
        mir_feature_1 = shuxing_feature[:value, :]
        dis_feature_1 = shuxing_feature[value:, :]

        mir_feature_2 = jiegou_feature[:value, :]
        dis_feature_2 = jiegou_feature[value:, :]
        #**************************************

        # 两个异构网络的共享编码器2
        shuxing_feature1 = self.hgnn_encoder3(heterogeneous, matrix_A)#177*128
        jiegou_feature1 = self.hgnn_encoder3(heterogeneous, matrix_B) #177*128

        hsum = (shuxing_feature1+jiegou_feature1)/2


        mir_feature_5 = hsum[:value, :]
        dis_feature_5 = hsum[value:, :]


        #**********特异
        reconstructionMD = self.decoder1(dis_feature_2, mir_feature_2)
        reconstructionG = self.decoder1(dis_feature_1, mir_feature_1)

        #***********************************************共享
        reconstructionMMDD = self.decoder1(dis_feature_5, mir_feature_5)


        reconstruction_en = self.decoder2(z_node, z_hyperedge)
        result = self.z_node_mean.mm(self.z_edge_mean.t())

        #效果奇差
        emb = torch.stack([reconstructionMMDD , reconstructionMD, reconstructionG ], dim=1)
        emb, att = self.attention(emb)
        result_h = emb

        recover = 0.4*result + 0.6*result_h
        #总，加两个特异性网络编码器+共享编码器
        return reconstruction_en, result, reconstructionG, reconstructionMD, reconstructionMMDD, result_h ,recover,mir_feature_1 , mir_feature_2 , mir_feature_5 , dis_feature_1 ,dis_feature_2 ,dis_feature_5, shuxing_feature, jiegou_feature, hsum



def create_resultlist(result,testset,Index_PositiveRow,Index_PositiveCol,Index_zeroRow,Index_zeroCol,test_length_p,zero_length,test_f):
    result_list = np.zeros((test_length_p+len(test_f), 1))
    for i in range(test_length_p):
        result_list[i,0] = result[Index_PositiveRow[testset[i]], Index_PositiveCol[testset[i]]]
    for i in range(len(test_f)):
        result_list[i+test_length_p, 0] = result[Index_zeroRow[test_f[i]], Index_zeroCol[test_f[i]]]
    return result_list

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())
#要计算两组向量之间的余弦相似度矩阵

def cosine_similarity_matrix(h1, h2):
    h1_norm = F.normalize(h1, p=2, dim=1)
    h2_norm = F.normalize(h2, p=2, dim=1)
    return torch.mm(h1_norm, h2_norm.t())


def compute_contrastive_loss(h1, h2, temperature=0.7):
    sim_mat = cosine_similarity_matrix(h1, h2)
    # Scale the similarity by temperature
    sim_mat_scaled = torch.exp(sim_mat / temperature)
    # Compute the loss
    positive_pairs = torch.diag(sim_mat_scaled)
    all_pairs_sum = torch.sum(sim_mat_scaled, dim=1)
    # Avoid division by zero
    all_pairs_sum = torch.clamp(all_pairs_sum, min=1e-9)
    contrastive_loss = -torch.log(positive_pairs / all_pairs_sum).mean()
    return contrastive_loss
