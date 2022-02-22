from __future__ import print_function, division
import argparse
from keras.models import Model
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from math import sqrt
from utils import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import fisher_exact
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
import warnings
import torch
from sklearn.metrics import auc
import os
import statsmodels.api as sm
from pandas.core.frame import DataFrame
import pickle
import statsmodels.stats.multitest as mt
from keras.utils.generic_utils import get_custom_objects
warnings.filterwarnings('ignore')
seed = 1
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
max_value_1 = 0.0
max_value_2 = 0.0
our_threshold = 0.05
global auc_value_1
global auc_value_2
global pr_con
global pr_num
global pr_pro
global roc_cell_20
global roc_cgc_20

methods_names = ['2020plus',  'CompositeDriver',
              'e-Driver', 'e-Driver3D', 'MuSiC',
              'MutSig2CV',
              'OncodriveFML']

cancers = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
           'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV',
           'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM',
           'UCEC','UCS', 'UVM']


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=2, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


get_custom_objects().update({'focal_loss': focal_loss})


class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int  # key and query dimension
    dim_v: int  # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=2):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        batch, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads  # 2
        dk = self.dim_k // nh  # dim_k of each head 1
        dv = self.dim_v // nh  # dim_v of each head 1

        q = self.linear_q(x.reshape(batch, dim_in)).reshape(batch, nh, dk)  # (batch, nh, n, dk) 5.reshape(16,5,2)
        k = self.linear_k(x.reshape(batch, dim_in)).reshape(batch, nh, dk)  # (batch, nh, n, dk)
        v = self.linear_v(x.reshape(batch, dim_in)).reshape(batch, nh, dv)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(1, 2)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, self.dim_v)  # batch, n, dim_v
        # print(att.shape)
        return att


class Transformer(nn.Module):

    def __init__(self, n_input,):
        super(Transformer, self).__init__()
        self.n = 33  # 47
        self.fc1 = nn.Linear(n_input, self.n)  # 34
        self.attn1 = MultiHeadSelfAttention(dim_in=self.n, dim_k=4, dim_v=8)  # 4
        self.fc5 = nn.Linear(8, 2)
        self.drop1 = nn.Dropout(0.08)

    def encoder(self, x):
        x1 = self.fc1(x)
        x1 = F.gelu(x1)
        x1 = self.drop1(x1)
        x2 = self.attn1(x1)
        x2 = F.gelu(x2)
        x5 = self.fc5(x2)
        x5 = F.softmax(x5)
        return x5

    def forward(self, x):
        z = self.encoder(x)
        return z


class TransModel(nn.Module):

    def __init__(self,
                 n_input,):
        super(TransModel, self).__init__()

        self.transformer = Transformer(
            n_input=n_input,
            )

        self.model = Model()
        self.model.compile(loss=focal_loss)

    def pretrain(self):
        pretrain_transformer(self.transformer)

    def forward(self, x):
        x = self.transformer(x)
        return x


criteria = focal_loss(alpha=0.4, gamma=1.75)   # 0.5 2.5


def pretrain_transformer(model):
    """
    pretrain transformer
    """
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_re = []
    for epoch in range(args.epoch):
        total_loss = 0.
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(torch.float32)
            x = x.to(device)
            optimizer.zero_grad()
            preds = model(x)
            y = torch.tensor(y, requires_grad=False).to(device)
            y = y.to(torch.int64)
            loss = criteria(preds, y)
            total_loss += loss
            loss.backward()
            optimizer.step()
        loss_re.append(total_loss.detach().cpu().numpy())

    plt.plot(loss_re)
    plt.title('Train loss')
    plt.show()
    plt.cla()


def draw_from_dict(dicdata, RANGE, heng=0,):
    by_value = sorted(dicdata.items(),key = lambda item:item[1],reverse=True)
    x = []
    y = []
    for d in by_value:
        x.append(d[0])
        y.append(d[1])
    if heng == 0:
        plt.bar(x[0:RANGE], y[0:RANGE])
        plt.show()
        return
    elif heng == 1:
        plt.barh(x[0:RANGE], y[0:RANGE])
        plt.show()
        return
    else:
        return "The value of heng is only 0 or 1!"


def fisher_ex(a, b, c, d):
    _, pvalue = fisher_exact([[a, b], [c, d]], 'greater')
    if pvalue < 1e-256:
        pvalue = 1e-256
    # p1 = -math.log10(pvalue)
    p1 = pvalue

    return p1


def train():
    model = TransModel(
        n_input=args.n_input,).to(device)
    print(model)
    print('---------------------------------------')
    model.transformer.train()
    model.pretrain()
    pre_data, _ = load_data(test_path)
    data = pre_data.data
    data = torch.Tensor(data).to(device)
    label = pre_data.label
    model.transformer.eval()
    pre = model.transformer(data).cpu()
    pos_prob_model = pre[:, -1].detach().numpy()

    cgc_data, _ = load_data(cgc_path)
    data = cgc_data.data
    data = torch.Tensor(data).to(device)
    cgc_label = cgc_data.label
    model.transformer.eval()
    pre = model.transformer(data).cpu()
    pos_prob_cgc_model = pre[:, -1].detach().numpy()

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    def other_roc(path, pr_path, num):
        df = pd.read_csv(r'../data/roc_data/%s/PANCAN.csv' % (path), sep=',')
        df['in_2018'] = 0
        test_2018 = pd.read_csv(pr_path, sep=',')
        test_2018_gene = test_2018['Hugo_Symbol'].values.tolist()

        for index, row in df.iterrows():
            if df.loc[index, 'gene'] in test_2018_gene:
                df.loc[index, 'in_2018'] = 1
        newdf1 = df.loc[df['gene'].isin(test_2018_gene)]
        newdf1 = df.loc[(df['in_2018'] == 1)]
        newdf1 = newdf1[['gene', 'pvalue']]
        lab = newdf1['gene']
        com = test_2018.loc[test_2018['Hugo_Symbol'].isin(lab)]
        com.to_csv(r'd:/other/%s.csv' % (path))
        com_2018 = com
        com = com[['Hugo_Symbol', 'class']]
        com.columns = ['gene', 'class']
        c = pd.merge(newdf1, com, on='gene', how='outer')  # 外连接
        c = c.drop_duplicates(subset=['gene'])
        c['pvalue'] = c[['pvalue']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        p = c['pvalue'].values.tolist()
        la = c['class'].values.tolist()
        list_x = c['pvalue'].values.tolist()
        list_y = c['class'].values.tolist()
        fpr, tpr, thresholds1 = roc_curve(list_y, list_x)
        auc1 = auc(fpr, tpr)
        if path == '2020plus' and num == 1:
            global roc_cell_20
            roc_cell_20 = auc1
            print(roc_cell_20)
        if path == '2020plus' and num == 2:
            global roc_cgc_20
            roc_cgc_20 = auc1
            print(roc_cgc_20)
        if num == 1:
            title = 'TCGA Roc Curve'
        else:
            title = 'CGC Roc Curve'
        plt.plot(fpr, tpr, label='%s(AUC=%0.3f)' % (path, auc1))
        plt.title(title)

    # file_names = ['ActiveDriver', 'e-Driver','2020plus',
    #  'CompositeDriver',  'CHASM',  'MuSiC',]
    file_names = ['2020plus']

    fpr, tpr, thresholds1 = roc_curve(label, pos_prob_model)
    auc1 = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='Trans-Driver(AUC=%0.3f)' % auc1)
    global auc_value_1
    auc_value_1 = auc1

    for i in file_names:
        other_roc(i, test_path, 1)
    plt.legend()
    plt.show()
    plt.cla()
    fpr, tpr, thresholds2 = roc_curve(cgc_label, pos_prob_cgc_model)
    auc2 = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='Trans-Driver(AUC=%0.3f)' % auc2)
    global auc_value_2
    auc_value_2 = auc2

    for i in file_names:
        other_roc(i, cgc_path, 2)
    plt.legend()
    plt.show()

    def prediction_gene(predata, path, hg):
        data = predata.data
        data = torch.Tensor(data).to(device)
        model.transformer.eval()
        pre = model.transformer(data).cpu()
        pos_prob_model = pre[:, -1].detach().numpy()
        sym = DataFrame(hg)
        pos_pd = DataFrame(pos_prob_model)
        data = pd.concat([sym, pos_pd], axis=1)
        data.columns = ['Hugo_Symbol', 'Score']
        data.to_csv(path, sep=',', index=False)

    df_back, hg = load_data(all_gene_path)
    df_raw, hg_1 = load_data(all_ori_genes_path)
    back_path = r'../data/back.csv'
    raw_to_path = r'../data/raw.csv'
    prediction_gene(df_back, back_path, hg)
    prediction_gene(df_raw, raw_to_path, hg_1)

    pro = pd.read_csv(back_path, sep=',')
    null_dist = sm.distributions.ECDF(pro['Score'].values.tolist())
    out_path = r'../data/Pancan.null'
    fp = open(out_path, 'wb')
    pickle.dump(null_dist, fp)
    fp.close()
    f = open(out_path, 'rb')
    null_dist = pickle.load(f)
    f.close()
    pro = pd.read_csv(raw_to_path, sep=',')
    pvals = 1 - null_dist(pro['Score'].values.tolist())
    pro['p'] = pvals
    _, qvals, _, _ = mt.multipletests(pvals=pvals, alpha=our_threshold, method='fdr_bh')
    pro['q'] = qvals
    pro_show = pro[pro['q'] < our_threshold]
    pro = pro.sort_values(by=['Score'], ascending=[False])
    save_path = r'../data/results/multiple_{}.csv'.format(1)
    gene_path = r'../data/roc_data/Trans-Driver/PANCAN.csv'
    sa_path = r'../data/cancer_type_data/Trans-Driver/PANCAN.csv'
    pro.to_csv(save_path, sep=',', header=True, index=False)
    pro = pro.rename(columns={'Hugo_Symbol': 'gene', "q": "qvalue"})
    pro.to_csv(gene_path, sep=',', header=True, index=False)
    pro.to_csv(sa_path, sep=',', header=True, index=False)

    cgc_data = pd.read_csv(r'../data/cgc_somatic.csv', sep='\t')  #'Gene'
    cgc_data = cgc_data['Gene'].values.tolist()
    hg_1_data = hg_1.values.tolist()
    all_con = [item for item in hg_1_data if item in cgc_data]
    len_cgc = len(cgc_data)
    len_pro = len(pro_show)
    df_test = pro_show['Hugo_Symbol'].values.tolist()
    con = [item for item in df_test if item in cgc_data]

    remain_pr_data = pro.loc[pro['qvalue'] >= our_threshold]
    remain_pr_data = remain_pr_data.drop_duplicates(['gene'])

    remain_pr_hg = remain_pr_data['gene'].values.tolist()

    remain_pr_cgc = [item for item in remain_pr_hg if item in cgc_data]

    len_pr_cgc = len(con)
    len_pr_not_cgc = len(pro_show) - len(con)
    len_remain_cgc = len(remain_pr_cgc)
    number_cancer_1 = 20000 - len(pro_show) - len_remain_cgc

    p = fisher_ex(len_pr_cgc, len_pr_not_cgc, len_remain_cgc, number_cancer_1)
    print(p)
    len_con = len(con)
    if len_pro == 0:
        fra = 0
    else:
        fra = len_con / len_pro
    print('{}/{}'.format(len_con, len_cgc))
    print('{}/{}'.format(len_con, len_pro))
    global pr_con
    pr_con = len_con
    global pr_num
    pr_num = len_pro
    global pr_pro
    pr_pro = fra
    print(fra)
    path = r'../data/cancer_type_new_fea_4'

    dirs = os.listdir(path=path)

    d = {}
    cal_df = pd.DataFrame(columns=['Cancer', 'ActiveDriver', 'e-Driver', '2020plus',
                                    'CompositeDriver', 'CHASM', 'OncodriveCLUST', 'Our', 'DriverNet', 'HotMAPS'])
    len_each = {}
    for file in dirs:

        file_path = path + '/' + file
        all_gene = file_path + '/' + 'all_sim_gene.csv'
        all_ori_gene = file_path + '/' + 'all_ori_gene.csv'

        df_back, hg = load_data(all_gene)
        df_raw, hg_1 = load_data(all_ori_gene)
        back_path = file_path + '/' + 'back.csv'
        raw_to_path = file_path + '/' + 'raw.csv'

        prediction_gene(df_back, back_path, hg)
        prediction_gene(df_raw, raw_to_path, hg_1)

        pro = pd.read_csv(back_path, sep=',')
        null_dist = sm.distributions.ECDF(pro['Score'].values.tolist())

        out_path = file_path + '/' + '{}.null'.format(file)
        fp = open(out_path, 'wb')
        pickle.dump(null_dist, fp)
        fp.close()
        f = open(out_path, 'rb')
        null_dist = pickle.load(f)
        f.close()
        pro = pd.read_csv(raw_to_path, sep=',')
        pvals = 1 - null_dist(pro['Score'].values.tolist())
        pro['p'] = pvals
        _, qvals, _, _ = mt.multipletests(pvals=pvals, alpha=our_threshold, method='fdr_bh')
        pro['q'] = qvals
        pro_show = pro[pro['q'] < our_threshold]
        pro = pro.sort_values(by=['Score'], ascending=[False])
        save_path = r'../data/cancer_type_data/Trans-Driver' + '/' + '%s.csv' %(file)
        pro = pro.rename(columns={'Hugo_Symbol': 'gene', "q": "qvalue"})
        pro.to_csv(save_path, sep=',', header=True, index=False)
        len_pro_show = len(pro_show)
        print(file + ':' + str(len_pro_show))
        len_each[file] = len_pro_show
    fish_res = {}
    draw_from_dict(len_each, len(len_each), 1, )
    for file in cancers:
        pr = {}
        other_methods_path = r'../data/cancer_type_data'
        method_dirs = os.listdir(path=other_methods_path)
        spec_cancers_path = r'../data/cell_2018'
        cancer_data_path = os.path.join(spec_cancers_path, file+'.csv')
        cancer_data = pd.read_csv(cancer_data_path, sep=',')
        cancer_data = cancer_data.drop_duplicates(['Gene'])
        cancer_data_hg = cancer_data['Gene'].values.tolist()
        methods = ['ActiveDriver', '2020plus',  'e-Driver',
                    'CompositeDriver',  'CHASM', 'OncodriveCLUST', 'DriverNet', 'Trans-Driver', 'HotMAPS', 'MuSiC']

        for method_file in method_dirs:
            if method_file not in methods:
                continue
            method_data_path = os.path.join(other_methods_path, method_file)
            other_methods_data_path = os.path.join(method_data_path, file+'.csv')
            if not os.path.exists(other_methods_data_path):
                continue
            data = pd.read_csv(other_methods_data_path, sep=',')
            if method_file == '2020plus' or method_file == 'CompositeDriver' or method_file == 'OncodriveCLUST' or method_file=='DriverNet':
                q = 0.05
            elif method_file == 'ActiveDriver':
                q = 0.0001
            elif method_file == 'OncodriveFML':
                q = 0.25
            elif method_file == 'HotMAPS':
                q = 0.01
            elif method_file == 'MuSiC':
                q = 0.0000000001
            elif method_file == 'MutSig2CV' or method_file == 'e-Driver' \
                or method_file == 'CHASM' or method_file == 'VEST':
                q = 0.1
            else:
                q = 0.05

            pr_data = data.loc[data['qvalue'] < q]

            pr_data = pr_data.drop_duplicates(['gene'])

            remain_pr_data = data.loc[data['qvalue'] >= q]
            remain_pr_data = remain_pr_data.drop_duplicates(['gene'])

            pr_data_hg = pr_data['gene'].values.tolist()
            remain_pr_hg = remain_pr_data['gene'].values.tolist()

            pr_cgc = [item for item in pr_data_hg if item in cancer_data_hg]
            remain_pr_cgc = [item for item in remain_pr_hg if item in cancer_data_hg]

            len_pr_cgc = len(pr_cgc)
            len_pr_not_cgc = len(pr_data) - len(pr_cgc)
            len_remain_cgc = len(remain_pr_cgc)
            number_cancer_1 = 20000 - len(pr_data) - len_remain_cgc
            p = fisher_ex(len_pr_cgc,  len_pr_not_cgc, len_remain_cgc,  number_cancer_1)
            fish_res[method_file] = p
            pr[method_file] = p
        pr['Cancer'] = file
        plt.cla()
        plt.title(file)
        cal_df = cal_df.append(pr, ignore_index=True)
        plt.cla()

    cal_df.to_csv(r'../results/each_cancer_fisher_1.csv', sep=',', index=False)
    print(cal_df)
    print(cal_df.median())
    cal_df.to_csv(r'../results/cal_median_1.csv', sep=',', index=False)
    for i in cal_df.median():
        print(i)


if __name__ == "__main__":

    m_lr = 0
    m_ba = 0
    m_er = 0
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--batch_size', default=13, type=int)
    parser.add_argument('--epoch', default=30, type=int)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    # features_pancan = r'D:/cacnerdata/con_2020feas_1/features_pancan_our_2020_1.csv'

    # train_path = r'D:/cacnerdata/pancan_fea/create_data_1/new_data/train_12_10.csv'
    train_path = r'../data/train.csv'
    # test_path = r'D:/cacnerdata/pancan_fea/create_data_1/new_data/test_11_14.csv'
    test_path = r'../data/test.csv'
    # all_ori_genes_path = r'D:/cacnerdata/pancan_fea/create_data_1/new_data/all_ori_fea.csv'
    all_ori_genes_path = r'../data/original.csv'
    # all_gene_path = r'D:/cacnerdata/pancan_fea/create_data_1/new_data/all_sim_fea.csv'
    all_gene_path = r'../data/simulation.csv'
    # cgc_path = r'D:/cacnerdata/pancan_fea/create_data_1/new_data/cgc_11_14.csv'
    cgc_path = r'../data/cgc.csv'
    args.n_input = 46
    dataset, _ = load_data(train_path)
    train()
    m_lr = args.lr
    m_ba = args.batch_size
    m_er = args.epoch
    max_value_1 = auc_value_1
    max_value_2 = auc_value_2
    with open(r'../results/log.txt', 'a+') as f:
        f.writelines(str(m_lr)+'\t'+str(m_ba)+'\t'+str(m_er) + '\t' + str(max_value_1) + '\t\t' + str(roc_cell_20) + '\t\t'
                     + str(max_value_2) + '\t\t' + str(roc_cgc_20) + '\t\t' + str(pr_con) + '\t' + str(pr_num) + '\t' + str(pr_pro) + '\t' + '\n')
    f.close()

