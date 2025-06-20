import pandas as pd
import numpy as np
import random
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


mutations = ['Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del',
             'In_Frame_Ins', 'Missense_Mutation', 'Nonsense_Mutation',
             'Nonstop_Mutation', 'Silent', 'Splice_Site', 'Translation_Start_Site',
             'DEL', 'INS', 'SNP', 'pnum', 'gene length',
             'silent', 'nonsense', 'splice site', 'missense', 'recurrent missense',
             'normalized missense position entropy', 'frameshift indel',
             'inframe indel', 'normalized mutation entropy',
             'Mean Missense MGAEntropy',  'lost start and stop',
             'missense to silent', 'non-silent to silent',
            ]

other = ['Mean VEST Score', 'inactivating p-value', 'entropy p-value', 'vest p-value', 'combined p-value',  'replication_time', 'HiC_compartment',
         'gene_betweeness', 'gene_degree']

copy_num = ['all_mean', 'all_var', 'up_mean',
            'up_var', 'down_mean', 'down_var']

gene_expresssion = ['expression_CCLE']

dnam = ['dnam_mean', 'dnam_var']


seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
cancers = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
           'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV',
           'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM',
           'UCEC', 'UCS', 'UVM']

# cancers = ['ACC', 'BLCA', 'BRCA']


def pan_cancer():
    result_path = r'../data/data/cell_2018_other_method_spe_cancer/Our/PANCAN.csv'
    all_ori_genes_path = r'../data/all_ori_fea.csv'
    fea_import_cal_1(result_path, all_ori_genes_path, cancer='Pan cancer features importances')
    # print(fea_import_cal_1())


def each_cancer():
    path = r'../data/cancer_type_new_fea_4'
    cal_df = pd.DataFrame(columns=['Cancer', 'mu', 'other',  'copy_num', 'dnam', 'ex'])
    fea = {}
    for i in cancers:
        result_path = r'../data/cancer_type_data/Trans-Driver' + '/' + i + '.csv'
        fea_path = path + '/' + i + '/' + 'all_ori_gene.csv'
        print(i)
        la = i + 'features importances'
        a = fea_import_cal_1(result_path, fea_path, cancer=i)
        fea[i] = a
        new = pd.DataFrame({'Cancer': i,
                            'mu': a[0],
                            'other': a[1],
                            'copy_num': a[2],
                            'dnam': a[3],
                            'ex': a[4]}, index=[0])

        cal_df = cal_df.append(new, ignore_index=True)

    cal_df.to_csv(r'../result/new_all_cancer_fea_import.csv', sep=',', index=False)


value_dict = {}
iter_dict = {}
df_fea = pd.DataFrame(columns={'fea', 'import'})


def each_cancer_top():
    path = r'../data/cancer_type_new_fea_4'
    for i in cancers:
        result_path = r'../data/cancer_type_data/Trans-Driver' + '/' + i + '.csv'
        fea_path = path + '/' + i + '/' + 'all_ori_gene.csv'
        print(i)
        fea_import_cal_2(result_path, fea_path, cancer=i)
        print('---------------------------------------')


def fea_import_cal(result_path, features_path, cancer,):
    reslt = pd.read_csv(result_path)
    all_fea = pd.read_csv(features_path)
    reslt.rename(columns={'gene': 'Hugo_Symbol', 'qvalue': 'q'}, inplace=True)
    # print(reslt)
    data = pd.merge(reslt, all_fea, on='Hugo_Symbol')
    df = data
    # print(df)
    list = df.columns.values.tolist()
    l = ['Hugo_Symbol', 'class', 'Score', 'p', 'q']
    for i in l:
        list.remove(i)
    ss = StandardScaler()
    df[list] = ss.fit_transform(df[list])
    targets = df['Score'].values
    features = df.drop(['Hugo_Symbol', 'class', 'Score', 'p', 'q'], axis=1).values

    feat_labels = list
    model = RandomForestClassifier()
    model.fit(features.astype(int), targets.astype(int))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fea_imp = pd.DataFrame(columns=['feature', 'importance'])
    for f in range(features.shape[1], ):
        da = {'feature': feat_labels[indices[f]], 'importance': importances[indices[f]]}
        fea_imp = fea_imp.append(da, ignore_index=True)

    mu = fea_imp.loc[fea_imp['feature'].isin(mutations)]['importance'].sum()
    cn = fea_imp.loc[fea_imp['feature'].isin(copy_num)]['importance'].sum()
    dn = fea_imp.loc[fea_imp['feature'].isin(dnam)]['importance'].sum()
    ot = fea_imp.loc[fea_imp['feature'].isin(other)]['importance'].sum()
    ex = fea_imp.loc[fea_imp['feature'].isin(gene_expresssion)]['importance'].sum()

    a = [cancer, mu, ot, cn, dn, ex]
    return a


def fea_import_cal_1(result_path, features_path, cancer,):
    reslt = pd.read_csv(result_path)
    all_fea = pd.read_csv(features_path)
    reslt.rename(columns={'gene': 'Hugo_Symbol', 'qvalue': 'q'}, inplace=True)

    data = pd.merge(reslt, all_fea, on='Hugo_Symbol')
    df = data

    list = df.columns.values.tolist()
    l = ['Hugo_Symbol', 'class', 'Score', 'p', 'q']
    for i in l:
        list.remove(i)
    ss = StandardScaler()
    df[list] = ss.fit_transform(df[list])
    targets = df['Score'].values
    features = df.drop(['Hugo_Symbol', 'class', 'Score', 'p', 'q'], axis=1).values

    feat_labels = list
    model = RandomForestClassifier()
    model.fit(features.astype(int), targets.astype(int))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fea_imp = pd.DataFrame(columns=['feature', 'importance'])
    for f in range(features.shape[1], ):
        da = {'feature': feat_labels[indices[f]], 'importance': importances[indices[f]]}
        fea_imp = fea_imp.append(da, ignore_index=True)
    print(fea_imp)
    mu = fea_imp.loc[fea_imp['feature'].isin(mutations)]['importance'].mean()

    cn = fea_imp.loc[fea_imp['feature'].isin(copy_num)]['importance'].mean()
 
    dn = fea_imp.loc[fea_imp['feature'].isin(dnam)]['importance'].mean()

    ot = fea_imp.loc[fea_imp['feature'].isin(other)]['importance'].mean()

    ex = fea_imp.loc[fea_imp['feature'].isin(gene_expresssion)]['importance'].mean()

    name_list = ['mutations', 'other', 'copy_num', 'dnam', 'ex']
    num_list = [mu, ot, cn, dn, ex]
    print(num_list)
    all_sum = sum(num_list)
    print(all_sum)
    for i in range(len(num_list)):
        num_list[i] = num_list[i] / all_sum

    plt.bar(range(len(num_list)), num_list, tick_label=name_list,  width=0.4, lw=0.5, alpha=0.9,)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(cancer)
    plt.savefig(r'../data/img_data/%s.png' % (cancer))
    plt.show()
    plt.cla()
    print(cancer, mu, ot, cn, dn,)

    print(num_list[0], num_list[1], num_list[2], num_list[3], num_list[4])
    save_cancers = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
               'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV',
               'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC',
               'UCS', 'UVM']

    if cancer == 'Pan cancer features importances':
        out_names = ['mutations', 'others', 'copy_number', 'gene_expression', 'methylation']
        out_values = [num_list[0], num_list[1], num_list[2], num_list[4], num_list[3]]
        df_out = pd.DataFrame({'feature': out_names, 'importance': out_values})
        df_out.to_csv('../data/new_results/pancan_feature_importance.csv', index=False)
    elif cancer in save_cancers:
        out_names = ['mutations', 'others', 'copy_number', 'gene_expression', 'methylation']
        out_values = [num_list[0], num_list[1], num_list[2], num_list[4], num_list[3]]
        df_out = pd.DataFrame({'feature': out_names, 'importance': out_values})
        df_out.to_csv(f'../data/new_results/{cancer}_feature_importance.csv', index=False)

    return num_list[0], num_list[1], num_list[2], num_list[3], num_list[4]


pan_cancer()
each_cancer()



def fea_import_cal_2(result_path, features_path, cancer,):
    reslt = pd.read_csv(result_path)
    all_fea = pd.read_csv(features_path)
    reslt.rename(columns={'gene': 'Hugo_Symbol', 'qvalue': 'q'}, inplace=True)

    data = pd.merge(reslt, all_fea, on='Hugo_Symbol')
    df = data

    list = df.columns.values.tolist()
    l = ['Hugo_Symbol', 'class', 'Score', 'p', 'q']
    for i in l:
        list.remove(i)
    ss = StandardScaler()
    df[list] = ss.fit_transform(df[list])
    targets = df['Score'].values
    features = df.drop(['Hugo_Symbol', 'class', 'Score', 'p', 'q'], axis=1).values

    feat_labels = list
    model = RandomForestClassifier()
    model.fit(features.astype(int), targets.astype(int))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    for f in range(10):

        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
        if feat_labels[indices[f]] in value_dict:
            imp = value_dict[feat_labels[indices[f]]] + importances[indices[f]]
            value_dict[feat_labels[indices[f]]] = imp
        else:
            value_dict[feat_labels[indices[f]]] = importances[indices[f]]

        if feat_labels[indices[f]] in iter_dict:
            imp = iter_dict[feat_labels[indices[f]]] + 1
            iter_dict[feat_labels[indices[f]]] = imp
        else:
            iter_dict[feat_labels[indices[f]]] = 1
        df_insert = pd.DataFrame({'fea': feat_labels[indices[f]], 'import': importances[indices[f]]}, index=[0])

        global df_fea
        df_fea = df_fea.append(df_insert, ignore_index=True)



each_cancer_top()
lis = sorted(value_dict.items(), key=lambda d:d[1], reverse=True)
lis_1 = sorted(iter_dict.items(), key=lambda d:d[1], reverse=True)
print(lis)
print(lis_1)

df_p = df_fea.groupby(['fea']).size().reset_index(name='num')
df_p.to_csv(r'../data/new_results/new_p.csv', index=False)
print(df_p)
df_1 = df_fea.groupby(['fea'])['import'].mean()
df_1.to_csv(r'../data/new_results/new_p_1.csv')
print(df_1)



def import_features():
    df = pd.read_csv(r'../data/all_ori_fea.csv', sep=',')
    score = pd.read_csv(r'../data/data/cell_2018_other_method_spe_cancer/Our/PANCAN.csv')
    df = pd.merge(df, score, on='Hugo_Symbol')
    list = df.columns.values.tolist()
    l = ['Hugo_Symbol', 'class', 'Score', 'p', 'q']
    for i in l:
        list.remove(i)
    ss = StandardScaler()
    df[list] = ss.fit_transform(df[list])
    targets = df['Score'].values
    features = df.drop(['Hugo_Symbol', 'class', 'Score',  'p', 'q'], axis=1).values

    feat_labels = list
    model = RandomForestClassifier()
    model.fit(features.astype('int'), targets.astype('int'))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(features.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


import_features()

