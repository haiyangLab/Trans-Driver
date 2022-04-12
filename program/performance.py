import pandas as pd
import os
from collections import Counter
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from collections import Counter
import warnings
from scipy.stats import fisher_exact
import sys
import math
warnings.simplefilter(action="ignore", category=FutureWarning)


methods = ['Trans-Driver', '2020plus', 'MuSiC', 'CompositeDriver', 'CHASM', 'e-Driver', 'OncodriveCLUST'
           'ActiveDriver']


def fisher_ex(a, b, c, d):
    _, pvalue = fisher_exact([[a, b], [c, d]], 'greater')
    if pvalue < 1e-256:
        pvalue = 1e-256
    p1 = -math.log10(pvalue)
    return p1


def cgc_fisher(con_cgc, remain_pr_hg, cgc_data, df_active):
    pr_cgc = [item for item in con_cgc if item in cgc_data]
    remain_pr_cgc = [item for item in remain_pr_hg if item in cgc_data]
    len_pr_cgc = len(pr_cgc)
    len_pr_not_cgc = len(df_active) - len(pr_cgc)
    len_remain_cgc = len(remain_pr_cgc)
    number_cancer_1 = 20000 - len(df_active) - len_remain_cgc
    p = fisher_ex(len_pr_cgc, len_pr_not_cgc, len_remain_cgc, number_cancer_1)
    return p


def tcga_fisher(con_cgc, remain_pr_hg, tcga_data, df_active):
    pr_tcga = [item for item in con_cgc if item in tcga_data]
    remain_pr_tcga = [item for item in remain_pr_hg if item in tcga_data]
    len_pr_tcga = len(pr_tcga)
    len_pr_not_tcga = len(df_active) - len(pr_tcga)
    len_remain_tcga = len(remain_pr_tcga)
    number_cancer_1 = 20000 - len(df_active) - len_remain_tcga
    p = fisher_ex(len_pr_tcga, len_pr_not_tcga, len_remain_tcga, number_cancer_1)
    return p


def plot_cgc_overlap(cgc_overlap_df, list_name='CGC', custom_order=None):
    """Create a bar plot for the fraction overlap with the cancer gene census (CGC).

    Parameters
    ----------
    cgc_overlap_df : pd.DataFrame
        Dataframe containing method names as an index and columns for '# CGC' and
        'Fraction overlap w/ CGC'
    custom_order : list or None
        Order in which the methods will appear on the bar plot
    """

    # Function to label bars
    def autolabel(rects):
        # attach some text labels
        for ii, rect in enumerate(rects):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2., height + .005, '%s' % (name[ii]),
                     ha='center', va='bottom', size=12)

    # order methods if no order given
    if custom_order is None:
        custom_order = cgc_overlap_df.sort_values('Fraction overlap w/ {0}'.format(list_name)).index.tolist()

    # make barplot
    name = cgc_overlap_df['# ' + list_name].tolist()
    # name = ['2020plus', 'ActiveDriver', 'CompositeDriver', 'HotMAPS', 'MuSiC', 'MutSig2CV', 'OncodriveCLUST', 'Our']
    with sns.axes_style('ticks'), sns.plotting_context('talk', font_scale=0.6):
        ax = sns.barplot(cgc_overlap_df.index,
                         cgc_overlap_df['Fraction overlap w/ {0}'.format(list_name)],
                         order=custom_order, color='black')

        # label each bar
        autolabel(ax.patches)

        # fiddle with formatting
        ax.set_xlabel('Methods')
        ax.set_ylabel('Fraction of predicted drivers\nfound in ' + list_name)
        sns.despine()
        plt.xticks(rotation=15, ha='right', va='top')
        plt.gcf().set_size_inches(6, 6)
        # change tick padding
        plt.gca().tick_params(axis='x', which='major', pad=0)

    # format layout
    plt.tight_layout()
    plt.show()


def plot_cgc_and_tcga_fisher():
    path = r'../data/cell_2018'
    cgc_data = pd.read_csv(r'../data/cgc_somatic.csv', sep='\t')
    cgc_data = cgc_data['Gene'].values.tolist()
    tcga_data = pd.read_csv(r'../data/cell_2018/PANCAN.csv')
    tcga_data = tcga_data['Gene'].values.tolist()
    dirs = os.listdir(path=path)
    q = 0.05
    list_all = []
    d = {}
    num_cgc_dict = {}
    num_signif_dict = {}
    fish_res = {}
    tcga_fish_res = {}
    df_pr = pd.DataFrame(columns=('methods', '# CGC', '# significant', 'Fraction overlap w/ CGC'))
    df_all = pd.DataFrame(columns=['gene'])
    for file in dirs:
        if file in methods:
            new_file = path + '\\' + file + '\\' + 'PANCAN.csv'
            # print(new_file)
            if file == 'ActiveDriver':
                print('ActiveDriver')
                q = 0.0001  # 0.0001
                df_active_1 = pd.read_csv(new_file, sep=',')
                df_active = df_active_1.loc[df_active_1['qvalue'] < q]
                df_active = df_active.drop_duplicates(['gene'])
                df_active = df_active['gene']
                con_cgc = df_active.values.tolist()
                con = [item for item in con_cgc if item in cgc_data]
                remain_pr_data = df_active_1.loc[df_active_1['qvalue'] >= q]
                remain_pr_data = remain_pr_data.drop_duplicates(['gene'])

                remain_pr_hg = remain_pr_data['gene'].values.tolist()

                fisher_cgc = cgc_fisher(con_cgc, remain_pr_hg, cgc_data, df_active)
                fisher_tcga = tcga_fisher(con_cgc, remain_pr_hg, tcga_data, df_active)


                fish_res['ActiveDriver'] = fisher_cgc
                tcga_fish_res['ActiveDriver'] = fisher_tcga
                # print('CGC fisher = {}'.format(fisher_cgc))
                # print('TCGA fisher = {}'.format(fisher_tcga))
                # print(len(con))
                df = df_active
                d['ActiveDriver'] = len(con)
                num_signif_dict['ActiveDriver'] = len(con_cgc)
                new = pd.DataFrame({'methods': 'ActiveDriver',
                                    '# CGC': len(con),
                                    '# significant': len(df_active),
                                    'Fraction overlap w/ CGC': len(con) / len(df_active)}, index=[len(df_pr)])
                df_pr = df_pr.append(new, ignore_index=True)
            elif file == 'OncodriveCLUST':
                print('OncodriveCLUST')
                q = 0.05  # 0.05
                df_clust_1 = pd.read_csv(new_file, sep=',')
                df_clust = df_clust_1.loc[df_clust_1['qvalue'] < q]
                df_clust = df_clust.drop_duplicates(['gene'])
                df_clust = df_clust['gene']
                con_cgc = df_clust.values.tolist()
                con = [item for item in con_cgc if item in cgc_data]

                remain_pr_data = df_clust_1.loc[df_clust_1['qvalue'] >= q]
                remain_pr_data = remain_pr_data.drop_duplicates(['gene'])

                remain_pr_hg = remain_pr_data['gene'].values.tolist()

                fisher_cgc = cgc_fisher(con_cgc, remain_pr_hg, cgc_data, df_clust)
                fisher_tcga = tcga_fisher(con_cgc, remain_pr_hg, tcga_data, df_clust)

                fish_res['OncodriveCLUST'] = fisher_cgc
                tcga_fish_res['OncodriveCLUST'] = fisher_tcga
                # print('CGC fisher = {}'.format(fisher_cgc))
                # print('TCGA fisher = {}'.format(fisher_tcga))
                # print(len(con))
                df = df_clust
                d['OncodriveCLUST'] = len(con)
                num_signif_dict['OncodriveCLUST'] = len(con_cgc)
                new = pd.DataFrame({'methods': 'OncodriveCLUST',
                                    '# CGC': len(con),
                                    '# significant': len(df_clust),
                                    'Fraction overlap w/ CGC': len(con) / len(df_clust)}, index=[len(df_pr)])
                df_pr = df_pr.append(new, ignore_index=True)

            elif file == '2020plus':
                print('2020plus')
                new_file = path + '\\' + file + '\\' + 'PANCAN_all.csv'
                q = 0.05  # 0.05
                df_20_1 = pd.read_csv(new_file, sep=',')
                # df_20_1 = df_20_1.loc[df_20_1['info'] == 'TYPE=driver']
                df_20 = df_20_1.loc[df_20_1['qvalue'] < q]
                df_20 = df_20.drop_duplicates(['gene'])
                df_20 = df_20['gene']
                con_cgc = df_20.values.tolist()
                con = [item for item in con_cgc if item in cgc_data]
                # con_cgc = [item for item in df_20_list if item in cgc_data]

                remain_pr_data = df_20_1.loc[df_20_1['qvalue'] >= q]
                remain_pr_data = remain_pr_data.drop_duplicates(['gene'])

                remain_pr_hg = remain_pr_data['gene'].values.tolist()
                fisher_cgc = cgc_fisher(con_cgc, remain_pr_hg, cgc_data, df_20)
                fisher_tcga = tcga_fisher(con_cgc, remain_pr_hg, tcga_data, df_20)
                fish_res['2020plus'] = fisher_cgc
                tcga_fish_res['2020plus'] = fisher_tcga
                # print('CGC fisher = {}'.format(fisher_cgc))
                # print('TCGA fisher = {}'.format(fisher_tcga))
                # print(len(con_cgc))
                df = df_20
                d['2020plus'] = len(con)
                num_signif_dict['2020plus'] = len(con_cgc)
                new = pd.DataFrame({'methods': '2020plus',
                                    '# CGC': len(con_cgc),
                                    '# significant': len(con_cgc),
                                    'Fraction overlap w/ CGC': len(con) / len(con_cgc)}, index=[len(df_pr)])
                df_pr = df_pr.append(new, ignore_index=True)

            elif file == 'CompositeDriver':
                q = 0.05  # 0.05
                print('CompositeDriver')
                df_cd_1 = pd.read_csv(new_file, sep=',')
                df_cd = df_cd_1.loc[df_cd_1['qvalue'] < q]
                df_cd = df_cd.drop_duplicates(['gene'])
                df_cd = df_cd['gene']
                con_cgc = df_cd.values.tolist()
                con = [item for item in con_cgc if item in cgc_data]

                remain_pr_data = df_cd_1.loc[df_cd_1['qvalue'] >= q]
                remain_pr_data = remain_pr_data.drop_duplicates(['gene'])

                remain_pr_hg = remain_pr_data['gene'].values.tolist()

                fisher_cgc = cgc_fisher(con_cgc, remain_pr_hg, cgc_data, df_cd)
                fisher_tcga = tcga_fisher(con_cgc, remain_pr_hg, tcga_data, df_cd)
                fish_res['CompositeDriver'] = fisher_cgc
                tcga_fish_res['CompositeDriver'] = fisher_tcga
                df = df_cd
                d['CompositeDriver'] = len(con)
                num_signif_dict['CompositeDriver'] = len(con_cgc)
                new = pd.DataFrame({'methods': 'CompositeDriver',
                                    '# CGC': len(con),
                                    '# significant': len(con_cgc),
                                    'Fraction overlap w/ CGC': len(con) / len(con_cgc)}, index=[len(df_pr)])
                df_pr = df_pr.append(new, ignore_index=True)
            elif file == 'MuSiC':
                print('MuSiC')
                q = 0.1  # 0.01
                df_hot_1 = pd.read_csv(new_file, sep=',')
                df_hot = df_hot_1.loc[df_hot_1['qvalue'] < q]
                df_hot = df_hot.drop_duplicates(['gene'])
                df_hot = df_hot['gene']
                con_cgc = df_hot.values.tolist()
                con = [item for item in con_cgc if item in cgc_data]

                remain_pr_data = df_hot_1.loc[df_hot_1['qvalue'] >= q]
                remain_pr_data = remain_pr_data.drop_duplicates(['gene'])

                remain_pr_hg = remain_pr_data['gene'].values.tolist()

                fisher_cgc = cgc_fisher(con_cgc, remain_pr_hg, cgc_data, df_hot)
                fisher_tcga = tcga_fisher(con_cgc, remain_pr_hg, tcga_data, df_hot)
                fish_res['MuSiC'] = fisher_cgc
                tcga_fish_res['MuSiC'] = fisher_tcga
                df = df_hot
                d['MuSiC'] = len(con)
                num_signif_dict['MuSiC'] = len(con_cgc)
                new = pd.DataFrame({'methods': 'MuSiC',
                                    '# CGC': len(con),
                                    '# significant': len(df_hot),
                                    'Fraction overlap w/ CGC': len(con) / len(df_hot)}, index=[len(df_pr)])
                df_pr = df_pr.append(new, ignore_index=True)

            elif file == 'e-Driver':
                print('e-Driver')
                q = 0.05
                df_e3d_1 = pd.read_csv(new_file, sep=',')
                df_e3d = df_e3d_1.loc[df_e3d_1['qvalue'] < q]
                df_e3d = df_e3d.drop_duplicates(['gene'])
                df_e3d = df_e3d['gene']
                con_cgc = df_e3d.values.tolist()
                con = [item for item in con_cgc if item in cgc_data]

                remain_pr_data = df_e3d_1.loc[df_e3d_1['qvalue'] >= q]
                remain_pr_data = remain_pr_data.drop_duplicates(['gene'])

                remain_pr_hg = remain_pr_data['gene'].values.tolist()

                fisher_cgc = cgc_fisher(con_cgc, remain_pr_hg, cgc_data, df_e3d)
                fisher_tcga = tcga_fisher(con_cgc, remain_pr_hg, tcga_data, df_e3d)
                fish_res['e-Driver'] = fisher_cgc
                tcga_fish_res['e-Driver'] = fisher_tcga
                # print('CGC fisher = {}'.format(fisher_cgc))
                # print('TCGA fisher = {}'.format(fisher_tcga))
                # print(len(con))
                df = df_e3d
                d['e-Driver'] = len(con)
                num_signif_dict['e-Driver'] = len(con_cgc)
                new = pd.DataFrame({'methods': 'e-Driver',
                                    '# CGC': len(con),
                                    '# significant': len(con_cgc),
                                    'Fraction overlap w/ CGC': len(con) / len(con_cgc)}, index=[len(df_pr)])
                df_pr = df_pr.append(new, ignore_index=True)

            elif file == 'CHASM':
                print('CHASM')
                q = 0.1
                df_chasm_1 = pd.read_csv(new_file, sep=',')
                df_chasm = df_chasm_1.loc[df_chasm_1['qvalue'] < q]
                df_chasm = df_chasm.drop_duplicates(['gene'])
                df_chasm = df_chasm['gene']
                con_cgc = df_chasm.values.tolist()
                con = [item for item in con_cgc if item in cgc_data]

                remain_pr_data = df_chasm_1.loc[df_chasm_1['qvalue'] >= q]
                remain_pr_data = remain_pr_data.drop_duplicates(['gene'])

                remain_pr_hg = remain_pr_data['gene'].values.tolist()

                fisher_cgc = cgc_fisher(con_cgc, remain_pr_hg, cgc_data, df_chasm)
                fisher_tcga = tcga_fisher(con_cgc, remain_pr_hg, tcga_data, df_chasm)
                fish_res['CHASM'] = fisher_cgc
                tcga_fish_res['CHASM'] = fisher_tcga
                # print('CGC fisher = {}'.format(fisher_cgc))
                # print('TCGA fisher = {}'.format(fisher_tcga))
                # print(len(con))
                df = df_chasm
                d['CHASM'] = len(con)
                num_signif_dict['CHASM'] = len(con_cgc)
                new = pd.DataFrame({'methods': 'CHASM',
                                    '# CGC': len(con),
                                    '# significant': len(con_cgc),
                                    'Fraction overlap w/ CGC': len(con) / len(con_cgc)}, index=[len(df_pr)])
                df_pr = df_pr.append(new, ignore_index=True)
            else:
                print('Trans-Driver')
                q = 0.05
                df_our_1 = pd.read_csv(new_file, sep=',')
                df_our = df_our_1.loc[df_our_1['qvalue'] < q]
                df_our = df_our.drop_duplicates(['gene'])
                df_our = df_our['gene']
                con_cgc = df_our.values.tolist()
                con = [item for item in con_cgc if item in cgc_data]

                remain_pr_data = df_our_1.loc[df_our_1['qvalue'] >= q]
                remain_pr_data = remain_pr_data.drop_duplicates(['gene'])

                remain_pr_hg = remain_pr_data['gene'].values.tolist()
                fisher_cgc = cgc_fisher(con_cgc, remain_pr_hg, cgc_data, df_our)
                fisher_tcga = tcga_fisher(con_cgc, remain_pr_hg, tcga_data, df_our)
                fish_res['Trans-Driver'] = fisher_cgc
                tcga_fish_res['Trans-Driver'] = fisher_tcga
                # print('CGC fisher = {}'.format(fisher_cgc))
                # print('TCGA fisher = {}'.format(fisher_tcga))
                # print(len(con))
                df = df_our
                d['Trans-Driver'] = len(con)
                num_signif_dict['Trans-Driver'] = len(con_cgc)
                new = pd.DataFrame({'methods': 'Trans-Driver',
                                    '# CGC': len(con),
                                    '# significant': len(con_cgc),
                                    'Fraction overlap w/ CGC': len(con) / len(con_cgc)}, index=[len(df_pr)])
                df_pr = df_pr.append(new, ignore_index=True)
            list_all.extend(df.values)

    overlap_df = pd.DataFrame({'# CGC': pd.Series(d),
                               '# significant': pd.Series(num_signif_dict)})
    overlap_df['Fraction overlap w/ CGC'] = overlap_df['# CGC'].astype(float) / overlap_df['# significant']
    print(overlap_df)
    fish_res = sorted(fish_res.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    me = []
    f_value = []
    for i in fish_res:
        me.append(i[0])
        f_value.append(i[1])
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.bar(me, f_value, width=0.8, lw=0.8, alpha=0.9, )
    plt.xticks(rotation=80)
    plt.title('Pancan cancer cgc fisher_exact')
    fig.tight_layout()
    plt.show()
    order = ['2020plus', 'Trans-Driver', 'CompositeDriver', 'ActiveDriver', 'e-Driver', 'CHASM',
              'MuSiC', 'OncodriveCLUST']
    overlap_df['st'] = overlap_df.index
    overlap_df['st'] = overlap_df['st'].astype('category')
    overlap_df['st'].cat.reorder_categories(order, inplace=True)
    overlap_df.sort_values('st', inplace=True)
    plot_cgc_overlap(overlap_df, custom_order=order)

    tcga_fish_res = sorted(tcga_fish_res.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    me = []
    f_value = []
    for i in tcga_fish_res:
        me.append(i[0])
        f_value.append(i[1])
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.bar(me, f_value, width=0.8, lw=0.8, alpha=0.9, )
    plt.xticks(rotation=80)
    plt.title('Pancan cancer tcga fisher_exact')
    fig.tight_layout()
    plt.show()
    order = ['2020plus', 'Trans-Driver', 'CompositeDriver', 'ActiveDriver', 'e-Driver', 'CHASM',
              'MuSiC', 'OncodriveCLUST']
    overlap_df['st'] = overlap_df.index
    overlap_df['st'] = overlap_df['st'].astype('category')
    overlap_df['st'].cat.reorder_categories(order, inplace=True)
    overlap_df.sort_values('st', inplace=True)
    plot_cgc_overlap(overlap_df, custom_order=order)

    path = r'../results/pancancer_fisher.csv'
    pd_data = pd.DataFrame(columns=('DataSet', 'Method', 'Fisher_value'))
    for i in fish_res:
        a = {'DataSet': 'CGC', 'Method': i[0], 'Fisher_value': i[1]}
        pd_data = pd_data.append(a, ignore_index=True)
    for i in tcga_fish_res:
        a = {'DataSet': 'TCGA', 'Method': i[0], 'Fisher_value': i[1]}
        pd_data = pd_data.append(a, ignore_index=True)
    pd_data.to_csv(path, sep=',', index=False)


def plot_con_with_method():
    path = r'../data/cell_2018'
    cgc_data = pd.read_csv(r'../data/cgc_somatic.csv', sep='\t')
    tcga_data = pd.read_csv(r'../data/cell_2018/PANCAN.csv')
    tcga_data = tcga_data['Gene'].values.tolist()
    cgc_data = cgc_data['Gene'].values.tolist()
    dirs = os.listdir(path=path)
    q = 0.05
    list_all = []
    d = {}

    df_all = pd.DataFrame(columns=['gene'])
    for file in dirs:
        if file in methods:
            new_file = path + '\\' + file + '\\' + 'PANCAN.csv'
            # print(new_file)
            if file == 'ActiveDriver':
                q = 0.0001  # 0.0001
                df_active = pd.read_csv(new_file, sep=',')
                df_active = df_active.loc[df_active['qvalue'] < q]
                df_active = df_active.drop_duplicates(['gene'])
                df_active = df_active['gene']
                con_cgc = df_active.values.tolist()
                con = [item for item in con_cgc if item in cgc_data]
                tcga_con = [item for item in con_cgc if item in tcga_data]
                print('ActiveDriver')
                print("CGC fisher", len(con))
                print("TCGA fisher", len(tcga_con))
                df = df_active
                d['ActiveDriver'] = len(con)
            elif file == 'OncodriveCLUST':
                q = 0.05  # 0.05
                df_clust = pd.read_csv(new_file, sep=',')
                df_clust = df_clust.loc[df_clust['qvalue'] < q]
                df_clust = df_clust.drop_duplicates(['gene'])
                df_clust = df_clust['gene']
                con_cgc = df_clust.values.tolist()
                con = [item for item in con_cgc if item in cgc_data]
                print('OncodriveCLUST')
                print(len(con))
                df = df_clust
                d['OncodriveCLUST'] = len(con)
            elif file == '2020plus':
                q = 0.05  # 0.05
                df_20 = pd.read_csv(new_file, sep=',')
                df_20 = df_20.loc[df_20['qvalue'] < q]
                df_20 = df_20.drop_duplicates(['gene'])
                df_20 = df_20['gene']
                df_20_list = df_20.values.tolist()
                # print(len(df_20_list))
                # con = [item for item in cgc_data if item in df_20_list]
                # # print(len(con))
                # con_cgc = df_20.values.tolist()
                con = [item for item in df_20_list if item in cgc_data]
                print('2020plus')
                print(len(con))
                df = df_20
                d['2020plus'] = len(con)
            elif file == 'CompositeDriver':
                q = 0.05  # 0.05
                df_cd = pd.read_csv(new_file, sep=',')
                df_cd = df_cd.loc[df_cd['qvalue'] < q]
                df_cd = df_cd.drop_duplicates(['gene'])
                df_cd = df_cd['gene']
                con_cgc = df_cd.values.tolist()
                con = [item for item in con_cgc if item in cgc_data]
                print('CompositeDriver')
                print(len(con))
                df = df_cd
                d['CompositeDriver'] = len(con)
            elif file == 'e-Driver':
                q = 0.1
                df_e3d = pd.read_csv(new_file, sep=',')
                df_e3d = df_e3d.loc[df_e3d['qvalue'] < q]
                df_e3d = df_e3d.drop_duplicates(['gene'])
                df_e3d = df_e3d['gene']
                con_cgc = df_e3d.values.tolist()
                con = [item for item in con_cgc if item in cgc_data]
                print('e-Driver')
                print(len(con))
                df = df_e3d
                d['e-Driver'] = len(con)
            elif file == 'MuSiC':
                q = 0.1
                df_ms = pd.read_csv(new_file, sep=',')
                df_ms = df_ms.loc[df_ms['qvalue'] < q]
                df_ms = df_ms.drop_duplicates(['gene'])
                df_ms = df_ms['gene']
                con_cgc = df_ms.values.tolist()
                con = [item for item in con_cgc if item in cgc_data]
                print('MuSiC')
                print(len(con))
                df = df_ms
                d['MuSiC'] = len(con)
            elif file == 'CHASM':
                q = 0.1
                df_chasm = pd.read_csv(new_file, sep=',')
                df_chasm = df_chasm.loc[df_chasm['qvalue'] < q]
                df_chasm = df_chasm.drop_duplicates(['gene'])
                df_chasm = df_chasm['gene']
                con_cgc = df_chasm.values.tolist()
                con = [item for item in con_cgc if item in cgc_data]
                print('CHASM')
                print(len(con))
                df = df_chasm
                d['CHASM'] = len(con)
            else:
                q = 0.05
                df_our = pd.read_csv(new_file, sep=',')
                df_our = df_our.loc[df_our['qvalue'] <= q]
                df_our = df_our.drop_duplicates(['gene'])
                df_our = df_our['gene']
                con_cgc = df_our.values.tolist()
                con = [item for item in con_cgc if item in cgc_data]
                print('Trans-Driver')
                print(len(con))
                df = df_our
                d['Trans-Driver'] = len(con)
            list_all.extend(df.values)
    result = Counter(list_all)
    df = pd.DataFrame.from_dict(result, orient='index').reset_index()
    df.columns = ['key', 'cnts']
    df = df.groupby(['cnts'])

    li = []
    for name, group in df:
        li.append(group['key'].values.tolist())

    df = pd.concat([pd.DataFrame(x) for x in li], axis=1)

    df.columns = ['1', '2', '3', '4', '5', '6', '7', '8']
    df = df.fillna(0)
    num_1 = df['1'].values.tolist()
    num_2 = df['2'].values.tolist()
    num_3 = df['3'].values.tolist()
    num_4 = df['4'].values.tolist()
    num_5 = df['5'].values.tolist()
    num_6 = df['6'].values.tolist()
    num_7 = df['7'].values.tolist()
    num_8 = df['8'].values.tolist()

    num_1 = [x for x in num_1 if x != 0]
    num_2 = [x for x in num_2 if x != 0]
    num_3 = [x for x in num_3 if x != 0]
    num_4 = [x for x in num_4 if x != 0]
    num_5 = [x for x in num_5 if x != 0]
    num_6 = [x for x in num_6 if x != 0]
    num_7 = [x for x in num_7 if x != 0]
    num_8 = [x for x in num_8 if x != 0]

    gene_path = r'../results/multiple_1.csv'
    gene_df = pd.read_csv(gene_path, sep=',')
    gene = gene_df.loc[gene_df['q'] <= 0.1]

    df_active = df_active.values.tolist()
    df_e3d = df_e3d.values.tolist()
    df_20 = df_20.values.tolist()
    df_cd = df_cd.values.tolist()
    df_our = df_our.values.tolist()
    df_ms = df_ms.values.tolist()
    df_chasm = df_chasm.values.tolist()
    df_clust = df_clust.values.tolist()

    def sub_cal(df, num):
        cnt = 0
        for i in num:
            if i in df:
                cnt += 1
        return cnt

    def cal(df):
        cnt1 = sub_cal(df, num_1)
        cnt2 = sub_cal(df, num_2)
        cnt3 = sub_cal(df, num_3)
        cnt4 = sub_cal(df, num_4)
        cnt5 = sub_cal(df, num_5)
        cnt6 = sub_cal(df, num_6)
        cnt7 = sub_cal(df, num_7)
        cnt8 = sub_cal(df, num_8)

        one = cnt1
        two = cnt2
        three = cnt3
        four = cnt4 + cnt5 + cnt6 + cnt7 + cnt8
        all = len(df)
        return [all, one, two, three, four]

    ac = cal(df_active)
    twozero_plus = cal(df_20)
    cd = cal(df_cd)
    our = cal(df_our)
    e3d = cal(df_e3d)
    ms = cal(df_ms)
    chasm = cal(df_chasm)
    clust = cal(df_clust)
    df_all = pd.DataFrame(columns=['Total', 'predicted by 1 method', 'two methods', 'three methods',
                                   'at least four methods'],

                          data=[cd, twozero_plus, ac, our, e3d, clust, chasm, ms])

    df_all['Methods'] = ['CompositeDriver', '2020plus', 'ActiveDriver',
                         'Trans-Driver', 'e-Driver', 'OncodriveCLUST', 'CHASM', 'MuSiC']
    order = ['Methods', 'Total', 'predicted by 1 method', 'two methods', 'three methods',
             'at least four methods']
    df_all = df_all[order]
    order = ['CompositeDriver', '2020plus', 'ActiveDriver',
             'Trans-Driver', 'e-Driver', 'OncodriveCLUST', 'CHASM', 'MuSiC']
    df_all.to_csv(r'../results/consistency.csv', sep=',', index=False)
    df_all.to_csv(r'../results/each_method_consistency.csv', sep=',', index=False)
    plot_method_overlap(df_all.copy(), custom_order=order)
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.bar(d.keys(), d.values(), width=0.8, lw=0.8, alpha=0.9, )
    plt.xticks(rotation=80)
    fig.tight_layout()
    fig.savefig(r"../results/bar1.png")
    plt.show()


def plot_method_overlap(overlap_df, custom_order=None):
    """Plot the fraction overlap of predicted driver genes with other methods.
    """
    # calculate the fractions
    overlap_df['predicted by 1 method'] = 1.0
    mycols = ['two methods', 'three methods', 'at least four methods']
    overlap_df.loc[:, mycols] = overlap_df.loc[:, mycols].astype(float).div(overlap_df['Total'], axis=0)
    overlap_df['three methods'] = overlap_df['three methods'] + overlap_df['at least four methods']
    overlap_df['two methods'] = overlap_df['two methods'] + overlap_df['three methods']

    # order methods in increasing order
    if custom_order is None:
        custom_order = overlap_df.sort_values('two methods')['Method'].tolist()

    # colors = ['white'] + sns.cubehelix_palette(3)[:3]
    colors = ['red', 'green', 'blue', 'orange']
    for i, col in enumerate(['predicted by 1 method',
                             'two methods', 'three methods', 'at least four methods']):
        with sns.axes_style('ticks', rc={'xtick.major.pad': -1.0}), sns.plotting_context('talk', font_scale=1):
            sns.barplot('Methods', col, data=overlap_df,
                        color=colors[i], label=col, order=custom_order, )

            # Finishing touches
            lgd = plt.legend(bbox_to_anchor=(1, .75), loc='upper left',
                             ncol=1, )
            plt.ylim((0, 1))
            plt.ylabel('Fraction of predicted drivers')
            plt.gca().set_xticklabels(custom_order, rotation=30, ha='right')
            fig = plt.gcf()
            fig.set_size_inches(11, 8)

            # set bar width to 1
            for container in plt.gca().containers:
                plt.setp(container, width=1)
            # remove extra ticks
            plt.gca().get_xaxis().tick_bottom()
            plt.gca().get_yaxis().tick_left()

            # change tick padding
            plt.gca().tick_params(axis='x', which='major', pad=0)

    plt.tight_layout()
    plt.show()


def multi_2_monoomics():
    cgc_data = pd.read_csv(r'../data/cgc_somatic.csv', sep='\t')
    cgc_data = cgc_data['Gene'].values.tolist()
    tcga_data = pd.read_csv(r'../data/cell_2018/PANCAN.csv')
    tcga_data = tcga_data['Gene'].values.tolist()
    only_mutation_path = r'../data/only_mut_PANCAN.csv'
    q = 0.05
    only_mutation_1 = pd.read_csv(only_mutation_path, sep=',')
    only_mutation = only_mutation_1.loc[only_mutation_1['qvalue'] < q]
    only_mutation = only_mutation.drop_duplicates(['gene'])
    only_mutation = only_mutation['gene']
    con_cgc = only_mutation.values.tolist()
    con = [item for item in con_cgc if item in cgc_data]

    remain_pr_data = only_mutation_1.loc[only_mutation_1['qvalue'] >= q]
    remain_pr_data = remain_pr_data.drop_duplicates(['gene'])

    remain_pr_hg = remain_pr_data['gene'].values.tolist()

    fisher_cgc = cgc_fisher(con_cgc, remain_pr_hg, cgc_data, only_mutation)
    fisher_tcga = tcga_fisher(con_cgc, remain_pr_hg, tcga_data, only_mutation)
    print('fisher_cgc:', fisher_cgc)
    print('fisher_tcga:', fisher_tcga)


if __name__ == '__main__':
    plot_cgc_and_tcga_fisher()
    plot_con_with_method()
    multi_2_monoomics()




