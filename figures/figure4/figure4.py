import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import copy
import os

from tms_tg import TMSTG, EPOCHISOLATORS, hem_CoilDir_selector
from itertools import zip_longest, product
from figures.helper_figs import (gb_addinfo, gp_extractor, delay_agg, compute_excludeIdx)


def plot(tms, activeNeus, xlim=None):
    oldSelParams = copy.deepcopy(tms.analysis_params['selectionParams'])

    # select TGcut epochs
    tgCutCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])),
                                     'TGcut': ('Lv1', 'Lv2', 'Lv3', 'Rv1', 'Rv2', 'Rv3')}}
    tms.analysis_params = tgCutCond
    selectBlocksinfo, selectBlocksinfoIdx = tms.filter_blocks

    # select TGcut animals
    tgCutAnimals = {'selectionParams':
                        {'Epoch': dict(zip_longest(EPOCHISOLATORS,
                                                   [tuple(selectBlocksinfo.index.get_level_values('Animal').unique()),
                                                    None]))}}
    tms.analysis_params = tgCutAnimals
    selectBlocksinfo, selectBlocksinfoIdx = tms.filter_blocks

    # exclude epochs
    otherPostConds = [{'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), postKind: 'Post'}} for
                      postKind in {'TG-Injection ', 'Skin-Injection', 'TGOrifice'} & set(tms.blocksinfo.columns)]
    excludeIdx = compute_excludeIdx(selectBlocksinfo, otherPostConds)
    epochIndices = selectBlocksinfo.index[~excludeIdx].unique()
    if not any(epochIndices):
        print(f'Cannot plot TGOrifice as the associated data is missing in this group of animals')
        return

    # compute peristimulus firing rate if not already done and save it in blocksinfo for later
    for epochIndex in epochIndices:
        tms.analysis_params = {'selectionParams': {'Epoch': {x: y if y != 'none' else None
                                                             for x, y in zip(epochIndices.names, epochIndex)}}}
        _, _ = tms.compute_firingrate(*tms.analysis_params['peristimParams']['smoothingParams'].values(),
                                      *tms.analysis_params['peristimParams']['timeWin'],
                                      tms.analysis_params['peristimParams']['trigger'])
    # reset analysis parameters
    tms.analysis_params = {'selectionParams': oldSelParams}

    # statistics
    animalNumsEpochNumsAndActiveNeuronNums = ((np.unique([item[0] for item in epochIndices]).size,
                                               len(epochIndices),
                                               [activeNeus[item].sum() for item in epochIndices]))

    df = tms.blocksinfo.loc[epochIndices, :]

    # convert CoilDir to a consistent nomenclature and set append column 'CHvsRH' with 'hemi' values\
    for hemi, coilDir in product(['same', 'opposite'], ['ML', 'LM']):
        bIdx = hem_CoilDir_selector(df, range(df.shape[0]), hemi, coilDir)
        df.loc[bIdx, 'CoilDir'] = coilDir
        df.loc[bIdx, 'CHvsRH'] = hemi

    # delay
    delay = df.groupby(by=df.index.names + ['CoilDir', 'CHvsRH', 'TGcut', 'CortexAbl', 'RecArea ']).apply(
        gp_extractor, delay_agg, activeNeus, 'TGcut', 'psActivity', None, tms)
    delay.reset_index(['CoilDir', 'CHvsRH', 'TGcut', 'CortexAbl', 'RecArea '], col_level=1, inplace=True)
    delay = delay.melt(id_vars=[item for item in delay.columns if item[0] == ''],
                       value_vars=[item for item in delay.columns if item[0] == 'MT'],
                       var_name=['colName', 'MT'], value_name='Delay (ms)')
    delay = pd.concat([delay,
                       delay.apply(lambda x: pd.Series(x[('', 'CHvsRH')] + '/' + x[('', 'CoilDir')],
                                                       index=['StimHem and CoilDir']), axis=1),
                       delay.apply(lambda x: pd.Series(x[('', 'CortexAbl')] + '/' + x[('', 'TGcut')],
                                                       index=['CortexAbl and TGcut']), axis=1)],
                      axis=1).drop(
        columns=[('', 'CoilDir'), ('', 'CHvsRH'), ('', 'TGcut'), ('', 'CortexAbl'), 'colName'])

    # plot delay
    plt.style.use('default')
    fig, axes = plt.subplots(1, 1,)
    swarmplot = sns.catplot(
        data=delay[(delay[('', 'RecArea ')] == 'MC') & (delay['MT'] > 0)],
        kind='swarm', x='CortexAbl and TGcut', y='Delay (ms)', hue='MT', col='StimHem and CoilDir',
        order=['No/No', 'LH/No', 'Both/No', 'LH/Lv1,Lv2,Lv3', 'Both/Lv1,Lv2,Lv3', 'LH/Rv1,Rv2,Rv3'],
        col_order=['same/ML', 'same/LM', 'opposite/ML', 'opposite/LM'],
        size=3, dodge=True, palette='deep', aspect=.5)
    for ax in swarmplot.axes.flat:
        ax.tick_params(axis='x', labelsize='small', labelrotation=45)
    swarmplot.figure.tight_layout()
    sns.move_legend(swarmplot, "upper right")
    plt.savefig('figure4_swarm.pdf', dpi='figure', format='pdf')
    swarmplot.figure.show()

    fig, axes = plt.subplots(1, 1, )
    boxplot = sns.catplot(
        data=delay[(delay[('', 'RecArea ')] == 'MC') & (delay['MT'] > 0)],
        kind='box', x='CortexAbl and TGcut', y='Delay (ms)', col='StimHem and CoilDir',
        order=['No/No', 'LH/No', 'Both/No', 'LH/Lv1,Lv2,Lv3', 'Both/Lv1,Lv2,Lv3', 'LH/Rv1,Rv2,Rv3'],
        col_order=['same/ML', 'same/LM', 'opposite/ML', 'opposite/LM'],
        aspect=0.5)
    for ax in boxplot.axes.flat:
        ax.tick_params(axis='x', labelsize='small', labelrotation=45)
    boxplot.figure.tight_layout()
    plt.savefig('figure4_box.pdf', dpi='figure', format='pdf')
    boxplot.figure.show()
    print('[(Number of Animals, Number of Epochs, Number of Neurons per epoch), ...]: ',
          animalNumsEpochNumsAndActiveNeuronNums)


if __name__ == '__main__':
    animalList = r'G:\Vishnu\data\TMSTG\animalList.xlsx'
    tms = TMSTG.load(animalList)
    # activeNeu = tms.stats_is_signf_active()
    if os.path.isfile('../../grandBlocksinfo'):
        grandBlocksinfo = pd.read_pickle('../../grandBlocksinfo')
        for col in {'psActivity', 'delay'} & set(grandBlocksinfo.columns):
            bIndicesInGBIndices = tuple(set(tms.blocksinfo.index.unique()) & set(grandBlocksinfo.index.unique()))
            for index in bIndicesInGBIndices:
                tms.blocksinfo.loc[index, col] = grandBlocksinfo.loc[index, col]
        tms.blocksinfo = tms.blocksinfo.where(pd.notnull(tms.blocksinfo), None)
        tms.filter_blocks = None

    plot(tms, pd.read_pickle("./activeNeu_TGcut"), xlim=[-20, 60])

    # save data after join and apply
    if os.path.isfile('../../grandBlocksinfo'):
        grandBlocksinfo = gb_addinfo(pd.read_pickle('../../grandBlocksinfo'), tms)
        grandBlocksinfo.to_pickle('../../grandBlocksinfo')