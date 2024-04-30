import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import copy
import os

from tms_tg import TMSTG, EPOCHISOLATORS
from itertools import zip_longest
from figures.helper_figs import (gb_addinfo, gp_extractor, pkFr_agg, delay_agg, compute_excludeIdx, violin_fill_false)


def plot(tms, activeNeus, xlim=None):
    oldSelParams = copy.deepcopy(tms.analysis_params['selectionParams'])
    postCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'TGOrifice': 'Post'}}
    otherPostConds = [{'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), postKind: 'Post'}} for
                      postKind in {'TG-Injection ', 'Skin-Injection'} & set(tms.blocksinfo.columns)]
    # + [{'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])),
    #                         'TGcut': ('Lv1', 'Lv2', 'Lv3', 'Rv1', 'Rv2', 'Rv3')}}])
    tms.analysis_params = postCond
    selectBlocksinfo, selectBlocksinfoIdx = tms.filter_blocks
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

    # peak firing rate
    pkFr = df.groupby(by=df.index.names).apply(gp_extractor, pkFr_agg, activeNeus, 'TGOrifice', 'psActivity',
                                               [(0, 10), (10, 20), (20, 30), (30, 40), (40, np.inf)], tms)
    pkFr = pkFr.melt(id_vars=['MT'], value_vars=['Pre', *np.setdiff1d(pkFr.columns, ['MT', 'Pre'])],
                     var_name='Time elapsed (min)', value_name='Peak firing rate (Hz)')

    # delay
    delay = df.groupby(by=df.index.names).apply(gp_extractor, delay_agg, activeNeus, 'TGOrifice', 'psActivity',
                                                [(0, 10), (10, 20), (20, 30), (30, 40), (40, np.inf)], tms)
    delay = delay.melt(id_vars=['MT'], value_vars=['Pre', *np.setdiff1d(delay.columns, ['MT', 'Pre'])],
                       var_name='Time elapsed (min)', value_name='Delay (ms)')

    plt.style.use('default')
    fig, axes = plt.subplots(2, 1)

    # plot late component peak firing rate
    sns.lineplot(data=pkFr.query("MT == 1.0 | MT == 0.9 | MT == 1.2"),
                 x='Time elapsed (min)', y='Peak firing rate (Hz)', hue='MT', style='MT',
                 markers=True, dashes=False, palette='deep', ax=axes[0])
    axes[0].set_title('TGOrifice-Injection', fontsize=12)

    # plot delay
    sns.swarmplot(data=delay.query("MT == 1.0 | MT == 0.9 | MT == 1.2"),
                  x='Time elapsed (min)', y='Delay (ms)', hue='MT', dodge=True, palette='deep',
                  size=3, legend=False, ax=axes[1])
    sns.violinplot(data=delay.query("MT == 1.0 | MT == 0.9 | MT == 1.2"),
                   x='Time elapsed (min)', y='Delay (ms)', hue='MT', inner=None,
                   cut=1, bw_adjust=0.5, ax=axes[1])
    violin_fill_false(axes[1])
    plt.savefig('figure3b.pdf', dpi='figure', format='pdf')
    plt.show()
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
