import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import copy
import os

from tms_tg import TMSTG, EPOCHISOLATORS
from itertools import zip_longest
from figures.helper_figs import (compute_excludeIdx, gb_addinfo, gp_extractor, pkFr_agg, delay_agg, violin_fill_false)


def plot(tms, activeNeus, xlim=None):
    """
        Plot population average firing rates and latency of late activity component.

    Args:
        tms: Instance of TMS class.
        activeNeus: Active neurons determined by statistical test.
        xlim (tuple): Limits for x-axis.

    Returns:
        None
    """

    # Save the original selection parameters
    oldSelParams = copy.deepcopy(tms.analysis_params['selectionParams'])

    # Define post-condition and other post-conditions
    postCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'Skin-Injection': 'Post'}}
    otherPostConds = [{'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), postKind: 'Post'}} for
                      postKind in {'TG-Injection ', 'TGOrifice'} & set(tms.blocksinfo.columns)]

    # Apply post-condition and filter blocks-info
    tms.analysis_params = postCond
    selectBlocksinfo, selectBlocksinfoIdx = tms.filter_blocks
    excludeIdx = compute_excludeIdx(selectBlocksinfo, otherPostConds)
    epochIndices = selectBlocksinfo.index[~excludeIdx].unique()

    # Check if there are any valid epoch indices to plot
    if not any(epochIndices):
        print(f'Cannot plot Skin-Injection as the associated data is missing in this group of animals')
        return

    # Compute peristimulus firing rate if not already done and save it in blocksinfo for later use
    for epochIndex in epochIndices:
        tms.analysis_params = {'selectionParams': {'Epoch': {x: y if y != 'none' else None
                                                             for x, y in zip(epochIndices.names, epochIndex)}}}
        _, _ = tms.compute_firingrate(*tms.analysis_params['peristimParams']['smoothingParams'].values(),
                                      *tms.analysis_params['peristimParams']['timeWin'],
                                      tms.analysis_params['peristimParams']['trigger'])
    # Restore original selection parameters
    tms.analysis_params = {'selectionParams': oldSelParams}

    # Update statistics
    animalNumsEpochNumsAndActiveNeuronNums = ((np.unique([item[0] for item in epochIndices]).size,
                                               len(epochIndices),
                                               [activeNeus[item].sum() for item in epochIndices]))

    # Extract relevant data from blocksinfo
    df = tms.blocksinfo.loc[epochIndices, :]

    # Rearrange peak firing rate
    pkFr = df.groupby(by=df.index.names).apply(gp_extractor, pkFr_agg, activeNeus, 'Skin-Injection', 'psActivity',
                                               [(0, 10), (10, 20), (20, 30), (30, 40), (40, np.inf)], tms)
    pkFr = pkFr.melt(id_vars=['MT'], value_vars=['Pre', *np.setdiff1d(pkFr.columns, ['MT', 'Pre'])],
                     var_name='Time elapsed (min)', value_name='Peak firing rate (Hz)')

    # Rearrange delay
    delay = df.groupby(by=df.index.names).apply(gp_extractor, delay_agg, activeNeus, 'Skin-Injection', 'psActivity',
                                                [(0, 10), (10, 20), (20, 30), (30, 40), (40, np.inf)], tms)
    delay = delay.melt(id_vars=['MT'], value_vars=['Pre', *np.setdiff1d(delay.columns, ['MT', 'Pre'])],
                       var_name='Time elapsed (min)', value_name='Delay (ms)')

    # Plot settings
    plt.style.use('default')
    fig, axes = plt.subplots(2, 1)

    # Plot late activity component's peak firing rate
    sns.lineplot(data=pkFr, x='Time elapsed (min)', y='Peak firing rate (Hz)', hue='MT', style='MT',
                 markers=True, dashes=False, palette='deep', ax=axes[0])
    axes[0].set_title('Skin-Injection', fontsize=12)

    # Plot delay of late activity component
    sns.swarmplot(data=delay, x='Time elapsed (min)', y='Delay (ms)', hue='MT', dodge=True, palette='deep',
                  size=3, legend=False, ax=axes[1])
    sns.violinplot(data=delay, x='Time elapsed (min)', y='Delay (ms)', hue='MT', inner=None,
                   cut=1, bw_adjust=0.5, ax=axes[1])
    violin_fill_false(axes[1])

    # Save and display the plot
    plt.savefig('figure3a_new.pdf', dpi='figure', format='pdf')
    plt.show()

    # Print statistics
    print('[(Number of Animals, Number of Epochs, Number of Neurons per epoch), ...]: ',
          animalNumsEpochNumsAndActiveNeuronNums)


if __name__ == '__main__':
    # epochs = [('20190714', 'MC', 'L5', 'none', 'none', '1400'),
    #           ('20191119', 'MC', 'L5', 'none', 'none', '1330'),
    #           ('20191201', 'MC', 'L5', 'none', 'none', '1300'),
    #           ('20191201', 'MC', 'L5', 'none', 'none', '1301')]

    # Load the animal list data
    animalList = r'G:\Vishnu\data\TMSTG\animalList.xlsx'
    tms = TMSTG.load(animalList)
    # activeNeu = tms.stats_is_signf_active()

    # Load grandBlocksinfo if it exists and merge it with the current blocksinfo
    if os.path.isfile('../../grandBlocksinfo'):
        grandBlocksinfo = pd.read_pickle('../../grandBlocksinfo')
        for col in {'psActivity', 'delay'} & set(grandBlocksinfo.columns):
            bIndicesInGBIndices = tuple(set(tms.blocksinfo.index.unique()) & set(grandBlocksinfo.index.unique()))
            for index in bIndicesInGBIndices:
                tms.blocksinfo.loc[index, col] = grandBlocksinfo.loc[index, col]
        tms.blocksinfo = tms.blocksinfo.where(pd.notnull(tms.blocksinfo), None)
        tms.filter_blocks = None

    # Generate plots
    plot(tms, pd.read_pickle("./activeNeu_Cortex"), xlim=[-20, 60])

    # Save the updated (join and apply) grandBlocksinfo if the original exists
    if os.path.isfile('../../grandBlocksinfo'):
        grandBlocksinfo = gb_addinfo(pd.read_pickle('../../grandBlocksinfo'), tms)
        grandBlocksinfo.to_pickle('../../grandBlocksinfo')
