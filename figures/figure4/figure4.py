import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import copy
import os

from tms_tg import TMSTG, EPOCHISOLATORS, hem_CoilDir_selector
from itertools import zip_longest, product
from figures.helper_figs import (gb_addinfo, gp_extractor, delay_agg, compute_excludeIdx)


def plot(tms, df, activeNeus, cortexAblAndTGcutConds=None, colOrder=None, savefig=False):
    """
        Plot latency of late activity component.

    Args:
        tms: Instance of TMS class.
        df: filtered blocksinfo.
        activeNeus: Active neurons determined by statistical test.
        cortexAblAndTGcutConds: List of different pairs of cortical-ablation and TGcut conditions.
        colOrder: List of different pairs of stimulus-hemisphere and coil-direction conditions.
    Parameters:
        savefig: Whether to save the figure
        .

    Returns:
        None
    """

    # Compute delay
    delay = df.groupby(by=df.index.names + ['CoilDir', 'CHvsRH', 'TGcut', 'CortexAbl', 'RecArea ']) \
        .apply(gp_extractor, delay_agg, activeNeus, 'TGcut', 'psActivity', None, tms)
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

    # Set parameters that are not passed in.
    if cortexAblAndTGcutConds is None:
        cortexAblAndTGcutConds = ['No/No', 'LH/No', 'Both/No',
                                  'LH/Lv1,Lv2', 'LH/Lv1,Lv2,Lv3', 'LH/Rv1,Rv2,Rv3', 'Both/Lv1,Lv2,Lv3',
                                  'LH/Lv1,Lv2,Lv3,Rv1,Rv2', 'Both/Lv1,Lv2,Lv3,Rv1,Rv2,Rv3']

        # Remove conditions from the list that are not present in the data and
        # append conditions to the list that are present in the data but missing from the list.
        cortexAblAndTGcutConds = [item for item in cortexAblAndTGcutConds
                                  if item in delay['CortexAbl and TGcut'].unique()]
        [cortexAblAndTGcutConds.append(item) for item in delay['CortexAbl and TGcut'].unique()
         if item not in cortexAblAndTGcutConds]

    if colOrder is None:
        colOrder = ['same/ML', 'same/LM', 'opposite/ML', 'opposite/LM']

        # Remove conditions from the list that are not present in the data and
        # append conditions to the list that are present in the data but missing from the list.
        colOrder = [item for item in colOrder if item in delay['StimHem and CoilDir'].unique()]
        [colOrder.append(item) for item in delay['StimHem and CoilDir'].unique() if item not in colOrder]

    # Plot delay as swarm plot
    plt.style.use('default')
    swarmplot = sns.catplot(
        data=delay,
        kind='swarm', x='CortexAbl and TGcut', y='Delay (ms)', hue='MT', col='StimHem and CoilDir',
        order=cortexAblAndTGcutConds,
        col_order=colOrder,
        size=3, dodge=True, palette='deep', aspect=.5)
    for ax in swarmplot.axes.flat:
        ax.tick_params(axis='x', labelsize='small', labelrotation=90)
    swarmplot.fig.suptitle(f"Data taken from area[s]: {delay[('', 'RecArea ')].unique()}", fontsize=16)
    swarmplot.figure.tight_layout()
    sns.move_legend(swarmplot, "upper right")
    if savefig:
        plt.savefig('figure4_swarm.pdf', dpi='figure', format='pdf')
    swarmplot.figure.show()

    # Plot delay as box plot
    boxplot = sns.catplot(
        data=delay,
        kind='box', x='CortexAbl and TGcut', y='Delay (ms)', col='StimHem and CoilDir',
        order=cortexAblAndTGcutConds,
        col_order=colOrder,
        aspect=0.5)
    for ax in boxplot.axes.flat:
        ax.tick_params(axis='x', labelsize='small', labelrotation=90)
    boxplot.fig.suptitle(f"Data taken from area[s]: {delay[('', 'RecArea ')].unique()}", fontsize=16)
    boxplot.figure.tight_layout()
    if savefig:
        plt.savefig('figure4_box.pdf', dpi='figure', format='pdf')
    boxplot.figure.show()


if __name__ == '__main__':

    # Load the animal list data
    animalList = r'G:\Vishnu\data\TMSTG\animalList.xlsx'
    tms = TMSTG.load(animalList)

    # Load grandBlocksinfo if it exists and merge it with the current blocksinfo
    if os.path.isfile('../../grandBlocksinfo'):
        grandBlocksinfo = pd.read_pickle('../../grandBlocksinfo')
        for col in {'psActivity', 'delay'} & set(grandBlocksinfo.columns):
            bIndicesInGBIndices = tuple(set(tms.blocksinfo.index.unique()) & set(grandBlocksinfo.index.unique()))
            for index in bIndicesInGBIndices:
                tms.blocksinfo.loc[index, col] = grandBlocksinfo.loc[index, col]
        tms.blocksinfo = tms.blocksinfo.where(pd.notnull(tms.blocksinfo), None)
        tms.filter_blocks = None

    # Define the selection parameters for TGcut animals
    tgCutAnimals = tms.blocksinfo[
        tms.blocksinfo['TGcut'].str.contains('Lv1|Lv2|Lv3|Rv1|Rv2|Rv3')
    ].index.get_level_values('Animal').unique()
    tms.analysis_params = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS,
                                                                         [tuple(tgCutAnimals), 'MC', None])),
                                               'MT': '>=0.8&<=1.2',
                                               'TGOrifice': '!Post',
                                               'CoilDir': 'LM|ML',
                                               'CHvsRH': 'same'}
                           }

    # Filter blocksinfo using above defined selection parameters
    selectBlocksinfo, selectBlocksinfoIdx = tms.filter_blocks

    # Either run statistical test to find out active neurons or load it, if previously computed and saved.
    activeNeus = pd.read_pickle("./activeNeu_TGcut") if os.path.isfile("./activeNeu_TGcut") \
        else tms.stats_is_signf_active()

    # Update statistics and print N's
    epochIndices = selectBlocksinfo.index.unique()
    animalNumsEpochNumsAndActiveNeuronNums = ((np.unique([item[0] for item in epochIndices]).size,
                                               len(epochIndices),
                                               [activeNeus[item].sum() for item in epochIndices]))
    print('[(Number of Animals, Number of Epochs, Number of Neurons per epoch), ...]: ',
          animalNumsEpochNumsAndActiveNeuronNums)

    # Check if there are any valid epoch indices to plot
    if not any(epochIndices):
        print(f'Cannot plot TGOrifice as the associated data is missing in this group of animals')
        quit()

    # Compute peristimulus firing rate if not already done and save it in blocksinfo for later use
    _, _ = tms.compute_firingrate(*tms.analysis_params['peristimParams']['smoothingParams'].values(),
                                  *tms.analysis_params['peristimParams']['timeWin'],
                                  tms.analysis_params['peristimParams']['trigger'])

    # Swarm- & boxplot accept the order in which different data classes are displayed, whose sequence is defined here.
    # cortexAblAndTGcutConds = ['No/No', 'LH/No', 'Both/No', 'LH/Lv1,Lv2,Lv3', 'Both/Lv1,Lv2,Lv3', 'LH/Rv1,Rv2,Rv3']

    # Generate plots
    plot(tms, selectBlocksinfo, activeNeus)

    # Save the updated (join and apply) grandBlocksinfo if the original exists
    if os.path.isfile('../../grandBlocksinfo'):
        grandBlocksinfo = gb_addinfo(pd.read_pickle('../../grandBlocksinfo'), tms)
        grandBlocksinfo.to_pickle('../../grandBlocksinfo')
