import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import pandas as pd
import copy
import os
import seaborn as sns

from tms_tg import TMSTG, EPOCHISOLATORS
from itertools import zip_longest
from figures.helper_figs import adjust_lim, plot_populationAvgFR, pick_delay, gb_addinfo
from lib.helper_tms_tg import merge_selectionParams


def plot(tms, activeNeus, colParam, axes, xlim=None, frKind=None, frYlim=None, delayYlim=None):
    """
        Plot population average firing rates and latency of late activity component.

    Args:
        tms: Instance of TMS class.
        activeNeus: Active neurons determined by statistical test.
        colParam (dict): Parameters for column.
        axes (array): matplotlib.axes.Axes for plotting
        xlim (tuple): Limits for x-axis.
        frKind (str): Type of peri-stim 'firing-rate plot' ('Average', or 'Normalized').
        frYlim (tuple): Limits for y-axis of 'peri-stim firing-rate' plot
        delayYlim (tuple): Limits for y-axis of delay plots showing the latency of late activity component

    Returns:
        None
    """

    # Define trace conditions as selection of different layers
    remSelectionParams = EPOCHISOLATORS.copy()
    remSelectionParams.pop(2)
    traceConds = ({'selectionParams': {'Epoch': {'Layer': 'L23', **dict(zip_longest(remSelectionParams, [None, ]))}}},
                  {'selectionParams': {'Epoch': {'Layer': 'L4', **dict(zip_longest(remSelectionParams, [None, ]))}}},
                  {'selectionParams': {'Epoch': {'Layer': 'L5', **dict(zip_longest(remSelectionParams, [None, ]))}}},
                  {'selectionParams': {'Epoch': {'Layer': 'L6', **dict(zip_longest(remSelectionParams, [None, ]))}}})

    # Define condition for zero MT
    zeroMTCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==0'}}

    # Define post-injection conditions that are present in this cohort
    postConds = [{'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), postKind: 'Post'}} for
                 postKind in {'Skin-Injection', 'TG-Injection ', 'TGOrifice'} & set(tms.blocksinfo.columns)]

    plt.style.use('default')
    colorsPlt = ('C0', 'C1', 'C2', 'C3')

    # Check if any data is available for the selected column parameter
    if not any(selectBlocksinfoIdx):
        print(f'Cannot plot for {colParam} as the associated data is missing in this group of animals')
        return frYlim, delayYlim

    # --------------------------------------------------------------------------------------------------------------
    # Add a title to the plot
    colName = colParam['selectionParams']['Epoch']['Region']
    axes[0].set_title(colName, fontsize=8)

    # --------------------------------------------------------------------------------------------------------------
    # Loop through trace conditions and do the plot
    delays = list()
    oldSelParams = copy.deepcopy(tms.analysis_params['selectionParams'])
    for idx, traceCond in enumerate(traceConds):
        tms.analysis_params = {'selectionParams': merge_selectionParams(
            copy.deepcopy(tms.analysis_params['selectionParams']), copy.deepcopy(traceCond['selectionParams']))}

        # Plot population activity
        plot_populationAvgFR(tms, activeNeus, ax=axes[0], kind=frKind, excludeConds=[zeroMTCond, *postConds],
                             lineLabel=traceCond['selectionParams']['Epoch']['Layer'], lineColor=colorsPlt[idx])

        # Compute delays
        delays.append(pick_delay(tms, activeNeus, excludeConds=[zeroMTCond, *postConds]))

        # Reset analysis parameters
        tms.analysis_params = {'selectionParams': oldSelParams}

    # Plot delay using swarmplot and violinplot
    sns.swarmplot(data=delays, color='k', size=2, ax=axes[1])
    sns.violinplot(data=delays, inner=None, cut=1, bw_adjust=0.5, ax=axes[1])
    axes[1].set_xticks(range(len(traceConds)),
                       labels=[item['selectionParams']['Epoch']['Layer'] for item in traceConds])

    # --------------------------------------------------------------------------------------------------------------
    # Adjust xlim if specified
    if xlim is not None:
        adjust_lim(axes[0], xlim, 'xlim')

    return axes[0].get_ylim(), axes[1].get_ylim()


if __name__ == '__main__':
    # Set the number of threads for Numba
    nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)

    # Define column parameters for analysis
    colParams = (
        {'selectionParams': {'Epoch': {'Region': 'MC', }, 'MT': '==1'}},
        {'selectionParams': {'Epoch': {'Region': 'SC', }, 'MT': '==1'}},
        {'selectionParams': {'Epoch': {'Region': 'VC', }, 'MT': '==1'}},
    )

    # Load data from the animal list file
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

    # Load or compute active neurons from statistical analysis
    # activeNeus = tms.stats_is_signf_active()
    activeNeus = pd.read_pickle("./activeNeu_Cortex")

    # Initialize lists to store statistics
    animalNumsEpochNumsAndActiveNeuronNums_perCol = list()

    # Define y-axis limits for delay plots
    dYlim = [np.inf, 0]

    # Create subplots
    fig, ax = plt.subplots(2, len(colParams))

    # Iterate over column parameters
    for colIdx in range(len(colParams)):
        # Set analysis parameters
        tms.analysis_params = copy.deepcopy(colParams[colIdx])
        selectBlocksinfo, selectBlocksinfoIdx = tms.filter_blocks
        epochIndices = selectBlocksinfo.index.unique()

        # Compute statistics
        animalNumsEpochNumsAndActiveNeuronNums_perCol.append((np.unique([item[0] for item in epochIndices]).size,
                                                              len(epochIndices),
                                                              [activeNeus[item].sum() for item in epochIndices]))

        # Plot firing rate and delay
        frYlim, delayYlim = plot(
            tms, activeNeus, colParams[colIdx], ax[:, colIdx], xlim=[-25, 50], frKind='Average')

        # Set labels and legends for the plots
        if colIdx == 0:
            ax[0][colIdx].set_ylabel('Average\nfiring rate', fontsize=8)
            ax[0][colIdx].legend(fontsize=6)
            ax[1][colIdx].set_ylabel('Delay (ms)', fontsize=8)

        # Update delay y-axis limits
        if delayYlim is not None:
            dYlim = [min(delayYlim[0], dYlim[0]), max(delayYlim[1], dYlim[1])]

    # Adjust y-axis limits for delay plots
    for colIdx in range(len(colParams)):
        adjust_lim(ax[1, :], dYlim, 'ylim')

    # Save and display the figure
    plt.savefig('figure1b.pdf', dpi='figure', format='pdf')
    plt.show()

    # Print the computed statistics
    print('[(Number of Animals, Number of Epochs, Number of Neurons per epoch), ...]: ',
          animalNumsEpochNumsAndActiveNeuronNums_perCol)

    # Save the updated (join and apply) grandBlocksinfo if the original exists
    if os.path.isfile('../../grandBlocksinfo'):
        grandBlocksinfo = gb_addinfo(pd.read_pickle('../../grandBlocksinfo'), tms)
        grandBlocksinfo.to_pickle('../../grandBlocksinfo')

    # Reset the number of threads for Numba
    nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)

