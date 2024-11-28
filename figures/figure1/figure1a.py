import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import copy
import pandas as pd
from statistics import mean

from tms_tg import TMSTG, EPOCHISOLATORS
from cachetools import cached
from cachetools.keys import hashkey
from itertools import zip_longest
from figures.helper_figs import (adjust_lim, ascertain_colName_from_colParams, selectEpochAndNeuron, fb,
                                 plot_MeanAndSEM, plot_populationAvgFR)


@cached(cache={}, key=lambda tms, uniqueEpochs: hashkey(uniqueEpochs))
def compute_raster(tms, uniqueEpochs):
    """
    Retrieve trialwise adjusted spiking time-stamps for specific epochs. This being a cached function avoids
    re-computing time-stamps for repeated calls using the same epochs.

    Args:
        tms: Instance of TMS class.
        uniqueEpochs: Unique epochs for which raster plots are computed.

    Returns:
        List: containing spike time-stamps from each block for specific epochs
    """
    return tms.psts


def construct_subplots(colParams, kind):
    """
    Construct subplots with specific number of rows depending on 'kind' arg and columns depending on 'colParams' arg.

    Args:
        colParams: Conditions to plot along each column.
        kind: Kind of plot ('rasterOnly', 'rasterAndPsfr', or 'rasterAndPopulationAvgFR').

    Returns:
        The output of matplotlib.pyplot.plt.subplots function called with specific number of rows and columns
    """
    match kind:
        case 'rasterOnly':
            return plt.subplots(3, len(colParams))
        case 'rasterAndPsfr' | 'rasterAndPopulationAvgFR':
            return plt.subplots(4, len(colParams))


def plot(tms, activeNeus, kind=None, colParams=None, xlim=None, epochAndNeuron=None):
    """
    Plot raster plots, peri-stimulus firing rates, and population average firing rates.

    Args:
        tms: Instance of TMS class.
        activeNeus: Active neurons determined by statistical test.
        kind (str): Type of plot ('rasterOnly', 'rasterAndPsfr', or 'rasterAndPopulationAvgFR').
        colParams (tuple): Parameters for columns.
        xlim (tuple): Limits for x-axis.
        epochAndNeuron (tuple): Tuple containing epoch and neuron information.

    Returns:
        None
    """

    assert type(colParams) == tuple or colParams is None, 'colNames has to be a list of dictionary items or None'
    assert kind in ('rasterOnly', 'rasterAndPsfr', 'rasterAndPopulationAvgFR'), 'unknown input for parameter "kind"'

    # Define raster categories to plot in separate rows
    rasterRowConds = ({'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '<1'}},
                      {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==1'}},
                      {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '>1'}})

    # Define condition for zero MT
    zeroMTCond = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS, [None, ])), 'MT': '==0'}}

    # Set plot styles
    plt.style.use('default')
    colorsPlt = ('C0', 'C1', 'C2')
    ylim = [np.inf, 0]
    fig, ax = construct_subplots(colParams, kind)
    animalNumsEpochNumsAndActiveNeuronNums_perCol = list()

    # Iterate over column parameters and do the plot
    for colIdx in range(len(colParams)):
        colName = ascertain_colName_from_colParams(colParams[colIdx])
        tms.analysis_params = copy.deepcopy(colParams[colIdx])
        selectBlocksinfo, selectBlocksinfoIdx = tms.filter_blocks
        epochIndices = selectBlocksinfo.index.unique()

        if not any(selectBlocksinfoIdx):
            print(f'Cannot plot for {colParams[colIdx]} as the associated data is missing in this group of animals')
            continue

        # Select an epoch and a neuron for plotting
        sampleBlocksinfo, sampleEpochIndex, neuIdx = (
            selectEpochAndNeuron(None if epochAndNeuron is None else epochAndNeuron[colIdx],
                                 tms, epochIndices, activeNeus, colParams[colIdx]))

        # Get the index of zeroMT ('MT' == 0) in order to exclude it from further selection (e.g., 'MT' <= 1)
        _, zeroMTIdx = fb(sampleBlocksinfo, zeroMTCond)

        # Select peristimulus timestamps and firing-rates pertaining to sampled epoch
        samplePSTS = compute_raster(tms, tuple(sampleBlocksinfo.index.to_numpy()))
        if kind in ('rasterAndPsfr', 'rasterAndPopulationAvgFR'):
            samplePSFR, ps_T = tms.compute_firingrate(
                *tms.analysis_params['peristimParams']['smoothingParams'].values(),
                *tms.analysis_params['peristimParams']['timeWin'],
                tms.analysis_params['peristimParams']['trigger'])
            sampleBaselineFR, ps_baseline_T = (
                tms.compute_firingrate('rectangular',
                                       np.diff(tms.analysis_params['peristimParams']['baselinetimeWin']).item(0),
                                       0.0,
                                       mean(tms.analysis_params['peristimParams']['baselinetimeWin']),
                                       tms.analysis_params['peristimParams']['baselinetimeWin'][1],
                                       tms.analysis_params['peristimParams']['trigger']))

        # Statistics
        animalNumsEpochNumsAndActiveNeuronNums_perCol.append((
            np.unique([item[0] for item in epochIndices]).size,
            list(zip((epochIndices), [activeNeus[item].sum() for item in epochIndices]))
        ))

        # --------------------------------------------------------------------------------------------------------------
        ax[0][colIdx].set_title(sampleEpochIndex[0] + '/' + colName + 'Neu-' + str(neuIdx), fontsize=6)
        for i in range(rowIdx := len(rasterRowConds)):
            _, blockIdx = fb(sampleBlocksinfo, rasterRowConds[i], ~zeroMTIdx)
            assert sum(blockIdx) >= 1, \
                f"epoch {sampleEpochIndex} does not have MT{rasterRowConds[i]['selectionParams']['MT']} values"

            # If there are multiple blocks with same rasterRowCond, select the one with maximum no. of Trigs
            if sum(blockIdx) > 1:
                idx = np.where(blockIdx)[0][sampleBlocksinfo.loc[blockIdx, 'no. of Trigs'].argmax()]
                blockIdx.iloc[np.setdiff1d(np.where(blockIdx)[0], idx)] = False

            # If first column in being plotted then set the ylabel
            if colIdx == 0:
                ax[i][colIdx].set_ylabel(
                    f'MT = {sampleBlocksinfo[blockIdx]["MT"].values[0]}\nTrials (N)', fontsize=8)

            # ----------------------------------------------------------------------------------------------------------
            # Plot raster
            ax[i][colIdx].eventplot([tms.analysis_params['peristimParams']['timeWin'][0] + item
                                     for item in samplePSTS[np.nonzero(blockIdx)[0][0]][neuIdx]],
                                    colors=colorsPlt[i])

            # ----------------------------------------------------------------------------------------------------------
            # Plot PSFR
            if kind == 'rasterAndPsfr':
                selectPSFR = samplePSFR[np.nonzero(blockIdx)[0][0]]
                plot_MeanAndSEM(selectPSFR.mean(axis=0)[:, neuIdx],
                                selectPSFR.std(axis=0)[:, neuIdx] / np.sqrt(selectPSFR.shape[0]),
                                ps_T,
                                ax[rowIdx][colIdx],
                                colorsPlt[i],
                                f'MT = {sampleBlocksinfo[blockIdx]["MT"].mean()}')
                if colIdx == 0:
                    ax[rowIdx][colIdx].set_ylabel('Firing rate (Hz)', fontsize=8)

            # ----------------------------------------------------------------------------------------------------------
            # plot population average firing rate
            if kind == 'rasterAndPopulationAvgFR':
                plot_populationAvgFR(samplePSFR, ps_T, selectBlocksinfo, zeroMTCond, activeNeus,
                                     rasterRowConds, ax[rowIdx][colIdx], colorsPlt,
                                     [item['selectionParams']['MT'] for item in rasterRowConds])

        if kind != 'rasterOnly':
            ylim = [min(ax[rowIdx][colIdx].get_ylim()[0], ylim[0]), max(ax[rowIdx][colIdx].get_ylim()[1], ylim[1])]

    # Adjust xlim if specified
    if xlim is not None:
        adjust_lim(ax, xlim, 'xlim')
    # if kind != 'rasterOnly':
    #     adjust_lim(ax[rowIdx, :], ylim, 'ylim')
    plt.savefig('figure1a.pdf', dpi='figure', format='pdf')
    plt.show()
    print('[(Number of Animals, Number of Epochs, Number of Neurons per epoch), ...]: ',
          animalNumsEpochNumsAndActiveNeuronNums_perCol)


if __name__ == '__main__':
    # Set the number of threads for Numba to optimize parallel processing
    nb.set_num_threads(max(1, int(nb.config.NUMBA_NUM_THREADS // 1.25)))

    # Define selection parameters for different epochs
    # epochs = (
    #     {'selectionParams': {'Epoch': {'Region': 'MC', 'Layer': 'L5'}}},
    #     {'selectionParams': {'Epoch': {'Region': 'SC', 'Layer': 'L5'}}},
    #     {'selectionParams': {'Epoch': {'Region': 'thal', 'Layer': None}, 'RecArea ': 'BZ'}},
    #     {'selectionParams': {'Epoch': {'Region': 'thal', 'Layer': None}, 'RecArea ': 'CZ'}},
    # )
    keys = ('Region', 'Layer', 'Animal', 'Mov', 'Depth', 'CoilHemVsRecHem')
    epochs = (
        {'selectionParams': {'Epoch': {key: value for key, value in zip_longest(keys, ('MC', 'L5'))}}},
        {'selectionParams': {'Epoch': {key: value for key, value in zip_longest(keys, ('SC', 'L5'))}}},
        {'selectionParams': {'Epoch': {key: value for key, value in zip_longest(keys, ('VC', 'L5'))}}}
    )
    #{'selectionParams': {'Epoch': {key: value for key, value in zip_longest(keys, ('VC', 'L5'))}}},

    # Load TMS data from the specified Excel file
    animalList = r'G:\Vishnu\data\TMSTG\animalList.xlsx'
    tms = TMSTG.load(animalList)

    # Set the peristimulus time window
    tms.analysis_params = {'peristimParams': {'timeWin': (-50.0, 350.0)}}

    # Load or compute active neurons from statistical analysis
    # activeNeu = tms.stats_is_signf_active()
    activeNeu = pd.read_pickle("./activeNeu_Cortex")

    # Define plot kinds for different types of plots
    plotKinds = ('rasterOnly', 'rasterAndPsfr', 'rasterAndPopulationAvgFR')

    # Plot raster and peri-stimulus firing rate (PSFR) for specified epochs and neurons
    # plot(tms, activeNeu, kind=plotKinds[1], colParams=epochs, xlim=[-50, 350], epochAndNeuron=
    #      ({'epochIndex': ('20200705', 'MC', 'L5', 'opposite', 'none', '1467'), 'neuIdx': 5},
    #       {'epochIndex': ('20200705', 'SC', 'L5', 'same', 'none', '1496'), 'neuIdx': 4},
    #       {'epochIndex': ('20200705', 'VC', 'L5', 'same', 'none', '1404'), 'neuIdx': 1}))

    plot(tms, activeNeu, kind=plotKinds[1], colParams=epochs, xlim=[-25, 50], epochAndNeuron=
         ({'epochIndex': ('20200520', 'MC', 'L5', 'none', 'none', '1438'), 'neuIdx': 1},
          {'epochIndex': ('20200520', 'SC', 'L5', 'none', 'none', '1565'), 'neuIdx': 1},
          {'epochIndex': ('20200705', 'VC', 'L5', 'same', 'none', '1404'), 'neuIdx': 0}))

    # Reset the number of threads for Numba to default
    nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
