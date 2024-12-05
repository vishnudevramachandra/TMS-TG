import numpy as np
import pandas as pd
import os

from tms_tg import TMSTG, EPOCHISOLATORS
from itertools import zip_longest
from figures.helper_figs import gb_addinfo
from figures.plot_types import plot_inj_effect


if __name__ == '__main__':
    # epochs = [('20190714', 'MC', 'L5', 'none', 'none', '1400'),
    #           ('20191119', 'MC', 'L5', 'none', 'none', '1330'),
    #           ('20191201', 'MC', 'L5', 'none', 'none', '1300'),
    #           ('20191201', 'MC', 'L5', 'none', 'none', '1301')]

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

    # Define the selection parameters for TGOrifice animals
    manipulation = 'Skin-Injection'
    skinInjAnimals = tms.blocksinfo[
        tms.blocksinfo[manipulation].fillna(value='No').str.contains('Post', case=False)
    ].index.get_level_values('Animal').unique()
    tms.analysis_params = {'selectionParams': {'Epoch': dict(zip_longest(EPOCHISOLATORS,
                                                                         [tuple(skinInjAnimals), 'MC', None])),
                                               'MT': '>=0.8&<=1.2'}
                           }

    # Filter blocksinfo using above defined selection parameters
    selectBlocksinfo, selectBlocksinfoIdx = tms.filter_blocks

    # Either run statistical test to find out active neurons or load it, if previously computed and saved.
    activeNeus = pd.read_pickle("./activeNeu_Cortex") if os.path.isfile("./activeNeu_Cortex") \
        else tms.stats_is_signf_active()

    # Update statistics and print N's
    epochIndices = selectBlocksinfo.index.unique()
    animalNumsEpochNumsAndActiveNeuronNums = ((np.unique([item[0] for item in epochIndices]).size,
                                               len(epochIndices),
                                               [activeNeus[item].sum() for item in epochIndices]))
    print('[(Number of Animals, Number of Epochs, Number of Active Neurons per epoch), ...]: \n',
          animalNumsEpochNumsAndActiveNeuronNums)

    # Check if there are any valid epoch indices to plot
    if not any(epochIndices):
        print(f'Cannot plot as the associated data is missing in this group of animals')
        quit()

    # Compute peristimulus firing rate if not already done and save it in blocksinfo for later use
    _, _ = tms.compute_firingrate(*tms.analysis_params['peristimParams']['smoothingParams'].values(),
                                  *tms.analysis_params['peristimParams']['timeWin'],
                                  tms.analysis_params['peristimParams']['trigger'])

    # The data falling withing these intervals is extracted separately and used subsequently in the plots
    postTimeIntervals = ['Pre', (0, 10), (10, 20), (20, 30), (30, 40), (40, np.inf)]

    # Generate plots
    plot_inj_effect(tms, selectBlocksinfo, activeNeus, manipulation, postTi=postTimeIntervals)

    # Save the updated (join and apply) grandBlocksinfo if the original exists
    if os.path.isfile('../../grandBlocksinfo'):
        grandBlocksinfo = gb_addinfo(pd.read_pickle('../../grandBlocksinfo'), tms)
        grandBlocksinfo.to_pickle('../../grandBlocksinfo')
