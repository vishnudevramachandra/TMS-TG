import numpy as np
import pandas as pd
from MATfile import MATfile
from MATdata import MATdata


def PSTH(data=None, time_win=None, avg={'win': 'rect', 'width': 1, 'overlap': 0}):
    assert data is not None and time_win is not None

    spiketimes = data['SpikeModel/SpikeTimes/data'].flatten()
    refs = data['SpikeModel/ClusterAssignment/data'].flatten()
    cluster_idx = [data[i].flatten().astype(int) - 1 for i in refs]
    clust_spktms = [spiketimes[idx] for idx in cluster_idx]

    step = avg['width'] * (1 - avg['overlap'])
    psth_T = np.arange(0, time_win[0][1] - time_win[0][0], step)
    psth_Y = np.zeros((len(psth_T), len(cluster_idx), len(time_win)), dtype=spiketimes.dtype)

    match avg['win']:
        case 'rect':
            func = lambda x: sum(np.logical_and(np.greater(x, -avg['width'] / 2),
                                                np.less_equal(x, avg['width'] / 2))
                                 ) / (avg['width'] * 1e-3)

        case 'gauss':
            func = lambda x: sum(np.exp(-0.5 * (x / avg['width']) ** 2) /
                                 (avg['width'] * np.sqrt(2 * np.pi))) / 1e-3

    # loop over trials
    for trl_n in range(len(time_win)):
        # loop over time steps
        for t_idx, t in enumerate(psth_T):
            t += time_win[trl_n][0]
            # loop over clusters (specifically the spiketimes)
            for clust_n, spktms in enumerate(clust_spktms):
                diff = spktms - t
                # insert the spike rate for each bin, for each neuron, for each trial
                psth_Y[t_idx, clust_n, trl_n] = func(diff)

    return psth_Y, psth_T


if __name__ == '__main__':
    matfile = MATfile('G:\Vishnu\Analysis\TMS-TG\lib\SLAnalys.mat')
    data = matfile.read()

    trigChan_ind = data['TrigChan_ind'][0, 0].astype(int) - 1
    refs = data['rawData/trigger'].flatten()
    trigger = [data[i].flatten() for i in refs][trigChan_ind] * 1e3
    start_t, end_t = -20, 100
    trigger = [(x + start_t, x + end_t) for x in trigger[0::1]]

    psth = PSTH(data, trigger)
    psth = PSTH(data, trigger, avg={'win': 'gauss', 'width': 3, 'overlap': 1/3})
