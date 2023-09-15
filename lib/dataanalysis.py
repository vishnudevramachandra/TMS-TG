import numpy as np
import pandas as pd
import lib.matpy as mp
import numba as nb
from lib.math_fcns import gaussian, rectangular


# TODO: caching
def peristim_firingrate_decorator(fcn):
    def wrapper(spikeTimes: list[np.ndarray],
                timeIntervals: list[tuple[float, float]],
                smoothingParams):
        assert spikeTimes is not None and timeIntervals is not None

        smWidth = smoothingParams['width']  # needed as numba.jit cannot optimize dict type
        match smoothingParams['win']:
            case 'gaussian':
                @nb.jit(nopython=True)
                def f(x):
                    return sum(gaussian(x, sig=smWidth)) / 1e-3

            case _:
                @nb.jit(nopython=True)
                def f(x):
                    return sum(rectangular(x, width=smWidth)) / (smWidth * 1e-3)

        step = smoothingParams['width'] * (1 - smoothingParams['overlap'])
        ps_T = np.arange(0, timeIntervals[0][1] - timeIntervals[0][0], step)
        ps_FR = np.zeros((len(timeIntervals), len(ps_T), len(spikeTimes)), dtype=np.float_)

        return fcn(ps_FR, ps_T, spikeTimes, timeIntervals, f)

    return wrapper


@peristim_firingrate_decorator
@nb.jit(nopython=True)
def peristim_firingrate(
        ps_FR, ps_T, spikeTimes, timeIntervals, smoothingFcn) -> tuple[np.ndarray, np.ndarray]:
    for trl_n, timeInterval in enumerate(timeIntervals):
        # loop over time steps
        for step_n, step in enumerate(ps_T):
            step += timeInterval[0]
            # loop over neurons (use timestamps of each neuron to assign firing rate)
            for i, singleUnitSpikeTimes in enumerate(spikeTimes):
                diff = step - singleUnitSpikeTimes
                # insert the firing rate for each neuron, for each time step, for each trial
                ps_FR[trl_n, step_n, i] = smoothingFcn(diff)
    return ps_FR, ps_T


def peristim_timestamp(
        spikeTimes: list[np.ndarray],
        timeIntervals: list[tuple[float, float]]) -> list[list[np.ndarray]]:
    assert spikeTimes is not None and timeIntervals is not None

    ps_TS = []
    for singleUnitSpikeTimes in spikeTimes:
        x = []
        for ti in timeIntervals:
            x.append(singleUnitSpikeTimes[
                         (ti[0] <= singleUnitSpikeTimes)
                         & (singleUnitSpikeTimes <= ti[1])]
                     - ti[0])
        ps_TS.append(x)

    return ps_TS


if __name__ == '__main__':
    # matfile = mp.MATfile('G:\Vishnu\Analysis\TMS-TG\data\SLAnalys.mat')
    matfile = mp.MATfile(r'G:\Vishnu\data\TMSTG\20180922\20180922_cfa_ipsi_L6_SLData_SpikeAmpTH3.0.mat')
    data = matfile.read()

    trigChanIdx = data['TrigChan_ind'][0, 0].astype(int) - 1
    refs = data['rawData/trigger'].flatten()
    trigger = [data[i].flatten() for i in refs][trigChanIdx] * 1e3

    startT, endT = -20, 100
    timeIntervals = nb.typed.List()
    [timeIntervals.append((x + startT, x + endT)) for x in trigger[::2]]

    multiUnitSpikeTimes = data['SpikeModel/SpikeTimes/data'].flatten()
    refs = data['SpikeModel/ClusterAssignment/data'].flatten()
    allSpikeTimes_neuronIdx = [data[i].flatten().astype(int) - 1 for i in refs]
    singleUnitsSpikeTimes = nb.typed.List()
    [singleUnitsSpikeTimes.append(multiUnitSpikeTimes[idx]) for idx in allSpikeTimes_neuronIdx]

    ps_TS = peristim_timestamp(singleUnitsSpikeTimes, timeIntervals)

    # ps_FR, ps_T = peristim_firingrate(singleUnitsSpikeTimes, timeIntervals)

    ps_FR, ps_T = peristim_firingrate(singleUnitsSpikeTimes, timeIntervals,
                                      smoothingParams={'win': 'gauss', 'width': 3.0, 'overlap': 1 / 3})

    ps_FR, ps_T = peristim_firingrate(singleUnitsSpikeTimes, timeIntervals,
                                      avg={'win': 'gauss', 'width': 3.0, 'overlap': 0.0})
