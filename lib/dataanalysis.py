import numpy as np
import lib.matpy as mp      # module for handling .mat files
import numba as nb
from scipy import stats

# import custom 1-D functions that use parallel computing for speed-up
from lib.math_fcns import gaussian, rectangular, triangular


def peristim_firingrate_decorator(fcn):
    """Decorator for calculating peristimulus firing rate."""

    def wrapper(spikeTimes: list[np.ndarray],
                timeWindows: np.ndarray,
                smoothingParams):
        # Ensure necessary inputs are provided
        assert spikeTimes is not None and timeWindows is not None, 'Both spikeTimes and timeWindow cannot be None'

        # Extract smoothing width from parameters (needed as numba.jit cannot optimize dict type)
        smWidth = smoothingParams['width']

        # Choose smoothing function based on parameters
        match smoothingParams['win']:
            case 'gaussian':
                @nb.jit(nopython=True)
                def f(x):
                    return sum(gaussian(x, sig=smWidth)) / 1e-3

            case 'triangular':
                @nb.jit(nopython=True)
                def f(x):
                    return sum(triangular(x, halfwidth=smWidth / 2)) / (smWidth * 1e-3)

            case _:
                @nb.jit(nopython=True)
                def f(x):
                    return sum(rectangular(x, halfwidth=smWidth / 2)) / (smWidth * 1e-3)

        # Calculate time points for firing rate calculation
        step = smoothingParams['width'] * (1 - smoothingParams['overlap'])
        ps_T = np.arange(0, stats.mode(timeWindows.ptp(axis=1))[0], step)

        # Initialize array for storing firing rates
        ps_FR = np.zeros((timeWindows.shape[0], len(ps_T), len(spikeTimes)), dtype=np.float_)

        # Call the provided function for calculation
        return fcn(ps_FR, ps_T, spikeTimes, timeWindows, f)

    return wrapper


@peristim_firingrate_decorator
@nb.jit(nopython=True, parallel=True)
def peristim_firingrate(
        ps_FR, ps_T, spikeTimes, timeWindows, smoothingFcn) -> tuple[np.ndarray, np.ndarray]:
    """Calculate peristimulus firing rate."""

    for trl_n in nb.prange(len(timeWindows)):
        # loop over time steps
        for step_n, step in enumerate(ps_T):
            step += timeWindows[trl_n, 0]
            # loop over neurons (also use the corresponding timestamps for that neuron)
            for i, singleUnitSpikeTimes in enumerate(spikeTimes):
                # insert the firing rate for each neuron, at each time step, for each trial
                ps_FR[trl_n, step_n, i] = smoothingFcn(singleUnitSpikeTimes - step)
    return ps_FR, ps_T


def peristim_timestamp(
        spikeTimes: list[np.ndarray],
        timeWindows: np.ndarray) -> list[list[np.ndarray]]:
    """Organize spike timestamps based on given time windows."""

    # Ensure necessary inputs are provided
    assert spikeTimes is not None and timeWindows is not None, 'Both spikeTimes and timeWindow cannot be None'

    ps_TS = []
    # Loop over spike times
    for singleUnitSpikeTimes in spikeTimes:
        x = []
        # Loop over time windows
        for ti in timeWindows:
            # Filter spike times within each window
            x.append(singleUnitSpikeTimes[
                         (ti[0] <= singleUnitSpikeTimes)
                         & (singleUnitSpikeTimes <= ti[1])]
                     - ti[0])
        ps_TS.append(x)

    return ps_TS


if __name__ == '__main__':
    # Read data from .mat file
    matfile = mp.MATfile(r'G:\Vishnu\data\TMSTG\20180922\20180922_cfa_ipsi_L6_SLData_SpikeAmpTH3.0.mat')
    data = matfile.read()

    # Extract trigger times
    trigChanIdx = data['TrigChan_ind'][0, 0].astype(int) - 1
    refs = data['rawData/trigger'].flatten()
    trigger = data[refs[trigChanIdx]].flatten()[::2] * 1e3

    # Define time windows
    startT, endT = -20, 100
    timeWindows = trigger[:, np.newaxis] + np.array([startT, endT])

    # Extract spike times
    multiUnitSpikeTimes = data['SpikeModel/SpikeTimes/data'].flatten()
    refs = data['SpikeModel/ClusterAssignment/data'].flatten()
    allSpikeTimes_neuronIdx = [data[i].flatten().astype(int) - 1 for i in refs]
    singleUnitsSpikeTimes = nb.typed.List()
    [singleUnitsSpikeTimes.append(multiUnitSpikeTimes[idx]) for idx in allSpikeTimes_neuronIdx]

    # Organize spike timestamps based on time windows
    ps_TS = peristim_timestamp(singleUnitsSpikeTimes, timeWindows)

    # Calculate peristimulus firing rate with Gaussian smoothing
    ps_FR, ps_T = peristim_firingrate(singleUnitsSpikeTimes, timeWindows,
                                      smoothingParams={'win': 'gaussian', 'width': 3.0, 'overlap': 1 / 3})

    # Calculate peristimulus firing rate with another set of parameters
    ps_FR, ps_T = peristim_firingrate(singleUnitsSpikeTimes, timeWindows,
                                      smoothingParams={'win': 'gaussian', 'width': 3.0, 'overlap': 0.0})

    # Change thread settings
    # nb.set_num_threads(max(1, int(nb.config.NUMBA_NUM_THREADS // 1.25)))
    # nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
