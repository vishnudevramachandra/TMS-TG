import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from figures.helper_figs import gp_extractor, pkFr_extr, delay_extr, violin_fill_false


def plot_inj_effect(tms, df, activeNeus, manipulation, postTi=None, savefig=False):
    """
        Plot the population average firing rates and the latency of late activity component.

    Args:
        tms: Instance of TMS class.
        df: filtered blocksinfo.
        activeNeus: Active neurons determined by statistical test
        manipulation: column name corresponding to the experimental manipulation done.
        postTi: List of time intervals (in mins)
    Parameters:
        savefig: Whether to save the figure or not

    Returns:
        None
    """

    # Set parameters that are not passed in.
    if postTi is None:
        postTi = [(0, 10), (10, 20), (20, 30), (30, 40), (40, np.inf)]

    # Rearrange peak firing rate
    pkFr = df.groupby(by=df.index.names).apply(
        gp_extractor, pkFr_extr, activeNeus, manipulation, 'psActivity', postTi, tms)
    pkFr = pkFr.melt(id_vars=['MT'], value_vars=[item for item in pkFr.columns if item != 'MT'],
                     var_name='Time elapsed (min)', value_name='Peak firing rate (Hz)')

    # Rearrange delay
    delay = df.groupby(by=df.index.names).apply(
        gp_extractor, delay_extr, activeNeus, manipulation, 'psActivity', postTi, tms)
    delay = delay.melt(id_vars=['MT'], value_vars=[item for item in delay.columns if item != 'MT'],
                       var_name='Time elapsed (min)', value_name='Delay (ms)')

    # Plot settings
    plt.style.use('default')
    fig, axes = plt.subplots(2, 1)

    # Plot late activity component's peak firing rate
    sns.lineplot(data=pkFr,
                 x='Time elapsed (min)', y='Peak firing rate (Hz)', hue='MT', style='MT',
                 markers=True, dashes=False, palette='deep', ax=axes[0])
    axes[0].set_title(manipulation, fontsize=12)

    # plot the latency of late activity component
    sns.swarmplot(data=delay,
                  x='Time elapsed (min)', y='Delay (ms)', hue='MT', dodge=True, palette='deep',
                  size=3, legend=False, ax=axes[1])
    sns.violinplot(data=delay,
                   x='Time elapsed (min)', y='Delay (ms)', hue='MT', inner=None,
                   cut=1, bw_adjust=0.5, ax=axes[1])
    violin_fill_false(axes[1])

    # Save and display the plot
    if savefig:
        plt.savefig(manipulation + '.pdf', dpi='figure', format='pdf')
    plt.show()