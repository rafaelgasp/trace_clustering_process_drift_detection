import matplotlib.pyplot as plt


def plot_drift_vertical_lines(log_size, resp_drift=None, label="Concept drift ground truth", lw=3, alpha=0.9):
    """
        Plots vertical lines corresponding to ground truth drifts

        Parameters:
        ------------
            log_size (int): Size of the event log to plot a drift at every 10%
            resp_drift (int/None): Plot a vertical line at every 'resp_drift' traces
            label (str): Label to write in the plot
            lw (int): Line width
            alpha (float): Line color transparency
    """
    plt.rcParams["font.family"] = "Times New Roman"
    first=True
    
    if resp_drift is None:
        resp_drift = int(log_size * 0.1)
        
    for i in range(resp_drift, log_size, resp_drift):
        if first:
            first=False
            plt.axvline(x=i, ls='--', lw=lw, c='darkgreen', label=label, alpha=alpha)
        else:
            plt.axvline(x=i, ls='--', lw=lw, c='darkgreen', alpha=alpha)


def plot_deteccao_drift(
    run_df, col, detected_drifts, y_true, rolling_means, lowers, uppers, save_png=""
):
    """
        Plots the execution of the drift detection method with the tolerance 
        boundaries and the rolling mean used to detect a drift.

        Parameters:
        ------------
            run_df (pd.DataFrame): Result of the trace clustering step, with the
                values from the features of tracking the trace clustering evolution
            col (str): Column in 'run_df' to be considered in the analysis 
            detected_drifts (list): List of index of detected drifts
            y_true (list): List of index of ground truth drifts
            rolling_means (list): Rolling average of values of 'col'
            lowers (list): List of lower tolerance boundaries over the traces
            uppers (list): List of uppers tolerance boundaries over the traces
            save_png (str): Name of png file. If == "" does not save as file
    """
    if save_png != "":
        plt.ioff()
    
    plt.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=(18, 4))    
    ax = plt.gca()
    ax.plot(run_df.index, run_df[col], c='#ff5f54', lw=5, label=col)
    
    ax.plot(run_df.index, rolling_means, c='#35b588', linestyle='-', lw=4, marker='.', markeredgewidth=4, label="Rolling average")
    ax.fill_between(run_df.index, lowers, uppers, facecolor='#52adff', alpha=0.1, label="Tolerance boundaries")
    ax.plot(run_df.index, uppers, c='#52adff', alpha=0.5, marker='v', markeredgewidth=4)
    ax.plot(run_df.index, lowers, c='#52adff', alpha=0.5, marker='^', markeredgewidth=4)

    first=True
    for val in y_true:
        if first:
            first=False
            ax.axvline(x=val, ls='--', lw=4, c='darkgreen', alpha=0.8, label="True concept drift")
        else:
            ax.axvline(x=val, ls='--', lw=4, c='darkgreen', alpha=0.8)

    first=True
    for val in detected_drifts:
        if first:
            first=False
            ax.axvline(x=val, ls='-.', lw=4, c='#deb100', alpha=0.8, label="Detected concept drift")
        else:
            ax.axvline(x=val, ls='-.', lw=4, c='#deb100', alpha=0.8)
            
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(30)


    leg = plt.legend(fontsize=32, loc='upper center', bbox_to_anchor=(0.48, -0.15),
              fancybox=True, shadow=False, prop={"family":"Times New Roman", "size":"26"},
              frameon=False, ncol=3, labelspacing=0.25, columnspacing=1)

    for line in leg.get_lines():
        line.set_linewidth(5)
    
    if save_png != "":
        plt.savefig(save_png, dpi=100, transparent=False)
        plt.close(plt.gcf())
        plt.ion()