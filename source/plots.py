import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from .drift_detection import get_metrics
import matplotlib
    
def plot_drift_vertical_lines(tamanho, resp_drift=None, label="Concept drift ground truth", lw=3, alpha=0.9):
    plt.rcParams["font.family"] = "Times New Roman"
    first=True
    
    if resp_drift is None:
        resp_drift = int(tamanho * 0.1)
        
    for i in range(resp_drift, tamanho, resp_drift):
        if first:
            first=False
            plt.axvline(x=i, ls='--', lw=lw, c='darkgreen', label=label, alpha=0.9)
        else:
            plt.axvline(x=i, ls='--', lw=lw, c='darkgreen', alpha=0.9)

def plot_deteccao_drift(
    df, col, detected_drifts, y_true, rolling_means, lowers, uppers, cluster_window_size=50, save_png=""
):
    if save_png != "":
        plt.ioff()
    
    plt.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=(18, 4))    
    ax = plt.gca()
    ax.plot(df.index, df[col], c='#ff5f54', lw=5, label=col)
    
    #ax.plot(df.index, rolling_means, c='#50A898', linestyle='-', lw=2, marker='.', markeredgewidth=3, label="média móvel")
    #ax.fill_between(df.index, lowers, uppers, facecolor='#52adff', alpha=0.1, label="tolerância do desvio padrão")

    ax.plot(df.index, rolling_means, c='#35b588', linestyle='-', lw=4, marker='.', markeredgewidth=4, label="Rolling average")
    ax.fill_between(df.index, lowers, uppers, facecolor='#52adff', alpha=0.1, label="Tolerance boundaries")
    ax.plot(df.index, uppers, c='#52adff', alpha=0.5, marker='v', markeredgewidth=4)
    ax.plot(df.index, lowers, c='#52adff', alpha=0.5, marker='^', markeredgewidth=4)

    first=True
    for val in y_true:
        if first:
            first=False
            #ax.axvline(x=val, ls='--', lw=3, c='darkgreen', alpha=0.5, label="$\it{Concept}$ $\it{drift}$ verdadeiro")
            ax.axvline(x=val, ls='--', lw=4, c='darkgreen', alpha=0.8, label="True concept drift")
        else:
            ax.axvline(x=val, ls='--', lw=4, c='darkgreen', alpha=0.8)

    first=True
    for val in detected_drifts:
        if first:
            first=False
            #ax.axvline(x=val, ls='-', lw=2, c='#e8d690', label="$\it{Concept}$ $\it{drift}$ predito")
            ax.axvline(x=val, ls='-.', lw=4, c='#deb100', alpha=0.8, label="Detected concept drift")
        else:
            ax.axvline(x=val, ls='-.', lw=4, c='#deb100', alpha=0.8)

    metrics = get_metrics(detected_drifts, y_true, cluster_window_size)
    # print(metrics)
    # plt.title("Precision: {}  Recall: {}  F1: {}  Delay:{}".format(
    #     "{0:.2f}%".format(metrics["Precision"]* 100),
    #     "{0:.2f}%".format(metrics["Recall"]* 100),
    #     "{0:.2f}%".format(metrics["F1"]* 100),
    #     "{0:.2f}".format(metrics["Delay"])
    # ))
            
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(30)

    # ax.set_xlabel('índice dos $\it{traces}$')
    # ax.set_xlabel('Trace index')
    # ax.set_xlim((0, df.index.max() + 100))
    # bbox_to_anchor=(0.5, 1.3),
    leg = plt.legend(fontsize=32, loc='upper center', bbox_to_anchor=(0.48, -0.15),
              fancybox=True, shadow=False, prop={"family":"Times New Roman", "size":"26"},
              frameon=False, ncol=3, labelspacing=0.25, columnspacing=1)

    for line in leg.get_lines():
        line.set_linewidth(5)

    # print(plt.yticks())
    
    if save_png != "":
        plt.savefig(save_png, dpi=100, transparent=False)
        plt.close(plt.gcf())
        plt.ion()