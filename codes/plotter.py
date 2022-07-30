import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Create Directory and Add Science Styles
plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2
os.makedirs('plots', exist_ok=True)


def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_actual_predicted(name, y_true, y_pred):
    try:
        os.makedirs(os.path.join('plots', name), exist_ok=True)
        # TODO: Add Column Names for Each Plot
        if y_true.shape == y_pred.shape:
            pdf = PdfPages(f'plots/{name}/active_v_predicted.pdf')
            for dim in range(y_true.shape[1]):
                fig, ax = plt.subplots()
                y_t, y_p = y_true[:, dim], y_pred[:, dim]
                ax.plot(smooth(y_t), label='True')
                ax.plot(smooth(y_p), label='Predicted')
                pdf.savefig(fig)
                plt.close()
            pdf.close()
        else:
            print(
                f'Cannot draw figure of difference shape where y_true shape = {y_true.shape[1]}, '
                f'y_pred shape = {y_pred.shape[1]}.')
    except BaseException as base:
        print(f'Error {base}')


def loss_and_threshold(name, loss, threshold):
    os.makedirs(os.path.join('plots', name), exist_ok=True)
    pdf = PdfPages(f'plots/{name}/output.pdf')
    fig, ax = plt.subplots()
    ax.figure(figsize=(20, 6))
    ax.plot(loss, label='Loss')
    ax.axhline(y=threshold, color='r', label='Threshold')
    ax.xlabel('Test Size')
    ax.ylabel('Loss')
    ax.legend(loc = 'upper right')
    ax.show()
    pdf.savefig(fig)
    
    
def loss_threshold_and_anomalies(name, loss, threshold):
    os.makedirs(os.path.join('plots', name), exist_ok=True)
    pdf = PdfPages(f'plots/{name}/anomalies.pdf')
    fig, ax = plt.subplots()
    ax.plot(loss['loss'], label='Loss')
    ax.axhline(y=threshold, color='orange', label='Threshold')
    ax.scatter(x=loss['index'], y=loss['anomaly'], label='Anomaly', color='red')
    ax.set_xlabel('Test Size')
    ax.set_ylabel('Loss')
    ax.legend(loc = 'upper right')
    pdf.savefig(fig)
    plt.close()
    pdf.close()

    
def plot_accuracies(accuracy_list, folder):
    os.makedirs(f'plots/{folder}/', exist_ok=True)
    trainAcc = [i[0] for i in accuracy_list]
    lrs = [i[1] for i in accuracy_list]
    plt.xlabel('Epochs')
    plt.ylabel('Average Training Loss')
    plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
    plt.twinx()
    plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', linewidth=1, linestyle='--', marker='.')
    plt.savefig(f'plots/{folder}/training-graph.pdf')
    plt.clf()
    plt.close()
    

def plot_losses(accuracy_list, folder):
    os.makedirs(f'plots/{folder}/', exist_ok=True)
    trainAcc = [i[0] for i in accuracy_list]
    plt.xlabel('Epochs')
    plt.ylabel('Average Training Loss')
    plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
    plt.twinx()
    plt.savefig(f'plots/{folder}/loss_graph.pdf')
    plt.clf()
    plt.close()
    
    
def plot_confusion_matrix(name, matrix):
    os.makedirs(os.path.join('plots', name), exist_ok=True)
    group_counts = ['{0:0.0f}'.format(value) for value in matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in matrix.flatten() / np.sum(matrix)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(matrix.shape)
    ax = sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    ax.yaxis.set_ticklabels(['Anomaly', 'Normal'])
    ax.xaxis.set_ticklabels(['Anomaly', 'Normal'])
    ax.get_figure().savefig(f'plots/{name}/confusion-matrix.png')
    
    
def plot_confusion_matrix_fl(folder, title, matrix):
    os.makedirs(os.path.join('plots', folder), exist_ok=True)
    group_counts = ['{0:0.0f}'.format(value) for value in matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in matrix.flatten() / np.sum(matrix)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(matrix.shape)
    ax = sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    ax.yaxis.set_ticklabels(['Anomaly', 'Normal'])
    ax.xaxis.set_ticklabels(['Anomaly', 'Normal'])
    ax.get_figure().savefig(f'plots/{folder}/confusion_matrix_{title}.png')
    plt.close()
    

def plotter(name, y_true, y_pred, ascore, labels):
	os.makedirs(os.path.join('plots', name), exist_ok=True)
	pdf = PdfPages(f'plots/{name}/output.pdf')
	for dim in range(y_true.shape[1]):
		y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim]
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True);
		ax1.set_ylabel('Value')
		ax1.set_title(f'Dimension = {dim}')
		ax1.plot(smooth(y_t), linewidth=0.2, label='True')
		ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
		ax3 = ax1.twinx()
		ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
		ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
		if dim == 0: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
		ax2.plot(smooth(a_s), linewidth=0.2, color='g')
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel('Anomaly Score')
		pdf.savefig(fig)
		plt.close()
	pdf.close()
 
 
def anomaly(name, ascore, threshold):
    os.makedirs(os.path.join('plots', name), exist_ok=True)
    pdf = PdfPages(f'plots/{name}/anomaly.pdf')
    for dim in range(ascore.shape[1]):
        a_s = ascore[:, dim]
        fig, ax = plt.subplots(1, 1, sharex=True)
        ax.plot(a_s, linewidth=0.2, color='g')
        ax.set_xlabel('Test Size')
        ax.set_ylabel('Anomaly Score')
        ax.axhline(threshold, linestyle='--', color='r')
        pdf.savefig(fig)
        plt.close()
    pdf.close()
    
 
def anomalies(folder, name, ascore, threshold):
    os.makedirs(os.path.join('plots', folder), exist_ok=True)
    pdf = PdfPages(f'plots/{folder}/{name}.pdf')
    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(ascore, linewidth=0.2, color='g')
    ax.set_xlabel('Test Size')
    ax.set_ylabel('Anomaly Score')
    ax.axhline(threshold, linestyle='--', color='r')
    pdf.savefig(fig)
    plt.close()
    pdf.close()
 
def plot_roc_auc_curve(models, fprs, tprs):
    os.makedirs(os.path.join('plots'), exist_ok=True)
    pdf = PdfPages(f'plots/roc_auc_curve.pdf')
    fig, ax = plt.subplots()
    for i in range(len(models)):
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC Curve')
        ax.plot(fprs[i], tprs[i], label=models[i])
        ax.legend(loc='lower right')
    pdf.savefig(fig)
    plt.close()
    pdf.close()


