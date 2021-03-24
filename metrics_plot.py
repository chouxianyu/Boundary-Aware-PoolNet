import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文


class Model:
    def __init__(self, method, metrics_path, color, linestyle, linewidth):
        self.method = method
        self.metrics_path = metrics_path
        self.color = color
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.recalls = None
        self.precisions = None
        self.F_measures = None
        self._load_metrics()
        
    def _load_metrics(self):
        self.recalls = np.loadtxt(os.path.join(self.metrics_path, 'recalls.csv'), delimiter=',', dtype='float')
        self.precisions = np.loadtxt(os.path.join(self.metrics_path, 'precisions.csv'), delimiter=',', dtype='float')
        self.F_measures = np.loadtxt(os.path.join(self.metrics_path, 'F_measures.csv'), delimiter=',', dtype='float')


class Plotter:
    def __init__(self, models, output_path='./eval_results/figures'):
        self.models = models
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self._plot_precision_recall_curve()
        self._plot_F_measure()

    def _plot_precision_recall_curve(self):
        # PR曲线
        plt.figure(dpi=200)
        for model in self.models:
            plt.plot(model.recalls, model.precisions, color=model.color,
                     label=model.method, linestyle=model.linestyle, linewidth=model.linewidth)
        plt.xlim(0, 1)
        plt.xticks(np.linspace(0, 1, 11))
        plt.ylim(0, 1)
        plt.yticks(np.linspace(0, 1, 11))
        plt.title('Precision-Recall 曲线')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid('minor', linestyle='-', linewidth=0.5, alpha=0.2)
        plt.legend()
        plt.savefig(os.path.join(self.output_path, 'PR曲线.png'))
        print('PR曲线.png Saved in', self.output_path)

    def _plot_F_measure(self):
        # F-measure
        plt.figure(dpi=200)
        x = range(256)
        for model in self.models:
            plt.plot(x, model.F_measures, color=model.color, label=model.method,
                     linestyle=model.linestyle, linewidth=model.linewidth)
        plt.xlim(0, 255)
        plt.ylim(0, 1)
        plt.xticks(range(0, 256, 25))
        plt.yticks(np.linspace(0, 1, 11))
        plt.title('F-measure 曲线')
        plt.xlabel('Threshold')
        plt.ylabel('F-measure')
        plt.grid('minor', linestyle='-', linewidth=0.5, alpha=0.2)
        plt.legend()
        plt.savefig(os.path.join(self.output_path, 'F-measure曲线.png'))
        print('F-measure曲线.png Saved in', self.output_path)


def main():
    methods = ['CapSal', 'PiCANet', 'DGRL', 'BASNet', 'U2Net', 'CPD', 'PoolNet', 'BAPoolNet']
    colors = ['seagreen', 'limegreen', 'steelblue', 'blue', 'darkviolet', 'violet', 'deeppink', 'red']
    linestyles = ['-.', '-.',  '-.', '--', '--', '-', '-', '-']
    linewidths = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 1]
    models = [Model(methods[i], os.path.join('./eval_results', methods[i]),
                    colors[i], linestyles[i], linewidths[i]) for i in range(8)]
    Plotter(models)


if __name__ == '__main__':
    main()
