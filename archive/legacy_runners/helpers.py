import numpy as np
import pandas as pd
import torch

import utils.experiment
import utils.misc
import utils.misc as misc_utils


class Reporter(object):
    """
    Collect and report values
        self.collect_value() collects value in `report_data_dict`, which is structured as:
            self.report_data_dict = {'value_name_A': [A1, A2, ...], ... }

        self.summarise() construct the report dictionary if called, which is structured as:
            self.report = {'value_name_A': {'mean': A_mean,
                                            'std': A_std,
                                            'list': [A1, A2, ...]}
                            }
    """
    def __init__(self):
        self.report_data_dict = {}
        self.report = {}
        self.id_list = []

    def reset(self):
        self.report_data_dict = {}
        self.report = {}
        self.id_list = []

    def collect_value(self, x):
        for name, value in x.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu()
            if name not in self.report_data_dict.keys():
                self.report_data_dict[name] = []
            self.report_data_dict[name].append(value)

    def summarise(self):
        for name in self.report_data_dict:
            self.report[name] = {
                'mean': np.mean(self.report_data_dict[name]),
                'std': np.std(self.report_data_dict[name]),
                'list': self.report_data_dict[name]
            }


class LossReporter(Reporter):
    def __init__(self):
        super(LossReporter, self).__init__()

    def log_to_tensorboard(self, tb_writer, step):
        for loss_name in self.report:
            tb_writer.add_scalar(f'losses/{loss_name}', self.report[loss_name]['mean'], global_step=step)


class MetricReporter(Reporter):
    def __init__(self):
        super(MetricReporter, self).__init__()

    def log_to_tensorboard(self, tb_writer, step):
        for metric_name in self.report:
            tb_writer.add_scalar(f'metrics/{metric_name}_mean', self.report[metric_name]['mean'], global_step=step)
            tb_writer.add_scalar(f'metrics/{metric_name}_std', self.report[metric_name]['std'], global_step=step)

    def save_mean_std(self, save_path):
        report_mean_std = {}
        for metric_name in self.report:
            report_mean_std[metric_name + '_mean'] = self.report[metric_name]['mean']
            report_mean_std[metric_name + '_std'] = self.report[metric_name]['std']
        utils.misc.save_dict_to_json(report_mean_std, save_path)

    def save_df(self, save_path, model_name):
        method_column = [str(model_name)] * len(self.id_list)
        df_dict = {'Method': method_column, 'ID': self.id_list}
        for metric_name in self.report:
            df_dict[metric_name] = self.report[metric_name]['list']

        df = pd.DataFrame(data=df_dict)
        df.to_pickle(save_path)
