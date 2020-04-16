import numpy as np
import pandas as pd
import torch
import utils.misc as misc_utils

class Reporter(object):
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
        for name in self.report:
            tb_writer.add_scalar(name, self.report[name]['mean'], global_step=step)



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
        misc_utils.save_dict_to_json(report_mean_std, save_path)

    def save_df(self, save_path):
        method_column = ['DL'] * len(self.id_list)
        df_dict = {'Method': method_column, 'ID': self.id_list}
        for metric_name in self.report:
            df_dict[metric_name] = self.report[metric_name]['list'],

        df = pd.DataFrame(data=df_dict)
        df.to_pickle(save_path)
