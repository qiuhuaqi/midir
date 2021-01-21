import logging
import os

import numpy as np
import pandas as pd


class MetricReporter(object):
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
    def __init__(self, id_list, save_dir, save_name='analysis_results'):
        self.id_list = id_list
        self.save_dir = save_dir
        self.save_name = save_name

        self.report_data_dict = {}
        self.report = {}

    def reset(self):
        self.report_data_dict = {}
        self.report = {}

    def collect(self, x):
        for name, value in x.items():
            if name not in self.report_data_dict.keys():
                self.report_data_dict[name] = []
            self.report_data_dict[name].append(value)

    def summarise(self):
        # summarise aggregated results to form the report dict
        for name in self.report_data_dict:
            self.report[name] = {
                'mean': np.mean(self.report_data_dict[name]),
                'std': np.std(self.report_data_dict[name]),
                'list': self.report_data_dict[name]
            }

    def save_mean_std(self):
        report_mean_std = {}
        for metric_name in self.report:
            report_mean_std[metric_name + '_mean'] = self.report[metric_name]['mean']
            report_mean_std[metric_name + '_std'] = self.report[metric_name]['std']
        # save to CSV
        csv_path = self.save_dir + f'/{self.save_name}.csv'
        save_dict_to_csv(report_mean_std, csv_path)

    def save_df(self):
        # method_column = [str(model_name)] * len(self.id_list)
        # df_dict = {'Method': method_column, 'ID': self.id_list}
        df_dict = {'ID': self.id_list}
        for metric_name in self.report:
            df_dict[metric_name] = self.report[metric_name]['list']

        df = pd.DataFrame(data=df_dict)
        df.to_pickle(self.save_dir + f'/{self.save_name}_df.pkl')


def save_dict_to_csv(d, csv_path, model_name='modelX'):
    for k, x in d.items():
        if not isinstance(x, list):
            d[k] = [x]
    pd.DataFrame(d, index=[model_name]).to_csv(csv_path)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        if os.path.exists(log_path):
            print("Logger already exists. Overwritting.")
            os.system("mv -f {} {}.backup".format(log_path, log_path))

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(stream_handler)

