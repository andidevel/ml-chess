import json
import h5py

from pathlib import Path


class DataConfig(object):

    def __init__(self, config_file):
        with open(config_file, 'r') as fp:
            self.config = json.load(fp)
        for k in self.config.keys():
            setattr(self, k, self.config[k])
        self.data_path = Path(self.data_folder)
        self.h5data = h5py.File(str(self.data_path.joinpath(self.data_filename)), 'r')

    def _get_dataset_name(self, type_data):
        for ds in self.datasets:
            # Return the first dataset name that match type_data
            if ds['type'] == type_data:
                return ds['name']
        return ''

    def get_train(self, n=None, copy=True):
        data_ds_name = self._get_dataset_name('train')
        labels_ds_name = self._get_dataset_name('labels_train')
        # Return X_train, Y_train
        if copy:
            if n is None:
                return (self.h5data[data_ds_name][:], self.h5data[labels_ds_name][:])
            else:
                return (self.h5data[data_ds_name][:n], self.h5data[labels_ds_name][:n])
        return (self.h5data[data_ds_name], self.h5data[labels_ds_name])

    def get_test(self, n=None, copy=True):
        data_ds_name = self._get_dataset_name('test')
        labels_ds_name = self._get_dataset_name('labels_test')
        # Return X_test, Y_test
        if copy:
            if n is None:
                return (self.h5data[data_ds_name][:], self.h5data[labels_ds_name][:])
            else:
                return (self.h5data[data_ds_name][:n], self.h5data[labels_ds_name][:n])
        return (self.h5data[data_ds_name], self.h5data[labels_ds_name])

    def close_data_file(self):
        self.h5data.close()
