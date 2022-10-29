""" Parse configuration file """

import os

import yaml

from tools.utils import makepath


class Config(dict):
    def __init__(self, default_cfg_path=None, **kwargs):
        """ If a config .yaml file exists in default_cfg_path it will be loaded """
        if default_cfg_path is not None and os.path.exists(default_cfg_path):
            loaded_config = self.load_cfg(default_cfg_path)
            super(Config, self).__init__(**loaded_config)
        else:
            super(Config, self).__init__(**kwargs)

    @staticmethod
    def load_cfg(load_path):
        """ Loads a config .yaml file from the given location """
        with open(load_path, 'r') as infile:
            loaded_config = yaml.safe_load(infile)
        return loaded_config if loaded_config is not None else {}

    def write_cfg(self, write_path=None):
        """ Writes a config .yaml file to the given location """
        if write_path is None:
            write_path = 'configuration.yaml'

        dump_dict = {k: v for k, v in self.items() if k != 'default_cfg'}
        makepath(write_path, isfile=True)
        with open(write_path, 'w') as outfile:
            yaml.safe_dump(dump_dict, outfile, default_flow_style=False)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
