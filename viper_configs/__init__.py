"""
Adapted From: https://github.com/cvlab-columbia/viper
"""

import os
from omegaconf import OmegaConf

# The default
config_names = os.getenv('CONFIG_NAMES', None)
if config_names is None:
    config_names = 'config_codellama'  # Modify this if you want to use another default config

configs = [OmegaConf.load('./viper_configs/base_config.yaml')]
if config_names is not None:
    for config_name in config_names.split(','):
        configs.append(OmegaConf.load(f'./viper_configs/{config_name.strip()}.yaml'))

# unsafe_merge makes the individual configs unusable, but it is faster
viper_config = OmegaConf.unsafe_merge(*configs)

