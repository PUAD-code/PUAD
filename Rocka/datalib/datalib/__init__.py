# -*- coding: utf-8 -*-
import os
import sys

from flask import Config

from . import defconfig


#: The configuration for datalib
config = Config(os.path.abspath(os.path.split(__file__)[0]))
config.from_object(defconfig)


def _init_config():
    for path in [os.path.expanduser('~/datalibrc'),
                 os.path.join(os.path.split(__file__)[0], '../config.py'),
                 os.path.join(os.path.split(__file__)[0], '../../config.py')]:
        if os.path.exists(path):
            config.from_pyfile(path)
    del sys.modules[__name__]._init_config


# load other modules
from .loader import *
