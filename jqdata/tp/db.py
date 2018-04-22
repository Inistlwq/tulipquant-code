#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys
from .db_helper import create_database_objects

sys.modules[__name__].__dict__.update(create_database_objects())
