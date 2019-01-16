#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: coelhorp
"""

import numpy as np
import pandas as pd

import power_means

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances

import moabb
from moabb.paradigms import MotorImagery
from moabb.datasets import BNCI2014001
from moabb.evaluations import WithinSessionEvaluation

from sklearn.pipeline import make_pipeline

moabb.set_log_level('info')

# define the pipelines for classification -- MDM and MeansField classifier
pipelines = {}

pipelines['MDM'] = make_pipeline(Covariances(estimator='lwf'),
                                 MDM())

plist = [1.000, 0.750, 0.500, 0.250, 0.100, 0.010, -0.010, -0.100, -0.250, -0.500, -0.750, -1.000]
pipelines['MeansField'] = make_pipeline(Covariances(estimator='lwf'),
                                        power_means.MeanFieldClassifier(plist=plist, meth_label='inf_means'))

paradigm = MotorImagery()
datasets = BNCI2014001()
overwrite = True  # set to True if we want to overwrite cached results

evaluation = WithinSessionEvaluation(paradigm=paradigm,
                                     datasets=datasets,
                                     suffix='examples', overwrite=overwrite)

results = evaluation.process(pipelines)
results.to_pickle('results_MotorImagery.pkl')
