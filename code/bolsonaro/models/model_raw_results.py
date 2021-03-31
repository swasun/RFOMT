from bolsonaro.utils import save_obj_to_pickle, load_obj_from_pickle

import os
import datetime


class ModelRawResults(object):

    def __init__(self, model_weights, training_time,
        datetime, train_score, dev_score, test_score,
        train_score_base, dev_score_base,
        test_score_base, score_metric, base_score_metric,
        train_coherence='', dev_coherence='', test_coherence='',
        train_correlation='', dev_correlation='', test_correlation='',
        train_scores='', dev_scores='', test_scores='',
        train_strength='', dev_strength='', test_strength=''):

        self._model_weights = model_weights
        self._training_time = training_time
        self._datetime = datetime
        self._train_score = train_score
        self._dev_score = dev_score
        self._test_score = test_score
        self._train_score_base = train_score_base
        self._dev_score_base = dev_score_base
        self._test_score_base = test_score_base
        self._score_metric = score_metric
        self._base_score_metric = base_score_metric
        self._train_coherence = train_coherence
        self._dev_coherence = dev_coherence
        self._test_coherence = test_coherence
        self._train_correlation = train_correlation
        self._dev_correlation = dev_correlation
        self._test_correlation = test_correlation
        self._train_scores = train_scores
        self._dev_scores = dev_scores
        self._test_scores = test_scores
        self._train_strength = train_strength
        self._dev_strength = dev_strength
        self._test_strength = test_strength

    @property
    def model_weights(self):
        return self._model_weights

    @property
    def training_time(self):
        return self._training_time

    @property
    def datetime(self):
        return self._datetime

    @property
    def train_score(self):
        return self._train_score

    @property
    def dev_score(self):
        return self._dev_score

    @property
    def test_score(self):
        return self._test_score

    @property
    def train_score_base(self):
        return self._train_score_base

    @property
    def dev_score_base(self):
        return self._dev_score_base

    @property
    def test_score_base(self):
        return self._test_score_base

    @property
    def score_metric(self):
        return self._score_metric

    @property
    def base_score_metric(self):
        return self._base_score_metric

    @property
    def train_coherence(self):
        return self._train_coherence

    @property
    def dev_coherence(self):
        return self._dev_coherence

    @property
    def test_coherence(self):
        return self._test_coherence

    @property
    def train_correlation(self):
        return self._train_correlation

    @property
    def dev_correlation(self):
        return self._dev_correlation

    @property
    def test_correlation(self):
        return self._test_correlation

    @property
    def train_scores(self):
        return self._train_scores

    @property
    def dev_scores(self):
        return self._dev_scores

    @property
    def test_scores(self):
        return self._test_scores

    @property
    def train_strength(self):
        return self._train_strength

    @property
    def dev_strength(self):
        return self._dev_strength

    @property
    def test_strength(self):
        return self._test_strength

    def save(self, models_dir):
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)
        save_obj_to_pickle(models_dir + os.sep + 'model_raw_results.pickle',
            self.__dict__)

    @staticmethod
    def load(models_dir):
        return load_obj_from_pickle(models_dir + os.sep + 'model_raw_results.pickle',
            ModelRawResults)
