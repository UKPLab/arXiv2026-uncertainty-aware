import copy
import os
import shutil
import subprocess as sp
import copy
import json

from sisyphus import *

import i6_core.util as util
from i6_core.returnn.config import instanciate_delayed

Path = setup_path(__package__)
    
class HuggingfaceSearchJob(Job):
  """
  Train a Huggingface transformer model
  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=1,
      gpu_rqmt=1,
      python_exe=None,
      sbatch_args=None,
      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.model_path = model_path
    self.config = config
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "gpumem": gpumem,
      "sbatch_args": sbatch_args,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_metric_file = self.output_path("metrics.json")
    self.out_search_file = self.output_path("search_output.json")
    self.out_checkpoints_dir = self.output_path("checkpoints", directory=True)
    self.out_cache_dir = self.output_path("cache", directory=True)

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'metric_output_file': self.out_metric_file,
      'prediction_output_file': self.out_search_file,
      'output_dir': self.out_checkpoints_dir.get_path(),
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    # self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
        #   "--nproc_per_node=1",
          os.path.join(tk.uncached_path(self.code_root), "predict.py"),
          self.out_config_file.get_path(),
          self.out_cache_dir.get_path()
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)


class CalibrateThresholdJob(Job):
  """
  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      hyps,
      search_data_config,
      scores,
      n,
      utility_metric,
      metric_threshold=None,
      src=None,
      time_rqmt=24,
      mem_rqmt=32,
      cpu_rqmt=1,
      gpu_rqmt=1,
      gpumem=12,
      sbatch_args=None,
      keep_only_best=False,
      distributed=False,
      **kwargs,
  ):
    """
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """
    self.code_root = code_root
    self.hyps = hyps
    self.metric_threshold = metric_threshold
    self.n = n
    self.scores = scores
    self.search_data_config = search_data_config
    self.src = src
    self.utility_metric = utility_metric

    self.distributed = distributed

    self.python_exe = gs.PYTHON_EXE

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "sbatch_args": sbatch_args,
    }

    if gpu_rqmt > 0:
        self.rqmt["gpumem"] = gpumem

    self.out_config_file = self.output_path("search_data_cofig.json", directory=False)
    self.out_threshold_file = self.output_path("thresholds.json", directory=False)

  def _get_run_cmd(self):
    run_cmd = [
          tk.uncached_path(self.python_exe),
        #   "--nproc_per_node=1",
          os.path.join(tk.uncached_path(self.code_root), "emnlp", "mbr", "thresholding.py"),
        self.hyps.get_path(),
        "-n", str(self.n),
        # "--eval-metric", "bleu",
        "--scores", self.scores.get_path(),
        "--out-threshold-file", self.out_threshold_file.get_path(),
        '--search_data_config_path', self.out_config_file.get_path(),
        '--utility_metric', self.utility_metric
    ]
    if self.src is not None:
      run_cmd.extend([
          "--src", self.src
      ])
    if self.metric_threshold is not None:
      run_cmd.extend([
          "--metric_threshold", str(self.metric_threshold)
      ])
    return run_cmd

  def create_files(self):
    with util.uopen(self.out_config_file, 'wt') as fp:
        json.dump(self.search_data_config, fp)
    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      if 'kwargs' in hash_kwargs and (hash_kwargs['kwargs'] is None or len(hash_kwargs['kwargs']) == 0):
        del hash_kwargs['kwargs']

      return super().hash(hash_kwargs)


 
class QualityEstimationReviewsJob(Job):
  """
  Train a Huggingface transformer model
  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=1,
      gpu_rqmt=1,
      python_exe=None,
      sbatch_args=None,
      review_file=None,
      review_score_file=None,
      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.model_path = model_path
    self.review_file = review_file
    self.review_score_file = review_score_file
    self.config = config
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "gpumem": gpumem,
      "sbatch_args": sbatch_args,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_metric_file = self.output_path("metrics.json")
    self.out_search_file = self.output_path("search_output.json")
    self.out_checkpoints_dir = self.output_path("checkpoints", directory=True)
    self.out_cache_dir = self.output_path("cache", directory=True)

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'metric_output_file': self.out_metric_file,
      'prediction_output_file': self.out_search_file,
      'output_dir': self.out_checkpoints_dir.get_path(),
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    self.config['method'] = "review_score_consistency"
    self.config['reviews_file'] = self.review_file
    self.config['scores_file'] = self.review_score_file
    # self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
        #   "--nproc_per_node=1",
          os.path.join(tk.uncached_path(self.code_root), "predict.py"),
          self.out_config_file.get_path(),
          self.out_cache_dir.get_path()
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)









class ReviewMBRJob(Job):
  """
  Train a Huggingface transformer model
  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=1,
      gpu_rqmt=1,
      python_exe=None,
      sbatch_args=None,
      review_file=None,
      quality_file=None,
      review_score_file=None,

      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.model_path = model_path
    self.review_file = review_file
    self.review_score_file = review_score_file
    self.quality_file = quality_file
    self.config = config
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "gpumem": gpumem,
      "sbatch_args": sbatch_args,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_metric_file = self.output_path("metrics.json")
    self.out_search_file = self.output_path("search_output.json")
    self.out_checkpoints_dir = self.output_path("checkpoints", directory=True)
    self.out_cache_dir = self.output_path("cache", directory=True)

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'metric_output_file': self.out_metric_file,
      'prediction_output_file': self.out_search_file,
      'output_dir': self.out_checkpoints_dir.get_path(),
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    self.config['method'] = "review_mbr"
    self.config['reviews_file'] = self.review_file
    self.config['scores_file'] = self.review_score_file
    self.config['quality_file'] = self.quality_file
    # self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
        #   "--nproc_per_node=1",
          os.path.join(tk.uncached_path(self.code_root), "predict.py"),
          self.out_config_file.get_path(),
          self.out_cache_dir.get_path()
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)








class ReviewEvaluationJob(Job):
  """
  Train a Huggingface transformer model
  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=1,
      gpu_rqmt=1,
      python_exe=None,
      sbatch_args=None,
      review_file=None,
      quality_file=None,
      mbr_file=None,
      review_score_file=None,

      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.model_path = model_path
    self.review_file = review_file
    self.review_score_file = review_score_file
    self.quality_file = quality_file
    self.config = config
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "gpumem": gpumem,
      "sbatch_args": sbatch_args,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_metric_file = self.output_path("metrics.json")
    self.out_search_file = self.output_path("search_output.json")
    self.mbr_file = mbr_file
    self.out_checkpoints_dir = self.output_path("checkpoints", directory=True)
    self.out_cache_dir = self.output_path("cache", directory=True)

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'metric_output_file': self.out_metric_file,
      'prediction_output_file': self.out_search_file,
      'output_dir': self.out_checkpoints_dir.get_path(),
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    self.config['method'] = "eval_reviews"
    self.config['reviews_file'] = self.review_file
    self.config['scores_file'] = self.review_score_file
    self.config['quality_file'] = self.quality_file
    self.config["mbr_file"] = self.mbr_file
    # self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
        #   "--nproc_per_node=1",
          os.path.join(tk.uncached_path(self.code_root), "predict.py"),
          self.out_config_file.get_path(),
          self.out_cache_dir.get_path()
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)




class MathdialQualityJob(Job):
  """
  Train a Huggingface transformer model
  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=1,
      gpu_rqmt=1,
      python_exe=None,
      sbatch_args=None,
      responses_file=None,
      acts_file=None,
      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.model_path = model_path
    self.review_file = responses_file
    self.review_score_file = acts_file
    self.config = config
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "gpumem": gpumem,
      "sbatch_args": sbatch_args,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_metric_file = self.output_path("metrics.json")
    self.out_search_file = self.output_path("search_output.json")
    self.out_checkpoints_dir = self.output_path("checkpoints", directory=True)
    self.out_cache_dir = self.output_path("cache", directory=True)

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'metric_output_file': self.out_metric_file,
      'prediction_output_file': self.out_search_file,
      'output_dir': self.out_checkpoints_dir.get_path(),
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    self.config['method'] = "mathdial_quality"
    self.config['reviews_file'] = self.review_file
    self.config['scores_file'] = self.review_score_file
    # self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
        #   "--nproc_per_node=1",
          os.path.join(tk.uncached_path(self.code_root), "predict.py"),
          self.out_config_file.get_path(),
          self.out_cache_dir.get_path()
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)



class MathDialMBRJob(Job):
  """
  Train a Huggingface transformer model
  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=1,
      gpu_rqmt=1,
      python_exe=None,
      sbatch_args=None,
      responses_file=None,
      quality_file=None,
      acts_file=None,

      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.model_path = model_path
    self.review_file = responses_file
    self.review_score_file = acts_file
    self.quality_file = quality_file
    self.config = config
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "gpumem": gpumem,
      "sbatch_args": sbatch_args,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_metric_file = self.output_path("metrics.json")
    self.out_search_file = self.output_path("search_output.json")
    self.out_checkpoints_dir = self.output_path("checkpoints", directory=True)
    self.out_cache_dir = self.output_path("cache", directory=True)

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'metric_output_file': self.out_metric_file,
      'prediction_output_file': self.out_search_file,
      'output_dir': self.out_checkpoints_dir.get_path(),
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    self.config['method'] = "mathdial_mbr"
    self.config['reviews_file'] = self.review_file
    self.config['scores_file'] = self.review_score_file
    self.config['quality_file'] = self.quality_file
    # self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
        #   "--nproc_per_node=1",
          os.path.join(tk.uncached_path(self.code_root), "predict.py"),
          self.out_config_file.get_path(),
          self.out_cache_dir.get_path()
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)



class MathDialEvalJob(Job):
  """
  Train a Huggingface transformer model
  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=1,
      gpu_rqmt=1,
      python_exe=None,
      sbatch_args=None,
      responses_file=None,
      quality_file=None,
      mbr_file=None,
      acts_file=None,

      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.model_path = model_path
    self.review_file = responses_file
    self.review_score_file = acts_file
    self.quality_file = quality_file
    self.mbr_file = mbr_file
    self.config = config
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "gpumem": gpumem,
      "sbatch_args": sbatch_args,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_metric_file = self.output_path("metrics.json")
    self.out_search_file = self.output_path("search_output.json")
    self.out_checkpoints_dir = self.output_path("checkpoints", directory=True)
    self.out_cache_dir = self.output_path("cache", directory=True)

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'metric_output_file': self.out_metric_file,
      'prediction_output_file': self.out_search_file,
      'output_dir': self.out_checkpoints_dir.get_path(),
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    self.config['method'] = "eval_mathdial"
    self.config['reviews_file'] = self.review_file
    self.config['scores_file'] = self.review_score_file
    self.config['quality_file'] = self.quality_file
    self.config['mbr_file'] = self.mbr_file
    # self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
        #   "--nproc_per_node=1",
          os.path.join(tk.uncached_path(self.code_root), "predict.py"),
          self.out_config_file.get_path(),
          self.out_cache_dir.get_path()
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)





class DialogActCalibrationJob(Job):
  """
  Train a Huggingface transformer model
  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=1,
      gpu_rqmt=1,
      python_exe=None,
      sbatch_args=None,
      responses_file=None,
      acts_file=None,
      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.model_path = model_path
    self.review_file = responses_file
    self.review_score_file = acts_file
    self.config = config
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "gpumem": gpumem,
      "sbatch_args": sbatch_args,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_metric_file = self.output_path("metrics.json")
    self.out_search_file = self.output_path("search_output.json")
    self.out_checkpoints_dir = self.output_path("checkpoints", directory=True)
    self.out_cache_dir = self.output_path("cache", directory=True)

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'metric_output_file': self.out_metric_file,
      'prediction_output_file': self.out_search_file,
      'output_dir': self.out_checkpoints_dir.get_path(),
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    self.config['method'] = "mathdial_calibration"
    self.config['reviews_file'] = self.review_file
    self.config['scores_file'] = self.review_score_file
    # self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
        #   "--nproc_per_node=1",
          os.path.join(tk.uncached_path(self.code_root), "predict.py"),
          self.out_config_file.get_path(),
          self.out_cache_dir.get_path()
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)



class ConformalMathDialMBRJob(Job):
  """
  Train a Huggingface transformer model
  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=1,
      gpu_rqmt=1,
      python_exe=None,
      sbatch_args=None,
      responses_file=None,
      quality_file=None,
      acts_file=None,
      conformal_file=None,
      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.model_path = model_path
    self.review_file = responses_file
    self.review_score_file = acts_file
    self.quality_file = quality_file
    self.conformal_file = conformal_file
    self.config = config
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "gpumem": gpumem,
      "sbatch_args": sbatch_args,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_metric_file = self.output_path("metrics.json")
    self.out_search_file = self.output_path("search_output.json")
    self.out_checkpoints_dir = self.output_path("checkpoints", directory=True)
    self.out_cache_dir = self.output_path("cache", directory=True)

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'metric_output_file': self.out_metric_file,
      'prediction_output_file': self.out_search_file,
      'output_dir': self.out_checkpoints_dir.get_path(),
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    self.config['method'] = "mathdial_mbr"
    self.config['reviews_file'] = self.review_file
    self.config['scores_file'] = self.review_score_file
    self.config['quality_file'] = self.quality_file
    self.config['threshold_file'] = self.conformal_file
    # self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
        #   "--nproc_per_node=1",
          os.path.join(tk.uncached_path(self.code_root), "predict.py"),
          self.out_config_file.get_path(),
          self.out_cache_dir.get_path()
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)


class MathDialMinMaxJob(Job):
  """
  Train a Huggingface transformer model
  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=1,
      gpu_rqmt=1,
      python_exe=None,
      sbatch_args=None,
      responses_file=None,
      quality_file=None,
      acts_file=None,
      conformal_file=None,
      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.model_path = model_path
    self.review_file = responses_file
    self.review_score_file = acts_file
    self.quality_file = quality_file
    self.conformal_file = conformal_file
    self.config = config
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "gpumem": gpumem,
      "sbatch_args": sbatch_args,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_metric_file = self.output_path("metrics.json")
    self.out_search_file = self.output_path("search_output.json")
    self.out_checkpoints_dir = self.output_path("checkpoints", directory=True)
    self.out_cache_dir = self.output_path("cache", directory=True)

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'metric_output_file': self.out_metric_file,
      'prediction_output_file': self.out_search_file,
      'output_dir': self.out_checkpoints_dir.get_path(),
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    self.config['method'] = "mathdial_minmax"
    self.config['reviews_file'] = self.review_file
    self.config['scores_file'] = self.review_score_file
    self.config['quality_file'] = self.quality_file
    self.config['threshold_file'] = self.conformal_file
    # self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
        #   "--nproc_per_node=1",
          os.path.join(tk.uncached_path(self.code_root), "predict.py"),
          self.out_config_file.get_path(),
          self.out_cache_dir.get_path()
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)




class ReviewCalibrationJob(Job):
  """
  Train a Huggingface transformer model
  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=1,
      gpu_rqmt=1,
      python_exe=None,
      sbatch_args=None,
      responses_file=None,
      scores_file=None,
      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.model_path = model_path
    self.review_file = responses_file
    self.review_score_file = scores_file
    self.config = config
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "gpumem": gpumem,
      "sbatch_args": sbatch_args,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_metric_file = self.output_path("metrics.json")
    self.out_search_file = self.output_path("search_output.json")
    self.out_checkpoints_dir = self.output_path("checkpoints", directory=True)
    self.out_cache_dir = self.output_path("cache", directory=True)

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'metric_output_file': self.out_metric_file,
      'prediction_output_file': self.out_search_file,
      'output_dir': self.out_checkpoints_dir.get_path(),
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    self.config['method'] = "review_calibration"
    self.config['reviews_file'] = self.review_file
    self.config['scores_file'] = self.review_score_file
    # self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
        #   "--nproc_per_node=1",
          os.path.join(tk.uncached_path(self.code_root), "predict.py"),
          self.out_config_file.get_path(),
          self.out_cache_dir.get_path()
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)




class ConformalReviewMBRJob(Job):
  """
  Train a Huggingface transformer model
  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=1,
      gpu_rqmt=1,
      python_exe=None,
      sbatch_args=None,
      review_file=None,
      quality_file=None,
      calibration_file=None,
      review_score_file=None,

      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.model_path = model_path
    self.review_file = review_file
    self.review_score_file = review_score_file
    self.quality_file = quality_file
    self.config = config
    self.calibration_file = calibration_file
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "gpumem": gpumem,
      "sbatch_args": sbatch_args,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_metric_file = self.output_path("metrics.json")
    self.out_search_file = self.output_path("search_output.json")
    self.out_checkpoints_dir = self.output_path("checkpoints", directory=True)
    self.out_cache_dir = self.output_path("cache", directory=True)

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'metric_output_file': self.out_metric_file,
      'prediction_output_file': self.out_search_file,
      'output_dir': self.out_checkpoints_dir.get_path(),
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    self.config['method'] = "review_mbr"
    self.config['reviews_file'] = self.review_file
    self.config['scores_file'] = self.review_score_file
    self.config['quality_file'] = self.quality_file
    self.config['threshold_file'] = self.calibration_file
    # self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
        #   "--nproc_per_node=1",
          os.path.join(tk.uncached_path(self.code_root), "predict.py"),
          self.out_config_file.get_path(),
          self.out_cache_dir.get_path()
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)




class MinMaxReviewMBRJob(Job):
  """
  Train a Huggingface transformer model
  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=1,
      gpu_rqmt=1,
      python_exe=None,
      sbatch_args=None,
      review_file=None,
      quality_file=None,
      calibration_file=None,
      review_score_file=None,

      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.model_path = model_path
    self.review_file = review_file
    self.review_score_file = review_score_file
    self.quality_file = quality_file
    self.config = config
    self.calibration_file = calibration_file
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "gpumem": gpumem,
      "sbatch_args": sbatch_args,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_metric_file = self.output_path("metrics.json")
    self.out_search_file = self.output_path("search_output.json")
    self.out_checkpoints_dir = self.output_path("checkpoints", directory=True)
    self.out_cache_dir = self.output_path("cache", directory=True)

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'metric_output_file': self.out_metric_file,
      'prediction_output_file': self.out_search_file,
      'output_dir': self.out_checkpoints_dir.get_path(),
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    self.config['method'] = "review_minmax"
    self.config['reviews_file'] = self.review_file
    self.config['scores_file'] = self.review_score_file
    self.config['quality_file'] = self.quality_file
    self.config['threshold_file'] = self.calibration_file
    # self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
        #   "--nproc_per_node=1",
          os.path.join(tk.uncached_path(self.code_root), "predict.py"),
          self.out_config_file.get_path(),
          self.out_cache_dir.get_path()
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)
