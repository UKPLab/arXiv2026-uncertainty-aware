# Uncertainty-Aware Generation and Decision-Making Under Ambiguity

[![Arxiv](https://img.shields.io/badge/Arxiv-2606.30578-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.30578)
[![License](https://img.shields.io/github/license/UKPLab/arXiv2026-uncertainty-aware)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

This repository contains sourcecode to reproduce and run experiments from the above paper that is currently a preprint.

> **Abstract:** With rapidly improving capabilities, Large Language Models (LLMs) are increasingly used in many complex real-world tasks. Beyond requiring in-depth knowledge and reasoning skills, many of these tasks exhibit a high degree of subjectivity and require that the outputs of the model can be trusted. While a lot of progress has been made to train better models, decision-making algorithms have received less attention. In this work, we present and evaluate various uncertainty-aware decision-making algorithms based on Bayesian decision theory and risk-averse decision making on the tasks of tutoring and automatic peer reviewing. Concretely, we take uncertainty over tutoring strategies and review scores into account when generating a tutor response or review and use conformal prediction to provide guarantees over strategy and score. We find empirically that these algorithms can improve the utility of the generations but need to be carefully implemented when ambiguity is high. For example, risk-averse rules can degrade performance by optimizing for generic outputs, while Bayesian methods tend to perform better. Our work uses techniques from decision theory to improve LLM-based decision-making and outlines open challenges for the community.

> **Disclaimer:** We do not promote automation of the peer-review process but aim to support human reviewers and authors.

Contact person: [Nico Daheim](mailto:nico.daheim@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.


## Getting Started
Running
```
pip install -r requirements.txt
```
will install all necessary packages.

All code is located in `arxiv2026_uncertainty_aware` and we `cd` into it next :).

The experiments of the paper were organized using the workflow manager [Sisyphus](https://github.com/rwth-i6/sisyphus). If you would like to make use of it, too, then please run:
```
git clone git@github.com:rwth-i6/sisyphus.git
cd sisyphus/
pip install -r requirements.txt
cd ..
mkdir alias
mkdir output
mkdir work
```
Sisyphus will use the directories as follows:
  1. `alias`: It's possible to identify aliases for each job to identify it quickly (as a default, a hash is appended to the jobclass name as an identifier), and sisyphus adds a symlink to the job under the alias.
  2. `output`: `tk.register_output("name", job_class.file)` registers an output under the filename `name` in the output folder that symlinks to `job_class.file`
  3. `work`: All jobs will be placed here under their hash.

## Usage

### Running experiments using Sisyphus

Examples for training with Sisyphus are found in the `config/` folder.
Running experiments for Mathdial (https://aclanthology.org/2023.findings-emnlp.372/) can be done via running the following:
```
cd arxiv2026_uncertainty_aware
sisyphus/sis --config config config/mathdial.py
```

In the same folder there is also code to run experiments on NLPEER (https://aclanthology.org/2023.acl-long.277/).

The config file also defines all the prompts that we use for the various generation, LLM-judge, and utility models.

The main code is found under `code`, where under `code/arxiv/methods/` there are various implementations of methods for sampling outputs, scoring them, evaluation, and conformal prediction.

Prediction is done using `code/arxiv/trainer/custom_trainer.py`, where vllm is called directly.


### Code Structure

The code is mainly based on the concept of ''methods'' that are found in the `/code/arxiv/methods/` folder which wrap all of the functionality needed to reproduce a certain method:
  1. Defining and loading Trainer and Data Collator classes
  2. Loading all datasets
  3. Defining and applying the preprocessing methods, defined in `/code/arxiv/methods/preprocessing`

To understand how the method classes are structured it's best to check `code/arxiv/methods/base.py` which defines a base class from which all methods inherit.

The main entry point for the code is `/code/arxiv/main_simple.py` that handles loading method classes, models, and running the Trainers.

## Cite

Please use the following citation:

```
@misc{daheim2026uncertaintyawaregenerationdecisionmakingambiguity,
      title={Uncertainty-Aware Generation and Decision-Making Under Ambiguity}, 
      author={Nico Daheim and Iryna Gurevych},
      year={2026},
      eprint={2606.30578},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.30578}, 
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

> The repo template is adapted from [python-project-template](https://github.com/rochacbruno/python-project-template/) by [rochacbruno](https://github.com/rochacbruno).
