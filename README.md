# Model-Based Minimum Bayes Risk Decoding


This repository contains the code for the experiments in [Model-Based Minimum Bayes Risk Decoding](https://arxiv.org/abs/2311.05263).

The code is provided mostly as is with little effort on refactoring.

## Installation

```
git clone git@github.com/jinnaiyuu/mbmbr
cd mbmbr
pip install requirements.txt
```

## Usage

The code runs in two steps.
1. `sample.sh` samples candidates.
2. `run_mbr.sh ` computes the MBR candidate from the candidates sampled.

### Sampling candidates

```
./experiments/sample.sh -d [DATASET] -s [NUMBER OF SAMPLES] 
```

### Computing MBR

```
./experiments/run_mbr.sh -d [DATASET] -s [NUMBER OF SAMPLES]
```


### Example: WMT'21 En-De

Sampling sequences on WMT'19 En-De

```
./experiments/sample.sh -d wmt19.en-de
```

Computing the MBR output on WMT'19 En-De

```
./experiments/run_mbr.sh -d wmt19.en-de
```
