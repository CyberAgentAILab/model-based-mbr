# Model-Based Minimum Bayes Risk Decoding


This repository contains the code for the experiments in [Model-Based Minimum Bayes Risk Decoding](https://arxiv.org/abs/2311.05263).

The code is tested on Ubuntu 20.04 using Python 3.8 and CUDA 11.0 (Docker image nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04).

## Installation

```
git clone git@github.com/CyberAgentAILab/mbmbr
cd mbmbr
pip install -r requirements.txt
```

## Usage

The code runs in two steps.
1. `sample.sh` samples candidates.
2. `run_mbr.sh` computes the MBR and MBMBR outputs from the sampled candidates.

### Sampling candidates

```
./experiments/sample.sh -d [DATASET] -s [NUMBER OF SAMPLES] 
```

### Computing MBR

```
./experiments/run_mbr.sh -d [DATASET] -s [NUMBER OF SAMPLES]
```


### Example: WMT'21 En-De

1. Use [sacrebleu](https://github.com/mjpost/sacrebleu) to prepare the benchmark dataset.
```
mkdir -p ./dataset/wmt19-text
sacrebleu -t wmt19 -l en-de --echo src > ./dataset/wmt19-text/wmt19.en-de.en
sacrebleu -t wmt19 -l en-de --echo ref > ./dataset/wmt19-text/wmt19.en-de.de
```

2. Sampling sequences on WMT'19 En-De

```
./experiments/sample.sh -d wmt19.en-de
```

3. Computing the MBR output on WMT'19 En-De

```
./experiments/run_mbr.sh -d wmt19.en-de
```

## Reference

[Jinnai, Y., Morimura, T., Honda, U., Ariu, K., & Abe, K. (2023). Model-based minimum bayes risk decoding. arXiv preprint arXiv:2311.05263.](https://arxiv.org/abs/2311.05263)

Bibtex:
```
@article{jinnai2023modelbased,
  title={Model-Based Minimum Bayes Risk Decoding}, 
  author={Yuu Jinnai and Tetsuro Morimura and Ukyo Honda and Kaito Ariu and Kenshi Abe},
  year={2023},
  journal={arXiv preprint arXiv:2311.05263}
}
```

## Contact
For any questions, feel free to raise an issue or contact me at jinnai_yu@cyberagent.co.jp.
