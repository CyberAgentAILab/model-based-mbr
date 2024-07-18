# Model-Based Minimum Bayes Risk Decoding


This repository contains the code for the experiments in [Model-Based Minimum Bayes Risk Decoding](https://openreview.net/pdf?id=qDUaH9xHVV).

The code is tested on Ubuntu 20.04 using Python 3.8 and CUDA 11.0 (Docker image nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04).

## Installation

```
git clone git@github.com/CyberAgentAILab/model-based-mbr
cd model-based-mbr
pip install -r requirements.txt
```

## Usage

The code runs in two steps.
1. `sample.sh` samples candidates.
2. `run_mbr.sh` computes the MBR and MBMBR outputs from the sampled candidates.

### 1. Sampling candidates

```
./experiments/sample.sh -d [DATASET] -s [NUMBER OF SAMPLES] 
```

### 2. Computing the MBR and MBMBR outputs

```
./experiments/run_mbr.sh -d [DATASET] -s [NUMBER OF SAMPLES]
```


## Example: WMT'19 En-De

1. Use [sacrebleu](https://github.com/mjpost/sacrebleu) to prepare the benchmark dataset.
```
mkdir -p ./dataset/wmt19-text
sacrebleu -t wmt19 -l en-de --echo src > ./dataset/wmt19-text/wmt19.en-de.en
sacrebleu -t wmt19 -l en-de --echo ref > ./dataset/wmt19-text/wmt19.en-de.de
```

2. Sampling sequences on WMT'19 En-De

```
./experiments/sample.sh -d wmt19.en-de -s 32
```

3. Computing the MBR output on WMT'19 En-De

```
./experiments/run_mbr.sh -d wmt19.en-de -s 32
```

## mbrs Library

Our repository is published to ensure the reproducibility of the research.  
For running MBMBR, it is also available in the [mbrs](https://github.com/naist-nlp/mbrs) library.
The library is maintained for running various versions of MBR decoding algorithms. It is compatible with both Huggingface's transformers and fairseq.  
I recommend using the mbrs library for quick experiments.  
It is also available via pypi: `pip install mbrs`.

## Reference

[Jinnai, Y., Morimura, T., Honda, U., Ariu, K., & Abe, K. (2024). Model-based minimum Bayes risk decoding for text generation. Forty-first International Conference on Machine Learning.](https://openreview.net/forum?id=qDUaH9xHVV)

Bibtex:
```
@inproceedings{
  jinnai2024modelbased,
  title={Model-Based Minimum Bayes Risk Decoding for Text Generation},
  author={Yuu Jinnai and Tetsuro Morimura and Ukyo Honda and Kaito Ariu and Kenshi Abe},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=qDUaH9xHVV}
}
```

## Contact
For any questions, feel free to raise an issue or contact me at jinnai_yu@cyberagent.co.jp.

## Acknowledgements

[MS COCO dataset](https://cocodataset.org/#home) is licensed under a [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/).
