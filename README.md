# Model-Based Minimum Bayes Risk Decoding


This repository contains the code for the experiments in [Model-Based Minimum Bayes Risk Decoding](https://proceedings.mlr.press/v235/jinnai24a.html).

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

MBMBR is also implemented in the [mbrs](https://github.com/naist-nlp/mbrs) library and is available via pypi: 

```pip install mbrs```

The mbrs library is maintained for running various versions of MBR decoding algorithms. It is compatible with both Huggingface's transformers and fairseq. 

## Reference

[Jinnai, Y., Morimura, T., Honda, U., Ariu, K. &amp; Abe, K.. (2024). Model-Based Minimum Bayes Risk Decoding for Text Generation. <i>Proceedings of the 41st International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i>.](https://proceedings.mlr.press/v235/jinnai24a.html)

Bibtex:
```

@InProceedings{pmlr-v235-jinnai24a,
  title = 	 {Model-Based Minimum {B}ayes Risk Decoding for Text Generation},
  author =       {Jinnai, Yuu and Morimura, Tetsuro and Honda, Ukyo and Ariu, Kaito and Abe, Kenshi},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {22326--22347},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/jinnai24a/jinnai24a.pdf},
  url = 	 {https://proceedings.mlr.press/v235/jinnai24a.html},
}

```

## Contact
For any questions, feel free to raise an issue or contact me at jinnai_yu@cyberagent.co.jp.

## Acknowledgements

[MS COCO dataset](https://cocodataset.org/#home) is licensed under a [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/).
