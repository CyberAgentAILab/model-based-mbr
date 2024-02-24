import os

import json
from collections.abc import Iterable
from glob import glob

import numpy as np
import pandas as pd

import torch
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers import StoppingCriteria, StoppingCriteriaList
import datasets

# TODO: this wanna be set by base directory
dataset_dir = './dataset'
sample_dir = './samples'
score_dir = './score' # not used
output_dir = './output' # not used
evaluate_dir = './evaluate'
prompt_dir = './prompts'
result_dir = './results'
matrix_dir = './matrix'

def load_model(dataset, torch_device, model_name):
    # TODO: It is important to specify "None" when generating texts using sequence-to-sequence models.
    # Otherwise, it will generate texts using language models.
    stop_tokens = []
    if model_name == "None":
        if 'wmt19' in dataset:
            mname = "facebook/wmt19-" + dataset.split('.')[1]
            tokenizer = FSMTTokenizer.from_pretrained(mname)
            model = FSMTForConditionalGeneration.from_pretrained(mname)
            model.to(torch_device)
        elif dataset in ['xsum', 'cnndm', 'samsum']:
            if dataset == 'samsum':
                mname = "philschmid/bart-large-cnn-samsum"
            else:
                mname = "facebook/bart-large-" + dataset
            model = BartForConditionalGeneration.from_pretrained(mname)
            tokenizer = BartTokenizer.from_pretrained(mname)
            model.to(torch_device)
        elif dataset in ['nocaps', 'mscoco', 'mscoco-ft']:
            if dataset == 'mscoco-ft':
                mname = "Salesforce/blip2-flan-t5-xl-coco"
            else:
                mname = "Salesforce/blip2-flan-t5-xl"
            tokenizer = AutoProcessor.from_pretrained(mname)
            model = Blip2ForConditionalGeneration.from_pretrained(mname, load_in_8bit=True, device_map="auto")

        else:
            assert False
    else:
        mname = model_name
        model_n = os.path.basename(model_name)
        if ('polylm' in model_n):
            tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False, use_fast=False, padding_size="left")
            if 'polylm' in model_n:
                stop_tokens = [213] # new line token \n
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_size="left", use_fast=True, trust_remote_code=True)
            if 'falcon' in model_n:
                # TODO: it doesn't fully solve the problem. There seems to be random newline tokens.
                stop_tokens = [193, 1001]
            elif 'bloomz' in model_n:
                stop_tokens = [2]
            else:
                print('Stop token is not set for', model_n)
                stop_tokens = []
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # TODO: List models which needs to be quantized to 4 bits.
        # Actually there is only a marginal drop in performance by 4 bit quants compared to 8bit.
        if ('Mistral' in model_name) or ('zephyr' in model_name) or ('mistral-7b-sft-beta' in model_name):
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, load_in_4bit=True, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, load_in_8bit=True, device_map="auto")


    return tokenizer, model, mname, stop_tokens

def load_dataset(dataset, ref=False):
    # TODO: Refactor: not so clean.
    if 'wmt19' in dataset:
        subdir_name = 'wmt19-text'
    elif 'iwslt17' in dataset:
        subdir_name = 'iwslt17-text'
    elif dataset == 'xsum':
        subdir_name = 'xsum'
    elif dataset == 'cnndm':
        subdir_name = 'cnn_cln' # TODO: wanna change the dirname but for backward compatibility.
    else:
        subdir_name = 'hf' # Using huggingface dataset

    if not (subdir_name == "hf"):
        if subdir_name in ['wmt19-text', 'iwslt17-text', 'wmt21']:
            # dataset is wmt19.{src}-{trg}

            src_lang = dataset.split(".")[1].split("-")[0]
            trg_lang = dataset.split(".")[1].split("-")[1]

            if not ref:
                filename = dataset + '.' + src_lang
            else:
                filename = dataset + '.' + trg_lang
        else:
            if not ref:
                filename = 'test.source'
            else:
                filename = 'test.target'

        path = os.path.join(dataset_dir, subdir_name, filename)
        
        with open(path) as f:
            lines = f.read().splitlines()
    else:
        if dataset == 'samsum':
            dataset = datasets.load_dataset("samsum", split='test')
            if not ref:
                lines = dataset['dialogue']
            else:
                lines = dataset['summary']
        elif dataset == 'nocaps':
            dataset = datasets.load_dataset("HuggingFaceM4/NoCaps", split='validation')
            if not ref:
                lines = dataset['image']
            else:
                lines = dataset['annotations_captions']
        elif dataset == 'mscoco-ft':
            # TODO: how do we refactor this? -> maybe inevitable complication
            mscoco_df = pd.read_csv('experiments/mscoco.csv')
            if not ref:
                dataset = datasets.load_dataset("HuggingFaceM4/COCO", split='test')
                images = []
                for ind in range(len(mscoco_df)):
                    image_pos = int(mscoco_df.iloc[ind]['img_pos'])
                    image = dataset[image_pos]['image']
                    images.append(image)
                lines = images
            else:
                lines = [eval(l) for l in mscoco_df['texts']]
        elif dataset == "e2e_nlg":
            dataset = datasets.load_dataset("e2e_nlg", split='validation')
            inputs_sorted = list(set(dataset['meaning_representation']))
            if not ref:
                inputs = {}
                for row in dataset:
                    src = row['meaning_representation']
                    if src in inputs.keys():
                        pass
                    else:
                        inputs[src] = row['meaning_representation']

                lines = list(inputs.values())
            else:
                targets = {}
                for row in dataset:
                    src = row['meaning_representation']
                    if src in targets.keys():
                        targets[src].append(row['human_reference'])
                    else:
                        targets[src] = [row['human_reference']]
                lines = list(targets.values())
        else:
            assert False

    assert isinstance(lines, Iterable)
    return lines

def load_kwargs(dataset):
    # Length penalty is not used in sampling. But for the purpose of keeping it same as in the original paper, it is set here.
    WMT19_KWARGS = dict(length_penalty=1.0, max_new_tokens=40, no_repeat_ngram_size=3)
    XSUM_KWARGS = dict(length_penalty=1.0, max_new_tokens=60, min_new_tokens=10, no_repeat_ngram_size=3)
    CNN_KWARGS = dict(length_penalty=2.0, max_new_tokens=140, min_new_tokens=55, no_repeat_ngram_size=3)
    SAMSUM_KWARGS = dict(length_penalty=2.0, max_new_tokens=142, min_new_tokens=56, no_repeat_ngram_size=3)
    CAPTION_KWARGS = dict(length_penalty=0.6, max_new_tokens=30)
    E2E_KWARGS = dict(max_new_tokens=40, min_new_tokens=5, no_repeat_ngram_size=3) # This is set by me but seems not optimal at all.
    LLM_KWARGS = dict(max_new_tokens=300, no_repeat_ngram_size=4)

    if ('wmt' in dataset) or ('iwslt17' in dataset):
        model_kwargs = WMT19_KWARGS
    elif (dataset == 'xsum'):
        model_kwargs = XSUM_KWARGS
    elif dataset == "samsum":
        model_kwargs = SAMSUM_KWARGS
    elif dataset == 'cnndm':
        model_kwargs = CNN_KWARGS
    elif (dataset == 'nocaps') or ('mscoco' in dataset):
        model_kwargs = CAPTION_KWARGS
    elif dataset in ['e2e_nlg', 'squad_v2', 'common_gen']:
        model_kwargs = E2E_KWARGS
    elif dataset in ['alpaca']:
        model_kwargs = LLM_KWARGS
    else:
        print('No parameters specified: default to LLM_KWARGS')
        model_kwargs = LLM_KWARGS
    return model_kwargs

def load_matrix(target_matrix_dir, filename, sim, n_samples):
    """
    Load matrix from the target_matrix_dir.
    The matrix is saved as a text file.
    The matrix stored may be larger than n_samples. In this case, it is truncated.
    If no files found, it returns None.
    """
    
    matrix_base = os.path.join(target_matrix_dir, filename + "_" + sim + "_")
    matrix_paths = glob(matrix_base + "*")
    
    cached_nsamples = [int(f[len(matrix_base):]) for f in matrix_paths]   
    larger_cahces = [c for c in cached_nsamples if c >= n_samples]
           
    if len(larger_cahces) == 0:
        return None
    
    # Load the smallest matrix larger than n_samples.
    # TODO: We should be able to clear matrix cache.
    min_nsamples = min(larger_cahces)

    matrix = np.loadtxt(matrix_base + str(min_nsamples))
    matrix = matrix[:n_samples, :n_samples]
    
    return matrix

    
def load_samples(dataset, sample_id, eps):
    # This function is old. not accepting input from recent sample files
    print("utils.load_samples: not maintained")
    sample_path = os.path.join(sample_dir, dataset, "{:04d}_eps-{:.2f}".format(sample_id, eps))

    with open(sample_path) as f:
        samples = f.read().splitlines()

    return samples


def load_samples_from_file(files, epsilon, topk, topp, do_sample=True, diverse_k=1, divpen=0.0):
    # TODO: Clean it up (backward compatibility)
    # To keep backward compatibility to the old format, it needs two steps.
    # First it loads in current format and it no files found, it loads in old format.
    filtered_files = []
    
    if do_sample:
        for filename in files:
            isnt_eps = not "eps-{:.2f}".format(epsilon) in filename
            
            # If topk is set to negative (e.g. -1), then it means that "topk" should not be in the filename.
            if topk < 0:
                isnt_topk = "topk" in filename
            else:
                isnt_topk = not "topk-{:02d}".format(topk) in filename
            
            if topp < 0:
                isnt_topp = "topp" in filename
            else:
                isnt_topp = not "topp-{:.2f}".format(topp) in filename

            if not (isnt_eps or isnt_topk or isnt_topp):
                filtered_files.append(filename)
    else:
        for filename in files:
            k_matches = "beam-{:02d}".format(diverse_k) in filename
            dp_matches = "divpen-{:.2f}".format(divpen) in filename

            if k_matches and dp_matches:
                filtered_files.append(filename)
    
    return filtered_files

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
      super().__init__()
      self.stops = stops
      self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
      stop_count = 0
      for stop in self.stops:
        stop_count += (stop == input_ids[0]).sum().item()

      if stop_count >= self.ENCOUNTERS:
          return True
      return False

def list_to_text(words):
    text = words[0]
    for w in words[1:]:
        text = text + " " + w
    return text

