# from transformers.models.auto.tokenization_auto import AutoTokenizer
import torch
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM
)
import yaml
from typing import Tuple

@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def load_model(model_checkpoint):
    # config = AutoConfig.from_pretrained(model_checkpoint)
    # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    return model
