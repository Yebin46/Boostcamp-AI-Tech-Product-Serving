import streamlit as st

import io
import os
import yaml

import torch
import numpy as np

from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from predict import load_model
from confirm_button_hack import cache_on_button_press

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

root_password = 'metamong'

def main():
    st.title("Summarization Model")

    model_checkpoint = 'gogamza/kobart-summarization'
    config = AutoConfig.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    model = load_model(model_checkpoint)
    model.eval()
    sentences = st.text_input('요약하고 싶은 문서를 입력하세요.')

    if sentences:
        raw_input_ids = tokenizer(sentences, max_length=1024, truncation=True)
        input_ids = [tokenizer.bos_token_id] + raw_input_ids['input_ids'][:-2] + [tokenizer.eos_token_id]
        num_beams = 3
        # generation_args.num_return_sequences = num_beams
        output_ids = model.generate(torch.tensor([input_ids]), num_beams=num_beams, num_return_sequences=3)
        if len(output_ids.shape) == 1  or output_ids.shape[0] == 1:
            title = tokenizer.decode(output_ids.squeeze().tolist(), skip_special_tokens=True)
            st.write(title)
        else:
            titles = tokenizer.batch_decode(output_ids.squeeze().tolist(), skip_special_tokens=True)
            for idx, title in enumerate(titles):
                st.write(idx, title)


@cache_on_button_press('Authenticate')
def authenticate(password) ->bool:
    print(type(password))
    return password == root_password

password = st.text_input('비밀번호를 입력하세요.', type="password")

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')