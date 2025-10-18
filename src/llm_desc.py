# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 13:14:15 2025

@author: Tania

Genera texto LLM para las descripciones pero tiene un valor random para precio
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# cargar dataset intermedio
csv_path = r"data_amazon/dir-elect-relog-roup-bols_fake_sem_desc_com_val.csv"
df = pd.read_csv(csv_path, sep=";")


# cargar modelo LLaMA
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
# para definir pad_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
    
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# definir prompts por categoría (ya en formato messages)
prompt_templates_chat = {
    "Eletronicos": [
        {"role": "system", "content": (
            "Você é um assistente especializado em resumir descrições de produtos."
            "Sua tarefa é reduzir as descrições de produtos eletrônicos. "
            "Ignore informações redundantes, de marca, de gênero, de uso ou promocionais. "
            "Retorne apenas uma versão resumida da descrição, com no máximo 4 palavras."
            "Responda exclusivamente em português."
        )},
        {"role": "user", "content": "Descrição: {desc}"}
    ],
    "Relogios": [
        {"role": "system", "content": (
            "Você é um assistente especializado em resumir descrições de produtos."
            "Sua tarefa é reduzir as descrições de relógios."
            "Ignore gênero, público-alvo, estilo ou adjetivos promocionais."
            "Retorne apenas uma versão resumida da descrição, com no máximo 4 palavras."
            "Responda exclusivamente em português."
        )},
        {"role": "user", "content": "Descrição: {desc}"}
    ],
    "Roupas, Calcados e Joias": [
        {"role": "system", "content": (
            "Você é um assistente especializado em resumir descrições de produtos."
            "Sua tarefa é reduzir as descrições de roupas, calçados e joias."
            "Ignore gênero, público-alvo, tamanho, cor, estilo ou ocasião."
            "Retorne apenas uma versão resumida da descrição, com no máximo 4 palavras."
            "Responda exclusivamente em português."
        )},
        {"role": "user", "content": "Descrição: {desc}"}
    ],
    "Bolsas": [
        {"role": "system", "content": (
            "Você é um assistente especializado em resumir descrições de produtos."
            "Sua tarefa é reduzir as descrições de bolsas"
            "Ignore gênero, público-alvo, tamanho, cor ou estilo. "
            "Retorne apenas uma versão resumida da descrição, com no máximo 4 palavras."
            "Responda exclusivamente em português."
        )},
        {"role": "user", "content": "Descrição: {desc}"}
    ]
}

# función de limpieza aplicada al final del chunk
def limpar_texto(res):
    if "[/INST]" in res:
        res = res.split("[/INST]", 1)[-1].strip()
    # eliminar duplicados consecutivos
    words = res.split()
    res = " ".join([w for i, w in enumerate(words) if i == 0 or w != words[i-1]])
    # quitar oraciones incompletas
    sentences = re.findall(r'[^.!?]*[.!?]', res)
    if sentences:
        res = " ".join(s.strip() for s in sentences)
    return res.strip()

# procesar en batches
batch_size = 8  
chunk_size = 10000

for start in range(0, len(df), chunk_size):
    end = min(start + chunk_size, len(df))
    df_chunk = df.iloc[start:end].copy()

    prompts = []
    for categoria, descricao in zip(df_chunk["Category"], df_chunk["Descricao"]):
        msgs = [
            {"role": m["role"], "content": m["content"].format(desc=descricao)}
            for m in prompt_templates_chat[categoria]
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    paraphrased = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,  # suficiente para 4 palavras
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        paraphrased.extend(results)

    # limpieza final sobre todo el chunk
    df_chunk["Descricao"] = [limpar_texto(r) for r in paraphrased]

    # guardar chunk procesado
    out_path = f"data_amazon/desc_chunk_{start}_{end}.csv"
    df_chunk.to_csv(out_path, sep=";", index=False)
    print(f"Chunk {start}-{end} guardado")

print("Procesamiento finalizado ")
# consolidado en dir-elect-relog-roup-bols_fake_com_desc_com_val.csv
