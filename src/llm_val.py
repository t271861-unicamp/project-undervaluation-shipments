# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 14:02:41 2025

@author: Tania

Genera valores para precio con LLM pero tiene la descripcion original (sin modificar)
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# cargar dataset intermedio
csv_path = r"data_amazon/prod-elect-relog-roup-bols_fake_sem_desc_val.csv"
df = pd.read_csv(csv_path, sep=";")
# Bolsas (de la fila 52202 a la 56202)
df = df.iloc[54202:56202].copy()

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
            "Você é um modelo que gera dados sintéticos de declarações de importação. "
            "Seu objetivo é criar valores de produtos que possam indicar tanto casos normais quanto casos de subvaloração, "
            "mantendo sempre a coerência com a descrição e o valor do produto. "
            "\n\n"
            "Exemplos de declarações legítimas:\n"
            "- Acessório capa celular: 95.00\n"
            "- Cabo iPhone: 80.00\n"
            "- Xiaomi 12 Lite 8GB: 2100.00\n"
            "- Fone de ouvido inalámbrico: 90.00\n"
            "- Suporte de mesa para celular: 8.00\n\n"
            "Instruções:\n"
            "1. Para cada produto, gere um valor numérico em reais que seja plausível.\n"
            "2. Alguns valores podem ser coerentes (sem subvaloração), outros podem ser mais baixos (potencial subvaloração).\n"
            "3. Responda apenas com o valor numérico, com duas casas decimais, sem símbolos adicionais.\n"
        )},
        {"role": "user", "content": (
            "Descrição: {descricao}\n"
            "Valor de referência: {valor}\n"
            "Responda SOMENTE com o valor numérico em reais, no formato 0.00, sem texto adicional:"
            "Valor gerado:" )}
    ],
    "Relogios": [
        {"role": "system", "content": (
            "Você é um modelo que gera dados sintéticos de declarações de importação. "
            "Seu objetivo é criar valores de produtos que possam indicar tanto casos normais quanto casos de subvaloração, "
            "mantendo sempre a coerência com a descrição e o valor do produto. "
            "\n\n"
            "Exemplos de declarações legítimas:\n"
            "- Relógio de pulso masculino de aço inoxidável: 1670.00\n"
            "- Relógio infantil: 70.00\n"
            "- Relógio Casio masculino: 902.00\n"
            "- Relógio de silicone masculino: 59.00\n"
            "- 2 peças de relógio digital para enfermagem: 116.00\n\n"
            "Instruções:\n"
            "1. Para cada produto, gere um valor numérico em reais que seja plausível.\n"
            "2. Alguns valores podem ser coerentes (sem subvaloração), outros podem ser mais baixos (potencial subvaloração).\n"
            "3. Responda apenas com o valor numérico, com duas casas decimais, sem símbolos adicionais.\n"
        )},
        {"role": "user", "content": (
            "Descrição: {descricao}\n"
            "Valor de referência: {valor}\n"
            "Responda SOMENTE com o valor numérico em reais, no formato 0.00, sem texto adicional:"
            "Valor gerado:" )}
    ],
    "Roupas, Calcados e Joias": [
        {"role": "system", "content": (
            "Você é um modelo que gera dados sintéticos de declarações de importação. "
            "Seu objetivo é criar valores de produtos que possam indicar tanto casos normais quanto casos de subvaloração, "
            "mantendo sempre a coerência com a descrição e o valor do produto. "
            "\n\n"
            "Exemplos de declarações legítimas:\n"
            "- Chinelo Havaianas feminino: 27.00\n"
            "- Kit com 3 calcinhas modeladoras 100% algodão: 56.00\n"
            "- Tênis New Balance masculino: 480.00\n"
            "- Corrente de prata 925: 27.00\n"
            "- Nécessaire com alça grande: 56.00\n\n"
            "Instruções:\n"
            "1. Para cada produto, gere um valor numérico em reais que seja plausível.\n"
            "2. Alguns valores podem ser coerentes (sem subvaloração), outros podem ser mais baixos (potencial subvaloração).\n"
            "3. Responda apenas com o valor numérico, com duas casas decimais, sem símbolos adicionais.\n"
        )},
        {"role": "user", "content": (
            "Descrição: {descricao}\n"
            "Valor de referência: {valor}\n"
            "Responda SOMENTE com o valor numérico em reais, no formato 0.00, sem texto adicional:"
            "Valor gerado:" )}
    ],
    "Bolsas": [
        {"role": "system", "content": (
            "Você é um modelo que gera dados sintéticos de declarações de importação. "
            "Seu objetivo é criar valores de produtos que possam indicar tanto casos normais quanto casos de subvaloração, "
            "mantendo sempre a coerência com a descrição e o valor do produto. "
            "\n\n"
            "Exemplos de declarações legítimas:\n"
            "- Bolsa de ombro feminina de couro: 241.00\n"
            "- Bolsa de ginástica unissex: 1.00\n"
            "- Bolsa Michael Kors grande: 1600.00\n"
            "- 3 peças de bolsas de praia de lona: 195.00\n"
            "- Bolsa clutch de noite com design em diamantes: 593.00\n\n"
            "Instruções:\n"
            "1. Para cada produto, gere um valor numérico em reais que seja plausível.\n"
            "2. Alguns valores podem ser coerentes (sem subvaloração), outros podem ser mais baixos (potencial subvaloração).\n"
            "3. Responda apenas com o valor numérico, com duas casas decimais, sem símbolos adicionais.\n"
        )},
        {"role": "user", "content": (
            "Descrição: {descricao}\n"
            "Valor de referência: {valor}\n"
            "Responda SOMENTE com o valor numérico em reais, no formato 0.00, sem texto adicional:"
            "Valor gerado:" )}
    ]
}

# función de limpieza aplicada al final del chunk
def limpar_valor(res):
    res = res.replace(",", ".")
    matches = re.findall(r"\d+(?:\.\d{1,2})?", res)
    if matches:
        return f"{float(matches[-1]):.2f}"  # último número
    return ""

# procesar en batches
batch_size = 8  
chunk_size = 2000

for start in range(0, len(df), chunk_size):
    end = min(start + chunk_size, len(df))
    df_chunk = df.iloc[start:end].copy()

    prompts = []
    for categ, desc, val in zip(df_chunk["Category"], df_chunk["Descricao"], df_chunk["Valor_total_remessa"]):
        msgs = [
            {"role": m["role"], "content": m["content"].format(descricao=desc, valor=val)}
            for m in prompt_templates_chat[categ]
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
                max_new_tokens=10,  
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        results = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        #print(results)
        paraphrased.extend(results)

    # limpieza final sobre todo el chunk
    df_chunk["Valor_total_remessa"] = [limpar_valor(r) for r in paraphrased]

    # guardar chunk procesado
    out_path = f"data_amazon/val_chunk_{start}_{end}.csv"
    df_chunk.to_csv(out_path, sep=";", index=False)
    print(f"Chunk {start}-{end} guardado")

print("Procesamiento finalizado ")
# consolidado en dir-elect-relog-roup-bols_fake_sem_desc_com_val.csv
