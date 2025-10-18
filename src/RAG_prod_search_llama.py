# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 14:11:51 2025

@author: Admin

    """

from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb
from chromadb.utils import embedding_functions

from sentence_transformers import SentenceTransformer, util  # Paso 2
import torch

import re

# Configuración
CHROMA_PATH = "prod_fakedb_embeddings"
COLLECTION_NAME = "prod-elect-relog-roup-bols_fake"
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"

# Modelo de lenguaje (LLaMA 2 7B Chat usado como causal LM)
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# Modelo de embeddings
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Base de datos y función de embeddings
client = chromadb.PersistentClient(CHROMA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

# Descripción consultada
descricao_consultada = "Relogio digital esportivo com 2 pecas"

# Recuperar descripciones similares
results = collection.query(
    query_texts=[descricao_consultada],
    n_results=5,
    include=["documents", "metadatas"]
)

similar_products = results["documents"][0]
metadatas = results["metadatas"][0]

print("Produtos mais semelhantes ao item pesquisado, identificados na base de dados:\n")
query_embedding = embedder.encode(descricao_consultada, convert_to_tensor=True)

for i, (descricao, meta) in enumerate(zip(similar_products, metadatas), 1):
    product_embedding = embedder.encode(descricao, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(query_embedding, product_embedding).item()
    similarity_percentage = round(similarity_score * 100, 2)

    min_price = meta.get("MinPrice", "N/A")
    max_price = meta.get("MaxPrice", "N/A")

    print(f"Produto {i}: {descricao}")
    print(f"Preço (min e max): R$ {min_price} - R$ {max_price}")
    print(f"Similaridade: {similarity_percentage}%")
    print("-" * 60)

print("\nAnálise comparativa entre o item pesquisado e os produtos recuperados:\n")

# Comparaciones usando LLaMA 2 Chat como causal
for i, descricao in enumerate(similar_products, start=1):
    
    messages = [
        {"role": "system", "content": (
            "Você é um assistente especializado em análise comparativa de produtos. "
            "Sua resposta deve ser breve, objetiva e baseada apenas nas informações fornecidas."
            "Não adicione características que não foram mencionadas."
            "Se houver informação insuficiente para comparar algum aspecto, indique isso claramente."
            "Responda exclusivamente em português."
        )},
        {"role": "user", "content":
            f"""Compare os dois produtos abaixo destacando:\n"
            "- Principais diferenças e semelhanças.\n"
            "- Como essas diferenças e semelhanças podem influenciar o valor percebido (preço esperado).\n\n"
            "Produto pesquisado: {descricao_consultada}\n"
            "Produto recuperado: {descricao}"
        """}
    ]

    # Convertir a formato que Llama-2 espera
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=350,  # un poco más de margen
            temperature=0.1,     # menor temperatura = menos invenciones
            top_p=0.9,
            do_sample=True, # False desactiva los parametros temp e top p
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    comparison = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Quitar el prompt (todo antes de la respuesta real)
    if "[/INST]" in comparison:
        comparison = comparison.split("[/INST]", 1)[-1].strip()
        
    # Eliminar tokens duplicados consecutivos (ejemplo: "produto produto")
    words = comparison.split()
    cleaned_words = []
    
    for w in words:
        if not cleaned_words or cleaned_words[-1] != w:
            cleaned_words.append(w)
    comparison = " ".join(cleaned_words)
    
    # Quitar oraciones incompletas al final (mantener solo frases terminadas en .!?)
    sentences = re.findall(r'[^.!?]*[.!?]', comparison)
    comparison = " ".join(s.strip() for s in sentences)
    
    print(f"Produto {i}:\n{comparison}\n{'-' * 60}")
