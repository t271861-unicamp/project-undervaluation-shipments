# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 15:55:31 2025

@author: Admin
"""

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, util

# ===============================
# Configuración
# ===============================
CHROMA_PATH = "prod_fakedb_embeddings"
COLLECTION_NAME = "prod-elect-relog-roup-bols_fake"
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"

# Modelo de embeddings
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Cliente ChromaDB
client = chromadb.PersistentClient(CHROMA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

# ===============================
# Función para calcular valor probable
# ===============================
def calcular_valor_probable(descricao_consultada, n_results=5):
    results = collection.query(
        query_texts=[descricao_consultada],
        n_results=n_results,
        include=["documents", "metadatas"]
    )

    similar_products = results["documents"][0]
    metadatas = results["metadatas"][0]

    query_embedding = embedder.encode(descricao_consultada, convert_to_tensor=True)

    valores_ponderados = []
    similitudes = []
    registros_log = []

    for descricao, meta in zip(similar_products, metadatas):
        product_embedding = embedder.encode(descricao, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(query_embedding, product_embedding).item()

        try:
            min_price = float(meta["MinPrice"])
            max_price = float(meta["MaxPrice"])
            valor_medio = (min_price + max_price) / 2
        except (ValueError, TypeError):
            continue 

        valores_ponderados.append(valor_medio * similarity_score)
        similitudes.append(similarity_score)

        registros_log.append({
            "Descricao_consultada": descricao_consultada,
            "Descricao_similar": descricao,
            "Similaridade": round(similarity_score, 4),
            "MinPrice": min_price,
            "MaxPrice": max_price,
            "Valor_medio": round(valor_medio, 2)
        })

    if not similitudes:
        return None, None, registros_log

    valor_probable = sum(valores_ponderados) / sum(similitudes)
    valor_probable = round(valor_probable, 2)
    similaridade_media = round(sum(similitudes) / len(similitudes), 4)

    return valor_probable, similaridade_media, registros_log


# ===============================
# Procesar CSV de entrada
# ===============================
csv_input = "data_amazon/dir-elect-relog-roup-bols_fake_com_desc_com_val_limpio.csv"
csv_output = "data_amazon/relatorio_DIR_selecionada.csv"
log_output = "data_amazon/log_descripciones_usadas.csv"

df = pd.read_csv(csv_input, sep=";")

valores_estimados = []
similaridades_medias = []
logs_totales = []

for desc in df["Descricao"]:
    valor, sim_media, registros_log = calcular_valor_probable(desc, n_results=5)
    valores_estimados.append(valor)
    similaridades_medias.append(sim_media)
    logs_totales.extend(registros_log)

# Agregar columnas
df["Valor_estimado"] = valores_estimados
df["Similaridade_media"] = similaridades_medias

# Guardar archivos
df.to_csv(csv_output, sep=";", index=False)

log_df = pd.DataFrame(logs_totales)
log_df.to_csv(log_output, sep=";", index=False)

print(f"Arquivo gerado: {csv_output}")
print(f"Log gerado: {log_output}")
