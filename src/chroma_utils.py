# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:02:49 2024

@author: Tania

Orden de ejecucion para la creacion de la colección: 3°
"""

import pathlib
import chromadb
from chromadb.utils import embedding_functions
from more_itertools import batched # divide una lista o iterable en lotes (batches) de tamaño fijo
from chromadb.errors import InvalidCollectionException

def build_chroma_collection(
    chroma_path: pathlib.Path,
    collection_name: str,
    embedding_func_name: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    distance_func_name: str = "cosine",
):
    """Create a ChromaDB collection"""

    chroma_client = chromadb.PersistentClient(chroma_path)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_func_name)
    
    try:
        collection = chroma_client.get_collection(collection_name)
        
    except InvalidCollectionException:
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_func,
            metadata={"hnsw:space": distance_func_name},)
        
        document_indices = list(range(len(documents))) # lista de 0 a 5869
        
        # Divide la lista en lotes de 166 elementos cada uno
        # Primer lote: [0, 1, 2, ..., 165]
        for batch in batched(document_indices, 166): # para los 5870 documentos, se crearán 35 lotes de 166 y 1 lote con el resto
            start_idx = batch[0] # índice inicial de un lote
            end_idx = batch[-1] # índice final del mismo
            
            collection.add(
                ids=ids[start_idx:end_idx],
                documents=documents[start_idx:end_idx],
                metadatas=metadatas[start_idx:end_idx],)
            
"""Cada lote de documentos, IDs, y metadata se agrega a la colección en cada iteración del bucle"""
