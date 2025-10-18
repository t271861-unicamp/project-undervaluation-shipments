# -*- coding: utf-8 -*-
"""
Created on Mon May  5 10:16:18 2025

@author: Tania
Orden de ejecucion para la creacion de la colección: 1°
"""
# Prepara la base de datos fake para ser vectorizada con ChromaDB

import pandas as pd
import pathlib

def prepare_prod_data(data_path: pathlib.Path):
    
    product_data = pd.read_csv(data_path, sep=";")
    
    # Create ids, documents, and metadatas data in the format chromadb expects
    ids = [f"description{i}" for i in range(product_data.shape[0])]
    documents = product_data["Description"].to_list()
    metadatas = product_data.drop(columns=["Description"]).to_dict(orient="records")

    return {"ids": ids, "documents": documents, "metadatas": metadatas}
