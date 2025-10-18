# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:25:13 2025

@author: Tania
"""

import pandas as pd
from unidecode import unidecode
import ftfy

def limpiar_texto_pandas(column):
    # Limpia una columna de texto en Pandas.
    return (
        column
        .fillna('')  # Rellena valores NaN con una cadena vacía
        .astype(str)  # Convierte todos los valores a cadenas
        .apply(ftfy.fix_text)  # Corrige problemas de codificación
        .apply(unidecode)  # Elimina acentos y caracteres especiales
        .str.replace(r'\s+', ' ', regex=True)  # Elimina espacios extra
        .str.strip()  # Elimina espacios al inicio y final
    )

DATA_PATH = "data_amazon/dir-elect-relog-roup-bols_fake_com_desc_com_val_sin_comillas.csv"
OUTPUT_PATH = "data_amazon/dir-elect-relog-roup-bols_fake_com_desc_com_val_limpio.csv"

product_data = pd.read_csv(DATA_PATH, sep=";")

#print(product_data.columns)
#print(product_data.dtypes)

# Limpia la columna 'Title'
product_data['Descricao'] = limpiar_texto_pandas(product_data['Descricao'])
# Limpia la columna 'Category'
# product_data['categoryName'] = limpiar_texto_pandas(product_data['categoryName'])

# Elimina filas cuya descripción comience con frases no deseadas
frases_invalidas = [
    "Sure, I'd be happy to help",
    "Sure, here is a",
    "Sure, here's a",
    "Sure, here are",
    "Sure!",
    "Sure, I can",
    "Here is a concise version",
    "I apologize"
]

mask = ~product_data['Descricao'].str.startswith(tuple(frases_invalidas))
product_data = product_data[mask]

# Elimina filas con Valor_total_remessa nulo o igual a 0
if "Valor_total_remessa" in product_data.columns:
    product_data = product_data[
        product_data["Valor_total_remessa"].notna() & 
        (product_data["Valor_total_remessa"] != 0)
    ]
    
# Guarda el DataFrame limpio en un nuevo archivo CSV
product_data.to_csv(OUTPUT_PATH, sep=";", index=False)

# Muestra el DataFrame con la columna transformada
print("Archivo limpio guardado en:", OUTPUT_PATH)

