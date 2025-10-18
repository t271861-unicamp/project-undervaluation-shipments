# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 08:45:55 2025

@author: Tania

Genera una bbdd con columnas fake
"""

import numpy as np
import pandas as pd
import random
from faker import Faker
import re

# Funcion para generar URL's fake

def generar_url_fake(descripcion, categoria):
    # Limpiar texto: minúsculas, quitar caracteres especiales, reemplazar espacios por guiones
    desc_clean = re.sub(r'[^a-z0-9]+', '-', descripcion.lower())
    cat_clean = re.sub(r'[^a-z0-9]+', '-', categoria.lower())
    
    # Quitar guiones del principio/final
    desc_clean = desc_clean.strip('-')
    cat_clean = cat_clean.strip('-')
    
    # Construir URL
    return f"https://www.alibaba.com/product/{cat_clean}/{desc_clean}.html"


# Initialize Faker and set seed for reproducibility
fake = Faker()
np.random.seed(42)
random.seed(42)

# Cargar csv
csv_path = r"data_amazon\prod-elect-relog-roup-bols_limpio.csv"
prod_df = pd.read_csv(csv_path, sep=";")

# Quitamos duplicados y valores nulos
prod_df = prod_df.dropna(subset=['title', 'categoryName', 'price'])
prod_df['price'] = pd.to_numeric(prod_df['price'], errors='coerce')
prod_df = prod_df.dropna(subset=['price'])

# Parameters
n_orders = 1000  # Number of orders to generate

# Countries for Pareto distribution
shipping_countries = ['CHINA', 'INDIA', 'PAKISTAN', 'SOUTH KOREA', 'USA']

# Pareto preferences
shipping_country_weights = np.array([0.8, 0.05, 0.05, 0.05, 0.05])

# Mantiene todos los registros del CSV original
orders = []
for _, row in prod_df.iterrows():

    company = fake.company()
    description = row['title']
    category = row['categoryName']
    min_price = round(float(row['price']), 2)
    max_price = round(min_price + np.random.uniform(0, 5), 2)
    
    url = generar_url_fake(description, category)

    # Uniform distribution for ItemCount
    min_ord = np.random.randint(1, 100)

    # Pareto distribution for shipping country
    shipping_country = np.random.choice(shipping_countries, p=shipping_country_weights)

    orders.append([
        description, min_ord, min_price, max_price, category, company, shipping_country, url 
    ])

# Create DataFrame
df = pd.DataFrame(orders, columns=[
    'Description', 'MinOrder', 'MinPrice', 'MaxPrice', 'Category', 'Manufacturer', 'ShipCountry', 'url' 
])

# Save to CSV
output_path = r"data_amazon/prod-elect-relog-roup-bols_fake.csv"
df.to_csv(output_path, index=False, sep=';')
print("Archivo guardado exitosamente en:", output_path)
print(df.head())
