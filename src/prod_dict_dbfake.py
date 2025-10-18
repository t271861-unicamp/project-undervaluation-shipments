# -*- coding: utf-8 -*-
"""
Created on Mon May  5 10:30:59 2025

@author: Tania
Orden de ejecucion para la creacion de la colección: 2°
"""

from prod_data_etl_dbfake import prepare_prod_data
DATA_PATH = "data_amazon/prod-elect-relog-roup-bols_fake.csv"

chroma_prod_data_dict = prepare_prod_data(DATA_PATH)
print(chroma_prod_data_dict.keys())
# dict_keys(['ids', 'documents', 'metadatas'])

print(chroma_prod_data_dict["ids"][-10]) 
# description990

print(chroma_prod_data_dict["documents"][-10])
# Smart TV 43 4K LG UHD THINQ AI 43UR7800PSA HDR Bluetooth Alexa Google Assistant Airplay2 3 HDMI

print(chroma_prod_data_dict["metadatas"][-10])
# {'MinOrder': 51, 
# 'MinPrice': 1899.0, 
# 'MaxPrice': 1899.32, 
# 'Category': 'TV, audio and cinema at home', 
# 'Manufacturer': 'Anderson Inc', 
# 'ShipCountry': 'CHINA', 
# 'url': 'https://www.alibaba.com/product/tv-audio-e-cinema-em-casa/smart-tv-43-4k-lg-uhd-thinq-ai-43ur7800psa-hdr-bluetooth-alexa-google-assistente-airplay2-3-hdmi.html'}
