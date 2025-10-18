# projeto-subvaloracao-remessas
Repositório que contém os arquivos utilizados no desenvolvimento do projeto de mestrado intitulado: Metodologia para detecção de fraude na importação de remessas internacionais: identificação de casos de subvaloração de mercadorias. 

# Descrição do Projeto

O projeto tem como objetivo desenvolver uma metodologia para auxiliar na detecção de fraudes de subvaloração em remessas internacionais, utilizando modelos de linguagem de grande escala (LLMs) e técnicas de Retrieval-Augmented Generation (RAG).  

A metodologia proposta abrange:
- Geração sintética de Declarações de Importação de Remessa (DIRs) com o uso de LLMs.  
- Construção de uma base de dados vetorial sintetica de produtos para fins comparativos.  
- Aplicação da técnica RAG no processo de verificação do valor declarado.  
- Avaliação comparativa entre diferentes LLMs.  
- Mapeamento do processo operacional de fiscalização por meio de entrevistas com atores dos setores público e privado.

# Estrutura do Repositório
```
projeto-subvaloracao-remessas/
├── data/ # Dados sintéticos e arquivos de produtos legitimos
│ ├── prod-elect-relog-roup-bols_limpio.csv # Arquivo de entrada para criação do dataset sintético de produtos
│ ├── prod-elect-relog-roup-bols_fake.csv # Dataset sintético de produtos
│ ├── prod-elect-relog-roup-bols_fake_sem_desc_val.csv # Dataset sintético de DIRs, sem geração de descrições ou valores
│ ├── dir-elect-relog-roup-bols_fake_sem_desc_com_val.csv # Dataset sintético de DIRs, com valores gerados por LLM, sem geração de descrições
│ ├── dir-elect-relog-roup-bols_fake_com_desc_com_val.csv # Dataset sintético de DIRs, com valores e descrições gerados por LLM
│ ├── dir-elect-relog-roup-bols_fake_com_desc_com_val_limpio.csv  # Dataset sintético de DIRs, com valores e descrições gerados por LLM, versão limpa
│ ├── relatorio_DIR_selecionada.csv # Lista de DIRs com valores estimados e índice de similaridade
| └── analisis_res_selecao_DIR.xlsx # Resumo em Excel das DIRs selecionadas
├── src/ # Códigos-fonte (.py)
│ ├── limpieza_csv_pandas.py # Limpeza e processamento de texto
│ ├── db_fake.py # Geração do conjunto de dados sintéticos de produtos
│ ├── prod_data_etl_dbfake.py # Função que prepara os dados para vetorização
│ ├── prod_dict_dbfake.py # Teste do script prod_data_etl_dbfake.py
| ├── chroma_utils.py # Criação da coleção chroma
| ├── RAG_prod_search_llama.py # Busca e comparação de produtos similares. Etapa de verificação do valor da remessa
| ├── db_fake_DIR.py # Geração do conjunto de dados sintéticos de DIRs (sem gerar os campos de descrição e valor)
| ├── llm_desc.py # Geração do campo descrição com LLM
| ├── llm_val.py # Geração do campo valor com LLM
| └── seleccion_DIRs.py # Seleção de DIRs suspeita de subvaloração. Etapa de seleção de DIRs
├── requirements.txt # Lista de dependências do projeto
└── README.md # Este arquivo
```
# Clonar o repositório
   bash
   git clone https://github.com/SEU_USUARIO/projeto-subvaloracao-remessas.git \
   cd projeto-subvaloracao-remessas

# Instalar dependências
 pip install -r requirements.txt

# Executar os scripts principais
 python src/RAG_prod_search_llama.py \
 python src/seleccion_DIRs.py

Projeto desenvolvido no âmbito do Mestrado em Engenharia de Produção e de Manufatura, FCA, UNICAMP, sob orientação dos prof. Cristiano Morini e Anibal Tavarez de Azevedo
Desenvolvido por Tania Lujan Alaya
