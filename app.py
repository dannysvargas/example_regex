import joblib
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

import re
import json

from typing import List

# Descargar os recursos necesarios para usar NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Folder POS tagger para portuguese - NLTK
nltk_tagger = joblib.load('POS_tagger_brill.pkl')

class PreprocessTextos:
    """ 
    Clase para implementar metodos de normalização e preprocessamento de lista texto. 
    """
    def __init__(self, textos):
        self.textos = textos
        self.nlp_nltk = nltk_tagger

    def preprocess(self) -> List[str]:
        """ Preprocessar texto segundo seja necessario. """
        # implementar segundo seja necessario
        #self.textos = [texto.replace(' - ', '-') for texto in self.textos]
        # lowercase
        self.textos = [texto.lower() for texto in self.textos]
        # remove multiple spaces
        self.textos = [re.sub(r'\s+', ' ', texto) for texto in self.textos]
        # remover links
        self.textos = [replace_links(texto, 'regexlink') for texto in self.textos]
        # remover (") chr(34) no começo e no final de uma linha (texto). 
        self.textos = [remove_chr34(texto) for texto in self.textos]
        # replace valores
        self.textos = [replace_valores(texto, 'regexvalor') for texto in self.textos]
        # replace data
        self.textos = [replace_data(texto, 'regexdata') for texto in self.textos]
        # replace cpf
        self.textos = [replace_cpf(texto, 'regexcpf') for texto in self.textos]
        # replace cnpj
        self.textos = [replace_cnpj(texto, 'regexcnpj') for texto in self.textos]

        return self.textos

    def split_nltk_listsentences(self) -> List[List[str]]:
        """ Dividir texto em sentenças usando NLTK. """
        textos_preprocessado = self.preprocess()
        list_sentencas = [ sent_tokenize(texto_preprocessado, language='portuguese')
                     for texto_preprocessado in textos_preprocessado]
        return list_sentencas

    def _split_nltk_tokens(self, texto: str) -> List[str]:
        """ Dividir sentenças em tokens usando NLTK. """
        tokens = word_tokenize(texto, language='portuguese')
        return tokens

    def split_nltk_tokens_list(self, textos:List[str]) -> List[List[str]]:
        """ Dividir cada texto de uma lista de textos em tokens"""
        result = [self._split_nltk_tokens(texto) for texto in self.textos]
        return result 


def remove_links(text:str) -> str:
    """Função que remove links de um texto como preprocessamento."""
    # pattern
    url_pattern = r'https?://\S+|www\.\S+'
    texto_limpo = re.sub(url_pattern, '', text, flags=re.IGNORECASE)
    return texto_limpo

def remove_chr34(texto:str) -> str:
    """Função que remove sequencia 'char(34)' no inicio e no fim"""
    pattern_ini = r'^chr\(34\)'
    pattern_end = r'chr\(34\)$'
    texto_limpo = re.sub(pattern_ini, '', texto)
    texto_limpo = re.sub(pattern_end, '', texto_limpo)
    return texto_limpo 

def replace_links(text:str, to_replace:str) -> str:
    """Função para substituir links de um texto como preprocessamento."""
    # pattern
    pattern = r'https?://\S+|www\.\S+'
    texto_mod = re.sub(pattern, to_replace, text, flags=re.IGNORECASE)
    return texto_mod 

def replace_valores(text:str, to_replace:str) -> str:
    """Função que substituir valores de um texto como preprocessamento."""
    # pattern 
    pattern = r'\b\d{1,3}(?:\.\d{3})*,\d{1,2}\b'
    texto_mod = re.sub(pattern, to_replace, text, flags=re.IGNORECASE)
    return texto_mod 

def replace_data(text:str, to_replace:str) -> str:
    """Função que substituir datas de um texto como preprocessamento."""
    # pattern 
    pattern = r'\b(\d{2}\.\d{2}\.\d{4}|\d{2}\/\d{2}\/\d{4}|\d{2}\.\d{4}|\d{2}\/\d{4}|\d{2}\/\d{2}\/\d{2}|\d{2}\/\d{1}\/\d{4}|\d{1}\/\d{1}\/\d{4})\b'
    texto_mod = re.sub(pattern, to_replace, text, flags=re.IGNORECASE)
    return texto_mod 

def replace_cpf(text:str, to_replace:str) -> str:
    """Função que substituir cpfs de um texto como preprocessamento."""
    # pattern 
    pattern = r'\b\d{3}\.?\d{3}\.?\d{3}(-| )?\d{2}\b'
    texto_mod = re.sub(pattern, to_replace, text, flags=re.IGNORECASE)
    return texto_mod 

def replace_cnpj(text:str, to_replace:str) -> str:
    """Função que substituir cnpjs de um texto como preprocessamento."""
    # pattern 
    pattern = r'\b0?\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b|\b0?\d{8}/?\d{6}\b'
    texto_mod = re.sub(pattern, to_replace, text, flags=re.IGNORECASE)
    return texto_mod 



if __name__ == '__main__':
    print('Arquivo python com as classes de preprocessamento e representação de textos')

    #load data
    with open('./dados.json', 'r', encoding='utf-8') as f:
        datajson = json.load(f)

    # Preprocessamento
    list_textos = [item['texto'] for item in datajson]
    print('numero de textos:', len(list_textos))

    processor = PreprocessTextos(list_textos)
    processor.preprocess()

    print('mostrar os 7 textos preprocessados')
    for texto_prep in processor.textos:
        print('\n',texto_prep)

