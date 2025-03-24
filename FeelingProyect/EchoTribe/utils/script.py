import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
import emoji
import re

#funcion de lematizacion y tokenizacion
lemmatizer = WordNetLemmatizer()
# Obtener stopwords en inglés
stop_words = set(stopwords.words('english'))

customs_stop_wards = stop_words - {"not", "no", "never", "ever"}#dejamos habilitadas estas palabras

def tokenice_and_lemati(text):
    tokens = word_tokenize(text.lower())#convertimos todo a minusc y luego lo tokenizamos jijijij
    filter_tokens = [token for token in tokens if token not in customs_stop_wards]
    lematized_tokens = [lemmatizer.lemmatize(token, pos='v') for token in filter_tokens]
    return ' '.join(lematized_tokens)

def cleanText(texto):
	texto = emoji.demojize(texto, delimiters=(" ", " ")).replace("_", " ")#convirtiendo emojia a letras
	texto_limpio = re.sub(r'[^a-zA-Z0-9\s]', '', texto)#quitar carcetres raros
	texto_limpio = tokenice_and_lemati(texto_limpio)
	return texto_limpio