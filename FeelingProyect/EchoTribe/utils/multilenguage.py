import spacy
import re

# Cargar los modelos de spaCy
nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp_es = spacy.load("es_core_news_sm", disable=["parser", "ner"])

PUNCTUATION_BEFORE = re.compile(r'(\w)([.,!?;:])')
PUNCTUATION_AFTER = re.compile(r'([.,!?;:])(\w)')

stop_words_en = set(spacy.lang.en.stop_words.STOP_WORDS) - {"not", "no", "never", "ever","bad"}
stop_words_es = set(spacy.lang.es.stop_words.STOP_WORDS) - {"no", "nunca", "jamás","mejor","peor","sé","muy","poco"}

def tokenice_and_lemati(text, process, language):
    # Validar idioma
    if language == 'en':
        nlp = nlp_en
        stop_words = stop_words_en
    elif language == 'es':
        nlp = nlp_es
        stop_words = stop_words_es
    else:
        raise ValueError("Idioma no soportado. Usa 'en' o 'es'.")
    
    # Preprocesamiento
    text = text.lower()
    text = PUNCTUATION_BEFORE.sub(r'\1 \2', text)
    text = PUNCTUATION_AFTER.sub(r'\1 \2', text)

    doc = nlp(text)
    tokens = [
        token.lemma_ if process == 1 else token.text
        for token in doc if token.is_alpha and token.text not in stop_words
    ]

    return " ".join(tokens)