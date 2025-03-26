import fasttext  # Para generar embeddings
from django.conf import settings

#Modelo de embedding para espaniol
fasttext_es = fasttext.load_model(settings.EMBEDDING_MODEL_ES_PATH)