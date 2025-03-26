from gensim.models import KeyedVectors
from django.conf import settings

# Cargar el modelo de embeddings
fasttext_en = KeyedVectors.load_word2vec_format(settings.EMBEDDING_MODEL_EN_PATH, binary=True)