import numpy as np
import os
import joblib
from langdetect import detect
from django.conf import settings 
from ...utils.EmbeddingEn import fasttext_en #importamos el embedding en ingles
from ...utils.EmbeddingEs import fasttext_es #importamos el embedding in spanish
from ...utils.multilenguage import tokenice_and_lemati
from rest_framework.decorators import api_view
from django.http import JsonResponse



# Cargando el modelo que entrenamos
modelo_path = os.path.join(settings.BASE_DIR, 'AI', 'modelo_multilingue.pkl')

##extrayendo los objetos que cargamos del modelo
objetos = joblib.load(modelo_path)
model = objetos['model']
le = objetos['label_encoder']
encoder = objetos['onehot_encoder']

@api_view(["POST"])
def predecir_sentimiento(request):
    try:
        # Extrae el texto del cuerpo de la petición
        texto = request.data.get('texto')
        idioma = detect(texto)
        modelo_embedding = fasttext_es if idioma == 'es' else fasttext_en

        #lo metemos a la funcion para tokenizar y lematizar
        texto_procesado = tokenice_and_lemati(texto,1,idioma)
    
        texto_procesado = texto.split()
    
        if idioma == 'es': 
            vectores = [modelo_embedding.get_word_vector(p) for p in texto_procesado if p]
        else:
            vectores = [modelo_embedding[p] for p in texto_procesado if p in modelo_embedding]
    
        embedding = np.mean(vectores, axis=0) if vectores else np.zeros(300)
        idioma_encoded = encoder.transform([[idioma]])
    
        X_nuevo = np.hstack([embedding.reshape(1, -1), idioma_encoded])
        sentimiento_num = model.predict(X_nuevo)[0]
    
        prediccion = le.inverse_transform([sentimiento_num])[0]
        return JsonResponse({
                    'success': True,
                    'message': 'Hemos procesado el texto con éxito',
                    'prediccion': prediccion
                })
    except KeyError as e:
        return JsonResponse({'success': False, 'message': str(e)}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)}, status=500)
   