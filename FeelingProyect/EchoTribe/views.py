import os
from django.conf import settings
import joblib
from django.http import JsonResponse
import json
from .utils.script import cleanText

# Construye la ruta absoluta al archivo
modelo_path = os.path.join(settings.BASE_DIR, 'AI', 'modelo_svm.pkl')
vectorizador_path = os.path.join(settings.BASE_DIR, 'AI', 'vectorizer_tfidf.pkl')

# Carga el modelo y el vectorizador
modelo = joblib.load(modelo_path)
vectorizador = joblib.load(vectorizador_path)

def evaluar_texto(request):
    if request.method == 'POST':
        try:
            texto = json.loads(request.body)#obteniendo el; texto
            texto_limpio = cleanText(texto)

            # Vectorizamos el texto
            texto_vectorizado = vectorizador.transform([texto_limpio])

            # Realizamos la predicción
            prediccion = modelo.predict(texto_vectorizado)

            # Convertir el array de NumPy a una lista
            prediccion_exponer = prediccion.tolist()

            return JsonResponse({
                'success': True,
                'message': 'Hemos procesado el texto con éxito',
                'prediccion': prediccion_exponer  # Usar la lista en lugar del array
            })
        except KeyError as e:
            # Convertir la excepción a una cadena
            return JsonResponse({'success': False, 'message': str(e)}, status=400)
        except Exception as e:
            # Convertir cualquier otra excepción a una cadena
            return JsonResponse({'success': False, 'message': str(e)}, status=500)
        

