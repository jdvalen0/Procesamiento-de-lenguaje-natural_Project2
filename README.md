# Caso de Estudio y Laboratorio: Fine-Tuning de Modelos NLP

## Resumen Ejecutivo

El despliegue corporativo de Inteligencia Artificial enfrenta un dilema persistente: ceder la privacidad de los datos enviando informacion sensible a APIs cerradas administradas por terceros, o entrenar modelos fundacionales In-House bajo estrictas normas de seguridad. 

Este proyecto demuestra una infraestructura completa y segura utilizando el ecosistema de codigo abierto de Hugging Face. El objetivo es tomar arquitecturas Transformer (pesos publicos y auditables) y aplicarles un ajuste fino ("Fine-Tuning") sobre conjuntos de datos especificos de dominio, logrando asi proteger el secreto corporativo y mantener precisiones de vanguardia sin dependencia de infraestructura foranea.

## Dominios de Implementacion y Arquitecturas

Para evidenciar la versatilidad de la metodologia, el laboratorio aborda tres tareas transversales en el procesamiento del lenguaje:

### 1. Sequence Classification (Clasificacion de Sentimiento Bursatil)
*   **Problema:** Categorizar el sentimiento financiero de noticias corporativas (Bullish, Bearish, Neutral) para respaldar la toma de decisiones algoritmicas en mercados de valores.
*   **Modelo:** `distilbert-base-uncased` (Arquitectura Encoder-Only). Ideal para procesar la secuencia global bidireccionalmente y extraer la semantica del vector `[CLS]`.
*   **Dataset:** `zeroshot/twitter-financial-news-sentiment`.

### 2. Sequence Labeling / NER (Mineria de Entidades Nombradas)
*   **Problema:** Analizar lineas de texto estructurado y extraer taxonomias corporativas precisas (Personas, Organizaciones, Localizaciones).
*   **Modelo:** `distilbert-base-uncased`. Operando a nivel de token en clasificacion (IOB format), integrando manejo matricial avanzado (indice `-100`) para evadir el impacto en la penalizacion del tokenizador sub-palabra (WordPiece).
*   **Dataset:** `Babelscape/wikineural`.

### 3. Sequence-to-Sequence (Summarization de Interacciones)
*   **Problema:** Absorber transcripciones largas generadas en departamentos de atencion al cliente y emitir resumenes cortos accionables.
*   **Modelo:** `t5-small` (Arquitectura Encoder-Decoder). Requiere auto-atencion enmascarada cruzada y generacion autorregresiva de lenguaje.
*   **Dataset:** `knkarthick/dialogsum`.

## Ciclo MLOps de 3 Fases

El repositorio elude las asunciones manuales (Hardcoding) integrando un ciclo de MLOps completo:

1. **Fase de Grid Search (Experimentacion Agil):** Se prueba cruzadamente la Tasa de Aprendizaje (LR) contra el Weight Decay sobre un sample pequeno (500 muestras). Las ejecuciones revelaron que `LR=5e-5` supero agresivamente a `2e-5` (quien registro un colapso total de F1=0.0 en NER por falta de traccion del gradiente).
2. **Fase de Entrenamiento a Gran Escala:** Una vez descubierta la matematica optima, se instancia un modelo limpio y se entrena profundamente sobre un dataset superior (Validando las restricciones de VRAM de la GPU T4 de Google Colab, hasta los limites permitidos). El checkpoint definitivo es salvado en memoria.
3. **Puesta en Produccion e Inferencia:** Las fases culminan evaluando los pesos finales del modelo campeon contra sentencias de texto crudo usando abstracciones directas de inferencia, emulando la estructura del flujo final del servidor de negocio.

## Resultados Empiricos y Rendimiento

La ejecucion del pipeline sobre un ambiente de aceleracion de hardware (GPU T4) demostro el impacto de las leyes de escalamiento y las arquitecturas optimizadas mediante Fine-Tuning profundo:

*   **Clasificacion de Sentimiento (Tarea 1 - 100% Datos):** El modelo DistilBERT alcanzo un **F1-Score Macro definitivo de 0.8380** (y Accuracy de **87.40%**) en la época 2. Esto demuestra su resiliencia al desbalance de clases y su gran capacidad de generalización en el dominio financiero extremo.
*   **Reconocimiento de Entidades (Tarea 2 - 100% Datos):** Se obtuvo un sobresaliente **F1-Score (seqeval) de 0.9026** (con Accuracy general de **98.73%**) en la época 2, validando el éxito matemático de la alineación de sub-tokens mediante el enmascaramiento `-100` sobre WordPiece a gran escala.
*   **Generacion de Resumenes (Tarea 3 - 24% Datos / 3000 muestras):** El modelo T5-small alcanzo un **ROUGE-1 de 0.3440** (con ROUGE-2 de **0.1261** y ROUGE-L de **0.2928**) tras 3 epocas de entrenamiento. La descongestión del optimizador mediante regularización ligera (Weight Decay = 0.01) otorgó libertad paramétrica al decodificador para estructurar abstracciones complejas.

## Instrucciones de Ejecucion y Despliegue

La complejidad matricial requiere ejecucion paralelizada (Hardware Acelerado GPU/TPU). Todo el pipeline ha sido consolidado y protegido en un unico cuaderno autoejecutable.

1. Subir a Google Drive / Google Colab unicamente el archivo **`Project2_NPL_FT.ipynb`**.
2. Configurar el entorno de ejecucion en Colab con acelerador de hardware (GPU T4 o TPU).
3. Ejecutar secuencialmente las celdas para reproducir los resultados empiricos.


