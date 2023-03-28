# Modelo Predictivo de clasificación
Creación de un modelo para predecir si un cliente abandonará el banco o no.

# Contenido
* Introducción
    * Objetivo
* Iniciación de datos
    * Examina datos
* Procesamiento datos
    * Pre-procesamiento de datos
        * Valores ausentes
    * Procesamiento de características
        * Codificación One-Hot
        * Escalado de caracteristicas
* Exploración gráfica de datos
* Entrenar un modelo
    * Modelo piloto con clases desequilibradas
        * Prueba de consistencia
    * Equilibrio de clases
        * Sobremuestreo
        * Submuestreo
    * Entrenamiento de modelos y mejora la calidad del modelo
        * 1-Algoritmo DecisionTreeClassifier
        * 2-Algoritmo Random Forest Classifier
        * 3-Algoritmo Logistic Regression
    * Curva ROC
* Conclusión general


## Librerías usadas:

- import pandas as pd
- import numpy as np
- import matplotlib.pyplot as plt
- from sklearn.preprocessing import StandardScaler
- from sklearn.model_selection import train_test_split
- from sklearn.tree            import DecisionTreeClassifier
- from sklearn.ensemble        import RandomForestClassifier
- from sklearn.linear_model    import LogisticRegression
- from sklearn.metrics import accuracy_score
- from sklearn.metrics import f1_score
- from sklearn.metrics import roc_curve
- from sklearn.metrics import roc_auc_score 
- from sklearn.utils import shuffle
