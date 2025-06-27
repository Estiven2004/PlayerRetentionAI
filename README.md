# PlayerRetentionAI 🎮🤖

Sistema inteligente basado en Machine Learning para predecir y analizar la retención de jugadores en videojuegos online. Este proyecto implementa múltiples algoritmos de clasificación para entender patrones de engagement y predecir el comportamiento futuro de los jugadores.

## 📋 Descripción del Proyecto

PlayerRetentionAI es un sistema de análisis predictivo que utiliza datos de comportamiento de jugadores para clasificar su nivel de engagement en cuatro categorías:
- **High**: Alto engagement
- **Medium High**: Engagement medio-alto
- **Medium Low**: Engagement medio-bajo  
- **Low**: Bajo engagement

El proyecto incluye análisis exploratorio de datos, preprocesamiento avanzado, implementación de múltiples algoritmos de ML y evaluación comparativa de rendimiento.

## 🗂️ Estructura del Proyecto

```
PlayerRetentionAI/
├── README.md                    # Documentación del proyecto
├── analisis_dataset.ipynb      # Análisis exploratorio inicial
├── preprocesamiento.ipynb      # Limpieza y preparación de datos
├── P2_IA.ipynb                 # Implementación de modelos ML
├── dataset_v1.pkl              # Dataset sin outliers, sin balanceo, sin escalado
├── dataset_v2.pkl              # Dataset sin outliers, balanceado, sin escalado
├── dataset_v3.pkl              # Dataset con outliers, sin balanceo, sin escalado
├── dataset_v4.pkl              # Dataset con outliers, balanceado, sin escalado
├── dataset_v5.pkl              # Dataset sin outliers, sin balanceo, escalado
├── dataset_v6.pkl              # Dataset sin outliers, balanceado, escalado
├── dataset_v7.pkl              # Dataset con outliers, sin balanceo, escalado
└── dataset_v8.pkl              # Dataset con outliers, balanceado, escalado
```

## 🔬 Metodología

### 1. Análisis Exploratorio (`analisis_dataset.ipynb`)
- Inspección inicial del dataset de comportamiento de jugadores
- Identificación de valores nulos y duplicados
- Redefinición de la variable objetivo `EngagementLevel`
- División de la clase "Medium" en "Medium_High" y "Medium_Low" basado en logros desbloqueados

### 2. Preprocesamiento (`preprocesamiento.ipynb`)
- **Codificación de variables categóricas**: One-Hot Encoding para variables nominales
- **Tratamiento de outliers**: Método IQR con opciones de eliminación completa o parcial
- **Balanceo de clases**: Submuestreo para equilibrar las categorías de engagement
- **Escalado de características**: StandardScaler para normalización
- **Generación de 8 versiones del dataset** con diferentes combinaciones de preprocesamiento

### 3. Modelado y Evaluación (`P2_IA.ipynb`)
Implementación y comparación de múltiples algoritmos:

#### 🌳 Decision Trees
- Criterio Gini con profundidad máxima de 5
- Análisis de importancia de características
- Visualización del árbol de decisión

#### 🔍 K-Nearest Neighbors (KNN)
- k=4 vecinos con métrica Minkowski (p=2)
- Escalado de características requerido
- Visualización de fronteras de decisión con PCA

#### 🚀 Support Vector Machines (SVM)
- Kernel lineal para clasificación multiclase
- Evaluación de rendimiento en diferentes versiones del dataset

#### 🧠 Redes Neuronales (TensorFlow/Keras)
- Arquitectura de red neuronal profunda
- Callbacks de EarlyStopping y ReduceLROnPlateau
- Entrenamiento por 100 épocas con optimización Adam

## 📊 Características del Dataset

El dataset incluye las siguientes variables:
- **Variables demográficas**: Age, Gender, Location
- **Comportamiento de juego**: SessionsPerWeek, AvgSessionDurationMinutes, PlayerLevel, AchievementsUnlocked
- **Preferencias**: GameGenre, GameDifficulty, InGamePurchases
- **Métricas de engagement**: PlayTimeHours
- **Variable objetivo**: EngagementLevel (4 clases)

## 🎯 Algoritmos Implementados

1. **Decision Tree Classifier**
   - Interpretabilidad alta
   - Análisis de importancia de características
   - Visualización gráfica del modelo

2. **K-Nearest Neighbors**
   - Algoritmo lazy learning
   - Sensible a la escala de datos
   - Bueno para patrones locales

3. **Support Vector Machine**
   - Efectivo en espacios de alta dimensionalidad
   - Robusto contra overfitting
   - Kernel lineal para eficiencia

4. **Redes Neuronales**
   - Capacidad de modelar relaciones complejas
   - Optimización automática de hiperparámetros
   - Regularización integrada

## 📈 Métricas de Evaluación

Para cada modelo se calculan:
- **Accuracy**: Precisión general del modelo
- **Precision**: Precisión por clase
- **Recall**: Sensibilidad por clase  
- **F1-Score**: Media armónica de precision y recall
- **Classification Report**: Reporte detallado por clase
- **Confusion Matrix**: Matriz de confusión para análisis de errores

## 🛠️ Tecnologías Utilizadas

- **Python 3.x**
- **Pandas**: Manipulación de datos
- **NumPy**: Computación numérica
- **Scikit-learn**: Algoritmos de ML y métricas
- **TensorFlow/Keras**: Redes neuronales
- **Matplotlib/Seaborn**: Visualización
- **Pickle**: Serialización de datasets

## 🚀 Cómo Ejecutar

1. **Instalar dependencias**:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

2. **Ejecutar notebooks en orden**:
   - `analisis_dataset.ipynb` - Análisis exploratorio
   - `preprocesamiento.ipynb` - Preparación de datos
   - `P2_IA.ipynb` - Entrenamiento y evaluación de modelos

3. **Cargar datasets preprocesados**:
```python
import pickle
with open('dataset_v1.pkl', 'rb') as f:
    dataset = pickle.load(f)
```

## 📊 Resultados Destacados

- Comparación sistemática de 4 algoritmos diferentes
- Evaluación en 8 variaciones del dataset
- Análisis del impacto del preprocesamiento en el rendimiento
- Identificación de características más importantes para predicción
- Optimización de hiperparámetros automática en redes neuronales

## 🎯 Aplicaciones Prácticas

- **Retención de jugadores**: Identificar jugadores en riesgo de abandono
- **Personalización**: Ajustar experiencia según nivel de engagement
- **Marketing dirigido**: Campañas específicas por segmento de jugador
- **Optimización de monetización**: Estrategias diferenciadas por tipo de jugador
- **Desarrollo de juegos**: Insights para diseño de mecánicas de engagement

## 👥 Equipo

**Práctica 2 - Grupo 1 - Equipo 7**  
Proyecto de Inteligencia Artificial

## 📄 Licencia

Este proyecto es desarrollado con fines académicos como parte de un curso de Inteligencia Artificial.

---

*Desarrollado con ❤️ para entender mejor el comportamiento de los jugadores en videojuegos online*
