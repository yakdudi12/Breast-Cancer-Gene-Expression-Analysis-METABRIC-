# Análisis de la Expresión Génica en Cáncer de Mama (METABRIC)  

Este proyecto realiza un análisis exploratorio y estadístico de datos (EDA) del dataset **METABRIC**, que contiene datos genómicos y clínicos de 1,980 muestras de cáncer de mama. Examina la relación entre los tratamientos y la supervivencia de los pacientes, utilizando herramientas de visualización y procesamiento de datos para comprender los factores que afectan la progresión del cáncer de mama.  

---

## 📊 Contexto  

El cáncer de mama es la forma más común de cáncer entre mujeres, con una incidencia anual de más de 2,1 millones de casos. Es responsable del mayor número de muertes relacionadas con el cáncer en mujeres.  

El objetivo principal de este análisis es **estimar con precisión la prognosis y la duración de la supervivencia**, explorando diferencias genéticas y respuestas a tratamientos. Esto puede ayudar a personalizar los tratamientos y evitar procedimientos innecesarios.  

El dataset METABRIC, fruto de una colaboración entre investigadores de Canadá y el Reino Unido, ha sido destacado en revistas científicas de prestigio como *Nature Communications* (Pereira et al., 2016).  

---

## 🛠️ Funcionalidades  

El código incluye una clase `EDA` que proporciona una variedad de métodos para realizar análisis exploratorios de datos. Entre sus principales funcionalidades se encuentran:  

### **Procesamiento de Datos**  
- **Carga de Datos:** Compatible con formatos CSV, Excel y bases de datos SQLite.  
- **Identificación de Valores Faltantes:** Visualización de valores nulos mediante mapas de calor.  
- **Detección y Eliminación de Duplicados:** Limpieza automática de datos duplicados.  

### **Análisis Exploratorio**  
- **Resumen Estadístico:** Genera estadísticas descriptivas para variables numéricas y categóricas.  
- **Visualización de Distribuciones:** Creación de diagramas de caja y gráficos de distribución categórica.  
- **Matriz de Correlación:** Visualiza las correlaciones entre variables numéricas.  

### **Análisis Específicos**  
- **Análisis por Edad:** Relación entre la edad al diagnóstico y la supervivencia global.  
- **Impacto de los Tratamientos:** Comparaciones visuales entre quimioterapia, hormonoterapia y radioterapia en términos de supervivencia.  
- **Análisis de Mortalidad:** Comparación porcentual de fallecimientos entre diferentes tratamientos.  
- **Análisis de Receptores Hormonales:** Comparación porcentual de fallecimientos y tamaños tumorales según receptores hormonales.  
- **Distribución por Etapa Tumoral:** Distribución porcentual de las etapas tumorales en la población del dataset.  

---

## 📁 Estructura del Proyecto  

```
📂 BreastCancerEDA  
 ├── 📄 main.py                   # Código principal del proyecto  
 ├── 📄 README.md                 # Este archivo  
 ├── 📂 Breast_cancer_Plots       # Gráficos generados  
 └── 📂 breast_cancer_dataset.zip # Carpeta comprimida con el dataset METABRIC  
```  

---

## 🧰 Librerías Utilizadas  

- **Pandas:** Manipulación de datos.  
- **Matplotlib:** Visualización de datos.  
- **Seaborn:** Visualizaciones avanzadas y estéticamente agradables.  
- **SQLite3:** Conexión y consulta de bases de datos.  
- **NumPy:** Operaciones matemáticas y estadísticas.  

---

## 📈 Resultados Esperados  

- **Identificación de Patrones Clave:** Relación entre características clínicas y resultados de supervivencia.  
- **Visualizaciones Significativas:** Gráficos claros y detallados que resumen el impacto de factores como la edad y los tratamientos en la supervivencia.  
- **Limpieza de Datos:** Dataset listo para análisis adicionales, incluyendo modelos predictivos.  

---

## ✒️ Referencias  

1. Pereira, B., Chin, S. F., et al. (2016). "The somatic mutation profiles of 2,433 breast cancers refine their genomic and transcriptomic landscapes." *Nature Communications*.  
2. cBioPortal: [https://www.cbioportal.org/](https://www.cbioportal.org/).  

---

## 🤝 Contribuciones  

Este proyecto forma parte de mi portafolio profesional. Si tienes sugerencias o comentarios, ¡estaré encantado de recibir tu feedback!  

---

## 📬 Contacto  

**Britez Santiago**  
[LinkedIn](https://www.linkedin.com/in/santiago-luis-britez-101a8a217)  
