// importamos una sesion en spark
import org.apache.spark.sql.SparkSession

// utilizamos las lineas de codigo para reportar errores reducidos
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// creamos una instancia de la sesion de spark
val spark = SparkSession.builder().getOrCreate()

// importamos la libreria de kmean para el algoritmo de agrupamiento
import org.apache.spark.ml.clustering.KMeans

// cargamos el dataset Wholesale customers Data
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("Wholesalecustomersdata.csv")

// seleccionamos las columnas asignadas para que inicie el conjunto de entrenamiento .
val feature_data = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")


// importamos la libreria de  VectorAssembler and Vectors
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

// creamos un nuevo objeto VectorAssembler para las columnas de caracteriticas como un conjunto de entrada recordando
//que no hay etiquetas(labels)
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")

//utilizamos el objeto assembler para transformar feature_data
// Call this new data training_data
val training_data = assembler.transform(feature_data).select("features")

// creamos un modelo Kmeand con K=3
val kmeans = new KMeans().setK(3).setSeed(1L)
