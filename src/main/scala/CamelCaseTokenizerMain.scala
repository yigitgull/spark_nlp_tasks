import org.apache.spark.sql.SparkSession
import com.johnsnowlabs.nlp.base.LightPipeline
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
object CamelCaseTokenizerMain extends App {

  val spark = SparkSession.builder()
    .appName("PunctuationTokenizer")
    .master("local")
    .getOrCreate()


  import spark.implicits._


  val CamelCaseSplitterAnnotator = new CamelCaseSplitterAnnotator()
    .setInputCol("text")
    .setOutputCol("tokens")

  val pipeline = new Pipeline().setStages(Array(
    CamelCaseSplitterAnnotator
  ))

  val data = Seq("CamelCase WikiCase PascalCase camel case InterCaps or WikiCase 22Feb2023 ")
  val dataFrame = spark.createDataFrame(data.map(Tuple1.apply)).toDF("text")

  val model = pipeline.fit(dataFrame)
  val lightPipeline = new LightPipeline(model)

  lightPipeline.transform(dataFrame).show(false)








}
