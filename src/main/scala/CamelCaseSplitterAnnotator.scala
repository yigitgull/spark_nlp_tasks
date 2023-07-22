import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

/**
 * import org.apache.spark.sql.SparkSession
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

  val data = Seq("CamelCase WikiCase PascalCase camel case 22Feb2023 ")
  val dataFrame = spark.createDataFrame(data.map(Tuple1.apply)).toDF("text")

  val model = pipeline.fit(dataFrame)
  val lightPipeline = new LightPipeline(model)

  lightPipeline.transform(dataFrame).show(false)

}

 +-------------------------------------------------------------------------+------------------------------------------------------------------------------------------------+
|text                                                                     |tokens                                                                                          |
+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------------+
|CamelCase WikiCase PascalCase camel case InterCaps or WikiCase 22Feb2023 |[Camel, Case, Wiki, Case, Pascal, Case, camel, case, Inter, Caps, or, Wiki, Case, 22, Feb, 2023]|
+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------------+


 */
class CamelCaseSplitterAnnotator(override val uid: String) extends Transformer with Params with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("CAMELCASE_SPLITTER"))

  val inputCol: Param[String] = new Param[String](this, "inputCol", "The input column")
  val outputCol: Param[String] = new Param[String](this, "outputCol", "The output column")

  def getInputCol: String = $(inputCol)

  def setInputCol(value: String): this.type = set(inputCol, value)

  def getOutputCol: String = $(outputCol)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  /**
   * Splits CamelCase according to regex
   */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val inputColName = getInputCol
    val outputColName = getOutputCol

    val assembleTokensUdf = udf { (text: String) =>
      text.split(" ").flatMap(a => a.split("""(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[0-9])(?=[A-Z][a-z])|(?<=[a-zA-Z])(?=[0-9])"""))
    }

    val tokens = dataset.withColumn(outputColName, assembleTokensUdf(col(inputColName)))

    tokens
  }

  /**
   * Checks and transforms schema
   */

  override def transformSchema(schema: StructType): StructType = {
    val inputColName = getInputCol
    val outputColName = getOutputCol

    require(schema.fieldNames.contains(inputColName),
      s"Input column '$inputColName' does not exist in the schema.")
    require(!schema.fieldNames.contains(outputColName),
      s"Output column '$outputColName' exists in the schema.")

    val outputColDataType = ArrayType(StringType)
    val outputCol = StructField(outputColName, outputColDataType, nullable = false)
    val outputFields = schema.fields :+ outputCol
    StructType(outputFields)
  }

  override def copy(extra: org.apache.spark.ml.param.ParamMap): CamelCaseSplitterAnnotator = defaultCopy(extra)
}

