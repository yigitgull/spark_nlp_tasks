
  import com.johnsnowlabs.nlp.DocumentAssembler
  import com.johnsnowlabs.nlp.annotators.{PunctuationRemover, Tokenizer}
  import org.apache.spark.ml.Pipeline
  import org.apache.spark.sql.SparkSession




  object PunctuationRemoverMain extends App {

    val spark = SparkSession.builder()
      .appName("PunctuationTokenizer")
      .master("local")
      .getOrCreate()


    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")


    val punctuationRemover = new PunctuationRemover()
      .setInputCols("token")
      .setOutputCol("punctuation_removed")


    val pipeline = new Pipeline().setStages(Array(
      documentAssembler,
      tokenizer,
      punctuationRemover
    ))

    import spark.implicits._

    val data = Seq("\"The quick brown fox jumps over the lazy dog, but the dog doesn't seem to mind; it simply yawns, stretches, and goes back to sleep!\"")
      .toDF("text")
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("punctuation_removed.result").show(truncate=false)




  }
