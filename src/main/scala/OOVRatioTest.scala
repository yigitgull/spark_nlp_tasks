

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{BertSentenceEmbeddings, RoBertaEmbeddings, WordEmbeddings, WordEmbeddingsModel}
import com.johnsnowlabs.nlp.annotators.{PunctuationRemover, Tokenizer}
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.util.io.ReadAs
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession


object OOVRatioMain extends App {

  val spark = SparkSession.builder()
    .appName("OOVRatioAnnotator")
    .master("local")
    .getOrCreate()


  val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

  val wordEmbeddings = RoBertaEmbeddings.pretrained()
  .setInputCols("document", "token")
  .setOutputCol("embeddings")

  val oovRatioAnnotator = new OOVRatioAnnotator()
    .setInputCols("embeddings")
    .setOutputCol("res")




  val pipeline = new Pipeline().setStages(Array(
    documentAssembler,
    tokenizer,
    wordEmbeddings,
    oovRatioAnnotator
  ))

  import spark.implicits._

  val data = Seq("the quick 1 brown ? + 45 ")
    .toDF("text")
  val result = pipeline.fit(data).transform(data)

  result.selectExpr("res.metadata").show(truncate = false)


}
