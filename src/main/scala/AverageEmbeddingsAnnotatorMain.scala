
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{BertSentenceEmbeddings, RoBertaSentenceEmbeddings, SentenceDetector}
import com.johnsnowlabs.nlp.annotators.{Lemmatizer, PunctuationRemover, Tokenizer}
import com.johnsnowlabs.nlp.embeddings.{XlmRoBertaEmbeddings, XlmRoBertaSentenceEmbeddings}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession


   object AverageEmbeddingsAnnotatorMain extends App {

     val spark = SparkSession.builder()
       .appName("PunctuationTokenizer")
       .master("local")
       .getOrCreate()


     val tokenizer = new Tokenizer()
       .setInputCols("sentence")
       .setOutputCol("token")

     val lemmatizer = new Lemmatizer()
       .setInputCols("token")
       .setOutputCol("lemma")

     val bertEmbeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_128")
       .setInputCols("sentence")
       .setOutputCol("bert_sentence_embeddings")

     val roBertaEmbeddings = RoBertaSentenceEmbeddings.pretrained()
       .setInputCols("sentence")
       .setOutputCol("RoBerta_sentence_embeddings")
       .setCaseSensitive(true)

     val embeddingsAnnotator = new AverageEmbeddingsAnnotator("")
       .setInputCols("bert_sentence_embeddings", "roBerta_sentence_embeddings")
       .setOutputCol("embeddings")


     import spark.implicits._

     val data = Seq("my name is yigit.ı am developer").toDF("text")

     val documentAssembler = new DocumentAssembler()
       .setInputCol("text")
       .setOutputCol("document")

     val sentenceDetector = new SentenceDetector()
       .setInputCols("document")
       .setOutputCol("sentence")

     val pipeline = new Pipeline()
       .setStages(Array(
         documentAssembler,
         sentenceDetector,
         tokenizer,
         bertEmbeddings,
         roBertaEmbeddings,
         embeddingsAnnotator

       ))

     val result = pipeline.fit(data).transform(data)
     result.selectExpr("explode(bert_sentence_embeddings) as bert").show(false)
     result.selectExpr("explode(roBerta_sentence_embeddings) as deBerta").show(false)
     result.selectExpr("explode(embeddings) as result").show(false)







   }
