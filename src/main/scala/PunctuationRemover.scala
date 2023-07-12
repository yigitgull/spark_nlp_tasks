
package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.util.Identifiable
import java.util.regex.Pattern


/**
 *  import com.johnsnowlabs.nlp.DocumentAssembler
 *  import com.johnsnowlabs.nlp.annotators.{PunctuationRemover, Tokenizer}
 *  import org.apache.spark.ml.Pipeline
 *  import org.apache.spark.sql.SparkSession
 *
 *
 *  val documentAssembler = new DocumentAssembler()
 *  .setInputCol("text")
 *  .setOutputCol("document")
 *
 *  val tokenizer = new Tokenizer()
 *  .setInputCols("document")
 *  .setOutputCol("token")
 *
 *
 *  val punctuationRemover = new PunctuationRemover()
 *  .setInputCols("token")
 *  .setOutputCol("punctuation_removed")
 *
 *
 *  val pipeline = new Pipeline().setStages(Array(
 *  documentAssembler,
 *  tokenizer,
 *  punctuationRemover
 *  ))
 *
 *  import spark.implicits._
 *
 * val data = Seq("\"The quick brown fox jumps over the lazy dog, but the dog doesn't seem to mind; it simply yawns, stretches, and goes back to sleep!\"")
 *  .toDF("text")
 * val result = pipeline.fit(data).transform(data)
 *
 *  result.selectExpr("punctuation_removed.result").show(truncate = false)
 *
 * }
 *
 * +-------------------------------------------------------------------------------------------------------------------------------------------------------+
 * |result                                                                                                                                                 |
 * +-------------------------------------------------------------------------------------------------------------------------------------------------------+
 * |[The, quick, brown, fox, jumps, over, the, lazy, dog, but, the, dog, doesn't, seem, to, mind, it, simply, yawns, stretches, and, goes, back, to, sleep]|
 * +-------------------------------------------------------------------------------------------------------------------------------------------------------+
 *
 *
 */


class PunctuationRemover(override val uid: String)
  extends AnnotatorModel[PunctuationRemover]
    with HasSimpleAnnotate[PunctuationRemover] {

  /**
   * Output annotator type: TOKEN
   */
  override val outputAnnotatorType: AnnotatorType = TOKEN

  /**
   * Input annotator type: TOKEN
   */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("PUNCTUATION_REMOVER"))




  /**
   * Tokens to be filtered out
   */

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val regex = " \\p{Punct}"
    val pattern = Pattern.compile(regex)

    val annotationsPunctuationRemover = annotations.filter(s => !pattern.matcher(s.toString()).find())



    annotationsPunctuationRemover.map { tokenAnnotation =>
      Annotation(
        outputAnnotatorType,
        tokenAnnotation.begin,
        tokenAnnotation.end,
        tokenAnnotation.result,
        tokenAnnotation.metadata)
    }
  }


}

 trait ReadablePunctuationRemoverModel
  extends ParamsAndFeaturesReadable[PunctuationRemover]
    with HasPretrained[PunctuationRemover] {

  /**
   * Java compliant-overrides
   * */
  override def pretrained(): PunctuationRemover = super.pretrained()
  override def pretrained(name: String): PunctuationRemover = super.pretrained(name)
  override def pretrained(name: String, lang: String): PunctuationRemover =
    super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): PunctuationRemover =
    super.pretrained(name, lang, remoteLoc)
}


/**
 *  This is the companion object of [[PunctuationRemover]]
 */
object PunctuationRemover
  extends ParamsAndFeaturesReadable[PunctuationRemover]


