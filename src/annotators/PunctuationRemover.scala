
package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._

import org.apache.spark.ml.param.{BooleanParam, Param, ParamValidators, StringArrayParam}
import org.apache.spark.ml.util.Identifiable




class PunctuationRemover(override val uid: String)
  extends AnnotatorModel[PunctuationRemover]
    with HasSimpleAnnotate[PunctuationRemover] {

  /** Output annotator type: TOKEN
   *
   * @group anno
   */
  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** Input annotator type: TOKEN
   *
   * @group anno
   */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("PUNCTUATION_REMOVER"))

  /** The words to be filtered out (Default: Stop words from MLlib)
   *
   * @group param
   */
  val PunctuationRemover: StringArrayParam = new StringArrayParam(
    this,
    "PunctuationRemover",
    "------")

  /** The words to be filtered out
   *
   * @group setParam
   */
  def setPunctuationRemover(value: Array[String]): this.type = set(punctuationRemover, value)

  /** The words to be filtered out
   *
   * @group getParam
   */
  def getPunctuationRemover: Array[String] = $(punctuationRemover)

  /** Whether to do a case-sensitive comparison over the stop words (Default: `false`)
   *
   * @group param
   */


  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val annotationsPunctuationRemover =  annotations.replaceAll("""[\p{Punct}]""", "")
  }) )
    .getOrElse (word)

    }

}

}

/** trait ReadablePunctuationRemoverModel
  extends ParamsAndFeaturesReadable[PunctuationRemover]
    with HasPretrained[PunctuationRemover] {
  override val defaultModelName: Some[String] = Some("stopwords_en")

  /** Java compliant-overrides */
  override def pretrained(): PunctuationRemover = super.pretrained()
  override def pretrained(name: String): PunctuationRemover = super.pretrained(name)
  override def pretrained(name: String, lang: String): PunctuationRemover =
    super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): PunctuationRemover =
    super.pretrained(name, lang, remoteLoc)
}
 */
object PunctuationRemover
  extends ParamsAndFeaturesReadable[PunctuationRemover]
    with ReadablePretrainedPunctuationRemoverModel
