import breeze.storage.ConfigurableDefault.fromV
import com.johnsnowlabs.nlp.AnnotatorType.{TOKEN, WORD_EMBEDDINGS}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType, HasInputAnnotationCols, HasPretrained, HasSimpleAnnotate, ParamsAndFeaturesReadable}
import org.apache.spark.ml.util.Identifiable

class OOVRatioAnnotator(override val uid: String)
    extends AnnotatorModel[OOVRatioAnnotator]
      with HasSimpleAnnotate[OOVRatioAnnotator]
      {

    def this() = this(Identifiable.randomUID("OOV_RATIO_ANNOTATOR"))


    // Implement the inputAnnotatorTypes member from the HasInputAnnotationCols trait
    override val outputAnnotatorType: AnnotatorType = WORD_EMBEDDINGS

    /**
     * Input annotator type : SENTENCE_EMBEDDINGS
     */
    override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.WORD_EMBEDDINGS)

    /**
     * Names of input annotation cols containing embeddings
     */
    override def setInputCols(value: Array[String]): this.type = set(inputCols, value)


    def setInputCol(value: Array[String]): this.type = set(inputCols, value)



    /**
     * Takes average of  two sentence embeddings
     */

    override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

        val count = annotations.size
        val trueValues = annotations.count(annotation => annotation.metadata("isOOV").toBoolean)
         annotations.map{

            annotation =>

              annotation

                annotation.copy(metadata = annotation.metadata+("oovRatio"-> ((trueValues/count)*100).toString) )
        }



    }


}



/**
 * This is the companion object of [[OOVRatioAnnotator]]
 */
object OOVRatioAnnotator
  extends ParamsAndFeaturesReadable[OOVRatioAnnotator]