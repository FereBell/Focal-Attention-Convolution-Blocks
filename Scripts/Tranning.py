from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy, BinaryIoU, Recall, Precision, AUC, TrueNegatives, TruePositives, FalseNegatives, FalsePositives
import tensorflow as tf

class Entre_editado(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        self.loss_fn= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= False)
        self.Ent_loss= Mean(name= 'Loss')
        self.Ent_exactitud= SparseCategoricalAccuracy(name= 'Entxactitud')
        self.Ent_AUC= AUC(name= 'AUC')
        self.Ent_TN= TrueNegatives(name= 'TN')
        self.Ent_FP= FalsePositives(name= 'FP')
        self.Ent_Precision= Precision(name= "Presicion")
        self.Ent_Recall= Recall(name= "Recall")
        self.Ent_IoU= BinaryIoU(name= "IoU")
        self.Val_loss= Mean(name= 'Loss')
        self.Val_exactitud= SparseCategoricalAccuracy(name= 'Entxactitud')
        self.Val_AUC= AUC(name= 'AUC')
        self.Val_TN= TrueNegatives(name= 'TN')
        self.Val_FP= FalsePositives(name= 'FP')
        self.Val_Precision= Precision(name= "Presicion")
        self.Val_Recall= Recall(name= "Recall")
        self.Val_IoU= BinaryIoU(name= "IoU")

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.loss_fn(y, y_pred)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        mask = tf.math.argmax(y_pred, axis=-1)
        mask = tf.expand_dims(mask, axis=-1)
        self.Ent_loss(loss)
        self.Ent_exactitud(y, y_pred)
        self.Ent_AUC(y, mask)
        self.Ent_TN(y, mask)
        self.Ent_FP(y, mask)
        self.Ent_Precision(y, mask)
        self.Ent_Recall(y, mask)
        self.Ent_IoU(y, mask)
        Specificity= self.Ent_TN.result()/(self.Ent_TN.result()+ self.Ent_FP.result())
        F_1= 2*((self.Ent_Precision.result()* self.Ent_Recall.result())/(self.Ent_Precision.result()+ self.Ent_Recall.result()))

        return {"loss": self.Ent_loss.result(), "exac": self.Ent_exactitud.result(), "AUC": self.Ent_AUC.result(), "pre": self.Ent_Precision.result(), "rec": self.Ent_Recall.result(),
                "sp": Specificity, "F1": F_1, "IoU": self.Ent_IoU.result()}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss= self.loss_fn(y, y_pred)
        mask = tf.math.argmax(y_pred, axis=-1)
        mask = tf.expand_dims(mask, axis=-1)

        self.Val_loss(loss)
        self.Val_exactitud(y, y_pred)
        self.Val_AUC(y, mask)
        self.Val_TN(y, mask)
        self.Val_FP(y, mask)
        self.Val_Precision(y, mask)
        self.Val_Recall(y, mask)
        self.Val_IoU(y, mask)
        Specificity= self.Val_TN.result()/(self.Val_TN.result()+ self.Val_FP.result())
        F_1= 2*((self.Val_Precision.result()* self.Val_Recall.result())/(self.Val_Precision.result()+ self.Val_Recall.result()))

        return {"loss": self.Val_loss.result(), "exac": self.Val_exactitud.result(), "AUC": self.Val_AUC.result(), "pre": self.Val_Precision.result(), "rec": self.Val_Recall.result(),
                "sp": Specificity, "F1": F_1, "IoU": self.Val_IoU.result()}

    @property
    def metrics(self):
        return [self.Ent_loss, self.Ent_exactitud, self.Ent_Precision, self.Ent_Recall, self.Ent_TN, self.Ent_FP, self.Ent_AUC, self.Ent_IoU,
                self.Val_loss, self.Val_exactitud, self.Val_Precision, self.Val_Recall, self.Val_TN, self.Val_FP, self.Val_AUC, self.Val_IoU]