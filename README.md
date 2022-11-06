# Text Classification Using Transfer Learning

## Scope
Apply transfer learning on DistilBert language model to classify text

## Example Data

| Sentence    | Sentence Label    | Word Count |
| :------------- | :-------------: | :-------------: |
| It is this hat that it is certain that he wa... | 1 | 12
| Her efficient looking up of the answer pleas... | 1 | 10
| You said she liked yourself | 0 | 5
| a pencil with that to write broke. | 0 | 7
| John enjoyed drawing trees for his syntax ho... | 1 | 8
| It was in the park last night that the polic... | 0 | 14

## Model Summary

| Layer (Type)  | Output Shape  | Number of Parameters | Connected To  |
| :------------- | :------------- | :-------------: | :------------- |
| inputs (InputLayer) | [(None, 128)] | 0 | [ ]                                  
| masks (InputLayer) | [(None, 128)] | 0 | [ ]                         
| tf_distil_bert_model (TFDistilBertModel) | TFBaseModelOutput (last_hidden_state=(None, 128, 768), hidden_states=None, attentions=None) | 66362880 | ['inputs[0][0]', 'masks[0][0]']
| tf.\_\_operators__.getitem (SlicingOpLambda) | (None, 768) | 0 | ['tf_distil_bert_model[0][0]']
| batch_normalization (BatchNormalization) | (None, 768) | 3072 | ['tf.\_\_operators__.getitem[0][0]']
| dense (Dense) | (None, 256) | 196864 | ['batch_normalization[0][0]']
| dropout_19 (Dropout) | (None, 256) | 0 | ['dense[0][0]']
| dense_1 (Dense) | (None, 128) | 32896 | ['dropout_19[0][0]']
| dropout_20 (Dropout) | (None, 128) | 0 | ['dense_1[0][0]']
| dense_2 (Dense) | (None, 1) | 129 | ['dropout_20[0][0]']

===========================================================================
**Total Parameters**: 66,595,841\
**Trainable Parameters**: 231,425\
**Non-trainable Parameters**: 66,364,416

## Training Results
Due to the class imbalance in the dataset (approximately 2.5 times more sentences labeled as 1 versus 0), precision-recall AUC is the model evaluation metric of choice.

| Epoch | Training Loss | Training PR AUC | Validation Loss | Validation PR AUC |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| 1 | 0.6205 | 0.7725 | 0.6159 | 0.8104 |
| 2 | 0.5457 | 0.8435 | 0.6141 | 0.8158 |
| 3 | 0.5243 | 0.8665 | 0.5907 | 0.8104 |
| 4 | 0.5055 | 0.8803 | 0.5720 | 0.8237 |
| 5 | 0.4864 | 0.8921 | 0.5560 | 0.8283 |
