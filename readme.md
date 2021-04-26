# Attention Based Question Answering
This project used [CoQa](https://github.com/stanfordnlp/coqa-baselines) datasets. This dataset is for conversational question answering, but I make this project for just simple question answering.

## Model Pipeline
This project consist of three module, Question Analysis, Passage Retriever, and Answer Finder

![](https://github.com/share424/attention-based-question-answeringblob/master/images/architechture.png?raw=true)

The pipeline will receive two input, the question and the context of the question (e.g. news, article, etc). The question will be feed to the Question analysis and the context will be splitted by sentence and feed to the passage retriever to get the proposed sentence that contains the answer. Finally the answer finder will be receive the question and the proposed sentence as input and generate predicted answer.

## Question Analysis
The point of the question analysis is to extract information from the question, you can use NER (Named entity recognition) to do this, but in this project I use [SBert](https://www.sbert.net/) to get the embedding of the question.

```python
# e.g.
question_encoder = SentenceTransformer('facebook-dpr-question_encoder-multiset-base')
question = "Where did she live?"
question_embedding = question_encoder.encode(question)
print(question_embedding)

# example output
# [ 4.06161875e-01 -1.38373017e-01 -1.14733957e-01  2.26605639e-01 ... ]
```

## Passage Retriever
The purpose of this module is to propose a sentence from the context texts that contains the answer of the question. This module will receive the embedding of the question and the embedding of every sentence in given context. And finally calculate the cosine similarity to get the proposed sentence. The proposed sentence will be used as input in the answer finder. To get better accuracy, I use top-3 sentence as the input, so we will get 3 best answer

```python
# e.g.
context_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-multiset-base')
context = "this is long article that contains the answer..."
# split the context by sentence
sentences = sent_tokenize(context)
# calculate the context embedding
context_embedding = context_encoder.encode(sentences)
# calculate similarities
similarities = util.pytorch_cos_sim(question_embedding, story_embedding).numpy()
# sort similarities
sorted_arg = np.argsort(similarities, axis=-1)[0][::-1]
# print the top-3 sentence
print(sorted_arg[:3])
```

## Answer Finder
This is the main module to generate the answer. This module will receive question and proposed sentence that may contains the answer.

![](https://github.com/share424/attention-based-question-answeringblob/master/images/answer-finder.png?raw=true)

this module consist of 3 step, Preprocessing, encoder, and decoder

### Preprocessing
This module will concat the question and the proposed sentence into one input text.

```python
# e.g.
question = "where did she live?"
proposed_sentence = "in a barn near a farm house, there lived a little white kitten"
# concat the question and proposed sentence with <sep> token
input_text = "<start> " + question + " <sep> " + proposed_sentence + " <end>"
print(input_text)
# output: <start> where did she live? <sep> in a barn near a farm house, there lived a little white kitten <end>
```

Every number that contains in the question or proposed sentence will be extracted

```python
text = "there are 5 dogs in the house"
output, numbers = preprocess_sentence(text)
print(output)
# output: <start> are <number> dogs in the house <end>
print(numbers)
# output: ['5']
```

this number will be feed to number decoder to get the correct number for the output

### Encoder and Decoder
This module is responsible to get the context of the question and generate the answer using attention

![](https://github.com/share424/attention-based-question-answeringblob/master/images/encoder-decoder.png?raw=true)

I use [BahdanauAttention](https://arxiv.org/abs/1409.0473) for the attention layer, you can check the paper for the detail implementation. Every generated `<number>` token will be passed to the number decoder to get the correct number. The number decoder will receive the hidden state and the extracted number from the preprocessing step and output the correct number.

Note: You can implement the number decoder mecanism for `NAME`, `LOCATION`, and `ENTITY` as well

## Example Output
![](https://github.com/share424/attention-based-question-answeringblob/master/images/result.png?raw=true)

## Results
The model get `74.16` for the `BLEU-4` score

## Pretrained Models
you can download the pretrained models [here](https://drive.google.com/drive/folders/14fuDTOEXh3P79qgN-ThCsl1F6fOyH065?usp=sharing)

## Reference
1. [Machine Translation](https://www.tensorflow.org/tutorials/text/nmt_with_attention)
2. [SBert](https://www.sbert.net/) for the sentence embedding
3. [CoQa](https://github.com/stanfordnlp/coqa-baselines)