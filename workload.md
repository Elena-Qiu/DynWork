## Workload

### Chatbot

#### Model

- blenderbot
  - facebook/blenderbot_small-90M
  - Source: https://huggingface.co/facebook/blenderbot_small-90M
  - Github: https://github.com/facebookresearch/ParlAI
  - Citation: [*Recipes for building an open-domain chatbot*](https://arxiv.org/abs/2004.13637)
- gpt
  - microsoft/DialoGPT-small
  - Source: https://huggingface.co/microsoft/DialoGPT-small
  - Github: https://github.com/microsoft/DialoGPT
  - Citation: [*DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation*](https://arxiv.org/abs/1911.00536)

#### Dataset

- cornell
  - Cornell Movie-Dialogs Corpus
  - Source: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
  - Citation: [*Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs*](https://arxiv.org/abs/1106.3077)
- convAI
  - ConvAI2 Dataset
  - Source: http://convai.io/data (data_tolokers.json)
  - Github: https://github.com/DeepPavlov/convai
  - Citation: [*The Second Conversational Intelligence Challenge (ConvAI2)*](https://arxiv.org/abs/1902.00098)

### Summarization

#### Model

- bart
  - facebook/bart-large-cnn
  - Source: https://huggingface.co/facebook/bart-large-cnn
  - Github: https://github.com/pytorch/fairseq/tree/master/examples/bart
  - Citation: [*BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*](https://arxiv.org/abs/1910.13461)
- t5
  - google/t5-base
  - Source: https://huggingface.co/t5-base
  - Github: [https](https://github.com/google-research/text-to-text-transfer-transformer)[://github.com/google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer)
  - Citation: [*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*](https://arxiv.org/abs/1910.10683)

#### Dataset

- cnn
  - CNN-DailyMail
  - Source: https://huggingface.co/datasets/cnn_dailymail
  - Github: https://github.com/abisee/cnn-dailymail
  - Citation: *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf)*

### Translation

#### Model

- mbart
  - facebook/mbart-large-50-many-to-many-mmt
  - Source: https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt
  - Github: https://github.com/pytorch/fairseq/tree/main/examples/mbart
  - Citation: [*Multilingual Translation with Extensible Multilingual Pretraining and Finetuning*](https://arxiv.org/abs/2008.00401)
- fsmt
  - facebook/wmt19-en-de
  - Source: https://huggingface.co/facebook/wmt19-en-de
  - Github: https://github.com/pytorch/fairseq/blob/main/examples/wmt19/README.md
  - Citation: [*Facebook FAIR's WMT19 News Translation Task Submission*](https://arxiv.org/abs/1907.06616)

#### Dataset

- wmt
  - Workshop on Statistical Machine Translation
  - Source: http://statmt.org/wmt18/translation-task.html#download (news-commentary-v13.en.gz)
  - Citation: [*Findings of the 2018 Conference on Machine Translation (WMT18)*](