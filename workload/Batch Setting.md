## Batch Test Setting

|     Task      |   Model_Dataset    | Server | Repeat_num | max_batchsize | Length_per |
| :-----------: | :----------------: | :----: | :--------: | :-----------: | :--------: |
|    Chatbot    | blenderbot_convAI  | node-1 |     20     |      128      |    P80     |
|    Chatbot    | blenderbot_cornell | node-1 |     20     |      128      |    P80     |
|    Chatbot    |     gpt_convAI     | node-1 |     20     |      128      |    P80     |
|    Chatbot    |    gpt_cornell     | node-1 |     20     |      128      |    P80     |
| Summarization |       t5_cnn       | node-1 |     10     |      100      |    P50     |
| Summarization |      bart_cnn      | node-1 |     10     |      100      |    P50     |
|  Translation  |      fsmt_wmt      |  nfs   |     10     |      100      |    P50     |
|  Translation  |     mbart_wmt      |  nfs   |     10     |      100      |    P50     |
| Static_Classification| resnet_imagenet| new_node-1|20|128|/|
