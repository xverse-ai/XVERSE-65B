<div align="center">
<h1>
  XVERSE-65B
</h1>
</div>

<p align="center">
        <a href="https://huggingface.co/xverse/XVERSE-65B">🤗 XVERSE-65B</a>&nbsp｜
        <a href="https://modelscope.cn/organization/xverse" rel="nofollow"><img src="resources/modelscope.png" width="20px" style="max-width: 100%;"> ModelScope</a>&nbsp｜&nbsp
        <a href="resources/wechat.png">💬 WeChat</a>
</p>

<h4 align="left">
    <p>
        <a href="README.md">中文</a> |
        <b>English</b>
    <p>
</h4>

## Update Information
**[2023/11/24]** Update the related information of the pre-training data.  
**[2023/11/06]** Released the XVERSE-65B base model.  

## Model Introduction

**XVERSE-65B** is a multilingual large language model, independently developed by Shenzhen Yuanxiang Technology. The models released this time is the base model **XVERSE-65B**. Its key features are as follows:

- **Model Structure**: XVERSE-65B uses the mainstream Decoder-only Transformer network structure, supports 16k context length, which can meet the need of longer multi-round dialogues, knowledge question-answering, and summarization. This makes the model more versatile in application scenarios.
- **Training Data**: The model has been thoroughly trained on a diversified and high-quality dataset consisting of 2.6 trillion of tokens, including more than 40 languages such as Chinese, English, Russian, and Spanish. The sampling ratio of different types of data is finely set, which makes the performance of Chinese and English excellent, and also takes into account the effect of other languages.
- **Tokenization**: Based on the BPE (Byte-Pair Encoding) algorithm, a tokenizer with a vocabulary size of 100,534 has been trained using hundreds of gigabytes of language data. This tokenizer is capable of supporting multilingual without the need for additional vocabulary expansion.
- **Training Framework**: The training utilizes FlashAttention2 for accelerated computation, and on top of 3D parallelism, virtual pipeline technology is applied to reduce the excessive bubble rate caused by longer pipelines and 16k context windows. This achieves a peak computational efficiency within the industry-leading range in the petaflop-scale cluster. Concurrently, through continuous optimization of cluster infrastructure operations, resource scheduling, training frameworks, and the scheduling platform, a highly stable, low-interruption, and robust fault-tolerant training system has been developed, enhancing the effective weekly training rate to 98.6%.

During the pre-training phase, **XVERSE-65B** primarily utilized 7 different types of data. The following table shows a comparison of the pre-training datasets of XVERSE-65B with some other well-known models:

| Data Type       | GPT3[^1] | Llama[^2] | BLOOM[^3] | PaLM[^4] | Chinchilla[^5] | Gopher[^6] | MT-NLG[^7] | XVERSE-65B |
|:---------------:|:--------:|:---------:|:---------:|:--------:|:--------------:|:----------:|:----------:|:----------:|
| Web Pages       | Y        | Y         | Y         | Y        | Y              | Y          | Y          | Y          |
| Code            |          | Y         | Y         | Y        | Y              | Y          | Y          | Y          |
| Encyclopedia    | Y        | Y         |           | Y        | Y              | Y          | Y          | Y          |
| Books           | Y        | Y         |           | Y        | Y              | Y          | Y          | Y          |
| Academic Papers |          | Y         |           |          |                |            | Y          | Y          |
| QA             | Y        | Y         |           | Y        |                |            | Y          | Y          |

> Note: 'Y' indicates that the data type was used.

The sampling ratios of different data types during the pre-training phase are as follows:
|                | Web Pages | Code | Encyclopedia | Books | Academic Papers |  QA | Other |
|:--------------:|:---------:|:----:|:------------:|:-----:|:---------------:|:----:|:-----:|
| Proportion (%) |   72.91   | 7.09 |     4.81     |  5.62 |       6.55      | 1.15 |  1.87 |

[^1]: GPT3 Paper: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
[^2]: LLaMA Paper: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
[^3]: BLOOM Paper: [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100)
[^4]: PaLM Paper: [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
[^5]: Chinchilla Paper: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
[^6]: Gopher Paper: [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446)
[^7]: MT-NLG Paper: [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](https://arxiv.org/abs/2201.11990)

## Model Evaluation

To comprehensively assess the performance of the model, we conducted extensive testing across a range of standard datasets, including C-Eval, CMMLU, Gaokao-Bench, MMLU, GAOKAO-English, AGIEval, RACE-M, CommonSenseQA, PIQA, GSM8K and HumanEval. These evaluations spanned multiple capabilities of the model, specifically including Chinese question answering, English question answering, language comprehension, common sense questioning, logical reasoning, mathematical problem-solving, and coding ability. The results of the evaluations are as follows:

|  Capability Dimension  |          Dataset           |        | XVERSE-65B | Llama1-65B | Llama2-70B | Falcon-180B | GPT-3.5 | GPT-4 |
| :--------------------: | :------------------------: | :----: | :--------: | :--------: | :--------: | :---------: | :-----: | :---: |
|       Chinese QA       |           C-Eval           | 5-shot |    68.6    |    38.8    |    49.9    |    54.2     |  54.4   | 68.7  |
|                        |           CMMLU            | 5-shot |    72.6    |    40.6    |    53.6    |    57.2     |  53.9   | 71.0  |
|                        |  Gaokao-Bench<sup>1</sup>  | 5-shot |    73.9    |    38.9    |    51.4    |    50.5     |    -    |   -   |
|       English QA       |            MMLU            | 5-shot |    70.8    |    63.4    |    68.9    |    70.5     |  70.0   | 86.4  |
|                        | GAOKAO-English<sup>1</sup> | 5-shot |    85.3    |    67.0    |    76.6    |    63.3     |    -    |   -   |
|  Chinese & English QA  |    AGIEval<sup>1</sup>     | 5-shot |    61.8    |    42.4    |    51.4    |    51.3     |    -    |   -   |
| Language Understanding |           RACE-M           | 0-shot |    90.6    |    67.9    |    81.5    |    87.6     |  85.6   | 93.7  |
|    Common Sense QA     |       CommonSenseQA        | 7-shot |    79.8    |    74.0    |    78.5    |    82.4     |  80.2   | 88.3  |
|       Reasoning        |            PIQA            | 0-shot |    80.4    |    82.8    |    82.8    |    85.3     |  81.7   | 89.2  |
|          Math          |           GSM8K            | 4-shot |    60.3    |    50.9    |    56.8    |    62.6     |  57.1   | 92.0  |
|         Coding         |         HumanEval          | 0-shot |    26.8    |    23.7    |    29.9    |      -      |  48.1   | 67.0  |

> <sup>1: Tests are conducted only on single-answer multiple-choice questions, thus excluding fill-in-the-blanks, open-ended questions, and multiple-answer multiple-choice questions.</sup>   

For all the comparison models mentioned above, we prioritize the disclosure of their officially published results. In the absence of official data, we refer to the reported outcomes from [OpenCompass Leaderboard](https://opencompass.org.cn/leaderboard-llm). Results not covered by the aforementioned sources are derived from our own evaluation pipline.   
For MMLU, we adopt the [evaluation tools](https://github.com/hendrycks/test) provided by the authors, C-Eval, AGIEval, GAOKAO-Bench, GAOKAO-English are the same as MMLU. For the remaining evaluation datasets, the [OpenCompass](https://github.com/open-compass/OpenCompass/) is employed for evaluation.

## Usage

### Hardware requirements
The following table lists the hardware resources required for inference and fine-tuning on XVERSE-65B:
|            | Type      | Kind             | Memory | GPU        |
| ---------- | --------- | ---------------- | ------ | ---------- |
| XVERSE-65B | Training  | LoRA with ZeRO-3 | 1500GB | 8*A800 80G |
| XVERSE-65B | Inference | BF16/FP16        | 500GB  | 2*A800 80G |

### Environment Setup

1. Clone this repository:

```shell
git clone https://github.com/xverse-ai/XVERSE-65B
cd XVERSE-65B
```

2. Install the dependencies using pip:

```shell
pip install -r requirements.txt
```

### Loading with Transformers

The XVERSE-65B model can be loaded for inference using the following code:

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> tokenizer = AutoTokenizer.from_pretrained("xverse/XVERSE-65B")
>>> model = AutoModelForCausalLM.from_pretrained("xverse/XVERSE-65B", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
>>> model = model.eval()
>>> inputs = tokenizer('北京的景点：故宫、天坛、万里长城等。\n深圳的景点：', return_tensors='pt').input_ids
>>> inputs = inputs.cuda()
>>> generated_ids = model.generate(inputs, max_new_tokens=64, eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.1)
>>> print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
```

### Web Demo

The following code can be used to start a web server. By entering the access address in the browser, you can perform inference with the XVERSE-65B model:

```shell
python chat_demo.py --port='port' --model_path='/path/to/model/' --tokenizer_path='/path/to/tokenizer/'
```

### Fine-tuning
XVERSE-65B allow developers to fine-tune for improved performance. Here, we attempted to use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for compatible fine-tuning training with XVERSE-65B, and tested it in an environment with 8 * Nvidia A800 80 GB + DeepSpeed.
Below, we provide the fine-tuning method using `LoRA with ZeRO-3`.


#### Environment Setup

Download the LLaMA-Factory project and [install dependencies] (https://github.com/hiyouga/LLaMA-Factory#getting-started) as required.

#### Training

Training launch script:
> Replace model_path with your own model path.

> Both XVERSE-65B and XVERSE-65B-Chat are trained based on bfloat16. It is recommended to use bfloat16 for fine-tuning training.
```bash
deepspeed --num_gpus 8 src/train_bash.py \
    --deepspeed deepspeed.json \
    --stage sft \
    --model_name_or_path model_path  \
    --do_train \
    --dataset alpaca_gpt4_zh \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir  output_model_path \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16
```
deep_speed.json parameter settings：
```json
{
    "train_micro_batch_size_per_gpu":"auto",
    "gradient_accumulation_steps":"auto",
    "gradient_clipping":"auto",
    "zero_allow_untested_optimizer":true,
    "fp16":{
        "enabled":false
    },
    "bfloat16":{
        "enabled":true
    },
    "zero_optimization":{
        "stage":3,
        "allgather_partitions":true,
        "reduce_scatter":true,
        "overlap_comm":false,
        "contiguous_gradients":true
    }
}

```

## Limitations and Disclaimer

Like all other Large Language Models (LLMs), XVERSE-65B may produce inaccurate, biased, or otherwise offensive content under certain circumstances. Therefore, please use the content generated by the model with caution and refrain from disseminating harmful content. Before deploying any application of XVERSE-65B, developers should conduct safety tests and optimization of the model according to its specific application.

We strongly warn against the use of the XVERSE-65B model for producing or spreading harmful information, or conducting any activities that might harm the public, national, or social security, or violate regulations. We assume no responsibility for any problems arising from the use of the XVERSE-65B model, whether it be data security issues, public opinion risks, or any risks and issues caused by misunderstanding, misuse, dissemination, or non-compliance with the model.

## Open Source License

The use of the source code in this repository must follow the [Apache-2.0](LICENSE) open-source license, while the use of the model weights of XVERSE-65B needs to adhere to the [Model License Agreement](MODEL_LICENSE.pdf).

The XVERSE-65B model weights are **fully open** to academic research and support **free commercial use**.  To apply for a commercial license, please fill in the [application form](https://chat.xverse.cn/home/business.html). For other questions or collaborations, please contact <opensource@xverse.cn>.

