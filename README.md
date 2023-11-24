<div align="center">
<h1>
  XVERSE-65B
</h1>
</div>

<p align="center">
        <a href="https://huggingface.co/xverse/XVERSE-65B">🤗 XVERSE-65B</a>&nbsp｜
        <a href="https://modelscope.cn/organization/xverse" rel="nofollow"><img src="resources/modelscope.png" width="20px" style="max-width: 100%;"> ModelScope</a>&nbsp｜&nbsp
        <a href="resources/wechat.png">💬 微信社区</a>
</p>

<h4 align="left">
    <p>
        <b>中文</b> |
        <a href="README_EN.md">English</a>
    <p>
</h4>

## 更新信息
**[2023/11/24]** 更新预训练数据的相关信息。  
**[2023/11/06]** 发布 65B 尺寸的 XVERSE-65B 底座模型。  

## 模型介绍

**XVERSE-65B** 是由深圳元象科技自主研发的支持多语言的大语言模型（Large Language Model），参数规模为 650 亿，本次开源的模型为底座模型 **XVERSE-65B**，主要特点如下：

- **模型结构**：XVERSE-65B 使用主流 Decoder-only 的标准 Transformer 网络结构，支持 16K 的上下文长度（Context Length），能满足更长的多轮对话、知识问答与摘要等需求，模型应用场景更广泛。
- **训练数据**：构建了 2.6 万亿 token 的高质量、多样化的数据对模型进行充分训练，包含中、英、俄、西等 40 多种语言，通过精细化设置不同类型数据的采样比例，使得中英两种语言表现优异，也能兼顾其他语言效果。
- **分词**：基于 BPE（Byte-Pair Encoding）算法，使用上百 GB 语料训练了一个词表大小为 100,534 的分词器，能够同时支持多语言，而无需额外扩展词表。
- **训练框架**：训练中采用 FlashAttention2 加速计算，3D 并行基础上采用虚拟流水线（virtual pipeline）技术，降低较长流水线和 16k 上下文窗口产生的过高气泡率，在千卡集群的峰值算力利用率达到业界前列。同时通过集群基础设施运营、资源调度、训练框架和调度平台协同等持续优化，打造出高稳定、低中断、强容错的训练系统，将每周有效训练率提升至 98.6%。

在预训练阶段，**XVERSE-65B** 主要使用了 7 类不同的数据类型。以下表格展示了 XVERSE-65B 与其他一些知名模型在预训练数据集方面的比较：

| 数据类别 | GPT3[^1] | Llama[^2] | BLOOM[^3] | PaLM[^4] | Chinchilla[^5] | Gopher[^6] | MT-NLG[^7] | XVERSE-65B |
|:-------:|:--------:|:---------:|:---------:|:--------:|:--------------:|:----------:|:----------:|:----------:|
| 网页类   | Y        | Y         | Y         | Y        | Y              | Y          | Y          | Y          |
| 代码类   |          | Y         | Y         | Y        | Y              | Y          | Y          | Y          |
| 百科类   | Y        | Y         |           | Y        | Y              | Y          | Y          | Y          |
| 书籍类   | Y        | Y         |           | Y        | Y              | Y          | Y          | Y          |
| 论文类   |          | Y         |           |          |                |            | Y          | Y          |
| 问答类   | Y        | Y         |           | Y        |                |            | Y          | Y          |

> 注：'Y' 表示使用了该类数据。

在预训练阶段，不同类别数据的采样比例如下所示：
|         | 网页类 | 代码类 | 百科类 | 书籍类 | 论文类 | 问答类 | 其他类 |
|:-------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 比例(%) | 72.91  | 7.09   | 4.81   | 5.62   | 6.55   | 1.15   | 1.87   |

[^1]: GPT3 Paper: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
[^2]: LLaMA Paper: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
[^3]: BLOOM Paper: [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100)
[^4]: PaLM Paper: [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
[^5]: Chinchilla Paper: [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556)
[^6]: Gopher Paper: [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446)
[^7]: MT-NLG Paper: [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](https://arxiv.org/abs/2201.11990)

## 评测结果

为了综合评估模型的性能，我们在一系列标准数据集上进行了全面测试，包括C-Eval、CMMLU、Gaokao-Bench、MMLU、GAOKAO-English、AGIEval、RACE-M、CommonSenseQA、PIQA、GSM8K和HumanEval。这些评估覆盖了模型在多个领域的能力，具体包括中文问答、英文问答、语言理解、常识问答、逻辑推理、数学问题解答以及编程能力。评估结果如下：

|  能力维度  |           数据集           |        | XVERSE-65B | Llama1-65B | Llama2-70B | Falcon-180B | GPT-3.5 | GPT-4 |
| :--------: | :------------------------: | :----: | :--------: | :--------: | :--------: | :---------: | :-----: | :---: |
|  中文问答  |           C-Eval           | 5-shot |    68.6    |    38.8    |    49.9    |    54.2     |  54.4   | 68.7  |
|            |           CMMLU            | 5-shot |    72.6    |    40.6    |    53.6    |    57.2     |  53.9   | 71.0  |
|            |  Gaokao-Bench<sup>1</sup>  | 5-shot |    73.9    |    38.9    |    51.4    |    50.5     |    -    |   -   |
|  英文问答  |            MMLU            | 5-shot |    70.8    |    63.4    |    68.9    |    70.5     |  70.0   | 86.4  |
|            | GAOKAO-English<sup>1</sup> | 5-shot |    85.3    |    67.0    |    76.6    |    63.3     |    -    |   -   |
| 中英文问答 |    AGIEval<sup>1</sup>     | 5-shot |    61.8    |    42.4    |    51.4    |    51.3     |    -    |   -   |
|  语言理解  |           RACE-M           | 0-shot |    90.6    |    67.9    |    81.5    |    87.6     |  85.6   | 93.7  |
|  常识问答  |       CommonSenseQA        | 7-shot |    79.8    |    74.0    |    78.5    |    82.4     |  80.2   | 88.3  |
|    推理    |            PIQA            | 0-shot |    80.4    |    82.8    |    82.8    |    85.3     |  81.7   | 89.2  |
|    数学    |           GSM8K            | 4-shot |    60.3    |    50.9    |    56.8    |    62.6     |  57.1   | 92.0  |
|    代码    |         HumanEval          | 0-shot |    26.8    |    23.7    |    29.9    |      -      |  48.1   | 67.0  |

> <sup>1：只针对其中的单项选择题进行测试，即排除了填空题、开放性问题和多项选择题</sup>   

对于上述所有比较模型，我们优先汇报其官方公布的结果。在缺少官方结果的情况下，我们采用了 [OpenCompass 榜单](https://opencompass.org.cn/leaderboard-llm)的报告结果。其他结果则来自于我们自行执行的评估流程所获得的数据。   
对于 MMLU ，我们采用作者提供的[评测工具](https://github.com/hendrycks/test)，C-Eval、AGIEval、GAOKAO-Bench、GAOKAO-English 与 MMLU 的评测方式相同，其余评测数据集使用 [OpenCompass 评估框架](https://github.com/open-compass/OpenCompass/)进行评估。

## 使用方法

### 硬件需求
下表列出了在 XVERSE-65B 上进行推理和微调所需要的硬件资源：
|            | 类型 | 方法             | 内存   | GPU        |
| ---------- | ---- | ---------------- | ------ | ---------- |
| XVERSE-65B | 训练 | LoRA with ZeRO-3 | 1500GB | 8*A800 80G |
| XVERSE-65B | 推理 | BF16/FP16        | 500GB  | 2*A800 80G |

### 环境安装

1. 下载本仓库：

```shell
git clone https://github.com/xverse-ai/XVERSE-65B
cd XVERSE-65B
```

2. 使用 pip 安装依赖：

```shell
pip install -r requirements.txt
```
### Transformers 加载方式

可通过以下代码加载 XVERSE-65B 模型来进行推理：

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

### 网页 Demo

可通过以下代码启动一个web server，在浏览器输入访问地址后，可使用 XVERSE-65B 模型进行推理：

```shell
python text_generation_demo.py --port='port' --model_path='/path/to/model/' --tokenizer_path='/path/to/tokenizer/'
```

### 模型微调

XVERSE-65B 支持开发者进行微调以实现更好的性能表现。在此我们尝试使用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 与 XVERSE-65B 进行兼容性微调训练，并在 8 * Nvidia A800 80 GB + DeepSpeed 的环境下进行了测试。
下面我们给出了使用`LoRA with ZeRO-3`的微调方法。

#### 环境准备

下载 LLaMA-Factory 项目并按其要求[安装依赖](https://github.com/hiyouga/LLaMA-Factory#getting-started)。

#### 启动训练

训练启动脚本：
> 其中 model_path 请替换为自己的模型路径

> XVERSE-65B 基于 bfloat16 训练的，建议选用 bfloat16 做微调训练。
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
deep_speed.json 参数配置：
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

## 局限性与免责申明

XVERSE-65B 与其他所有 LLM 一样，在某些情况下可能会产生不准确、有偏见或其他令人反感的内容。因此，请谨慎使用模型生成的内容，请勿将生成的有害内容进行传播，在部署任何 XVERSE-65B 的应用之前，开发人员应根据其具体应用对模型进行安全测试和调优。

我们强烈警告不要将 XVERSE-65B 模型用于制造或传播有害信息，或进行任何可能损害公众、国家、社会安全或违反法规的活动。如果使用 XVERSE-65B 模型产生任何问题，无论是数据安全问题、公共舆论风险，还是模型被误解、滥用、传播或不合规使用所引发的任何风险和问题，我们将不承担任何责任。

## 模型开源协议

使用本仓库的源码需要遵循 [Apache-2.0](LICENSE) 开源协议，使用 XVERSE-65B 的模型权重则需要遵循[模型许可协议](MODEL_LICENSE.pdf)。

XVERSE-65B 模型权重对学术研究**完全开放**，并且支持**免费商用**。如需申请商业许可证，请填写【[申请表](https://chat.xverse.cn/home/business.html)】，如有其他问题或合作，请联系 <opensource@xverse.cn>。

