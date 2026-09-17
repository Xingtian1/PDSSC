# PDSSC vs. SemantiCodec

本文档基于两部分材料进行对照：

- 我们当前稿件：[paper.tex](C:/Users/cheng/Desktop/SpeechTokenizer-main/IEEE-conference-template-062824/writing/paper.tex)
- 对比论文：`C:\Users\cheng\Desktop\SemantiCodec_An_Ultra_Low_Bitrate_Semantic_Audio_Codec_for_General_Sound.pdf`

说明：这里的比较主要围绕论文目标、技术路线和论文定位，不是简单比较谁“绝对更强”。二者有明显交集，但研究问题并不相同。

## 一、相似点

1. 都属于“低码率语义化离散音频/语音表示”路线。  
两篇工作都不再满足于传统波形保真式压缩，而是希望通过离散 token 表示，把更重要的语义信息优先保留下来。

2. 都强调“语义信息”和“声学细节”之间的分工。  
SemantiCodec 用 semantic encoder + acoustic encoder 的双分支思路，把语义与细节拆开；PDSSC 用语义蒸馏驱动的分层 RVQ，把底层语义骨架和高层细节逐步分离。

3. 都依赖预训练表征来增强语义性。  
SemantiCodec 的语义分支建立在 AudioMAE 特征与 k-means 语义聚类上；PDSSC 的语义优先分层建立在教师语义表征约束之上。

4. 都不是“仅靠量化器本身”来获得好结果。  
两者都把量化后的离散表示和一个更强的生成/恢复模块结合起来：SemantiCodec 用 latent diffusion decoder，PDSSC 用 receiver-side flow completion。

5. 都可以被解释为“先抓主干，再补细节”的思想。  
只是 SemantiCodec 是结构上先 semantic 后 acoustic，PDSSC 是传输顺序上先 semantic backbone 后 higher-layer detail。

## 二、不同点

### 1. 研究目标不同

- **SemantiCodec** 的核心目标是做一个面向 **general sound** 的 ultra-low-bitrate semantic audio codec。  
  它关注的是：在语音、环境声、音乐等广义音频上，以极低 token rate 压缩，同时保留更强的语义信息，便于后续 audio language modeling 和高质量重建。

- **PDSSC** 的核心目标是做一个面向 **packet-switched speech communication** 的极低码率语义通信系统。  
  我们关注的是：在真实通信约束下，同时解决极低码率、丢包、渐进传输、时延和部署效率问题。

一句话概括：  
SemantiCodec 更像“面向通用音频与语言建模的语义 codec”；PDSSC 更像“面向受限链路语音通信的语义传输系统”。

### 2. 任务定义不同

- SemantiCodec 主要讨论的是 **编码-重建质量** 与 **token semantic richness**。
- PDSSC 讨论的是 **编码-传输-丢包-恢复** 的全链路问题。

因此，PDSSC 比 SemantiCodec 多了几个 SemantiCodec 并未真正处理的通信层问题：

- 主动码率选择
- 被动分组丢失
- 渐进式传输
- 接收端缺失补全
- 端到端时延与实时部署

### 3. 表示组织方式不同

- **SemantiCodec**：双编码器、两层量化。  
  第一层语义 token 由 AudioMAE 特征经过 k-means 语义聚类得到；第二层 acoustic token 用可学习 VQ 补充细节。它的“语义/声学分工”主要是 **结构性双分支分离**。

- **PDSSC**：单编码器、分层 RVQ、语义优先排序。  
  我们不是把系统拆成 semantic branch 和 acoustic branch，而是在 **同一个 layered codec 内部**，通过语义蒸馏让低层先承载语义骨架，再由高层逐步补充说话人相关与残差信息。它的核心不是双分支，而是 **传输优先级驱动的分层组织**。

这点是一个本质差别。  
SemantiCodec 更强调“分成两类 token”；PDSSC 更强调“同一套分层 token 的优先级次序”。

### 4. 生成/恢复方式不同

- **SemantiCodec** 用 latent diffusion model 作为 decoder，目标是从 token 条件中重建高质量通用音频。
- **PDSSC** 用 low-step flow completion model，目标不是从零生成，而是对 **不完整但已有语义骨架的 latent** 做快速补全。

因此两者的推理问题也不同：

- SemantiCodec 偏向“条件生成式解码”
- PDSSC 偏向“缺失细节恢复式解码”

后者天然更贴近通信恢复场景，因为接收端已经拿到了部分有效表示，不必从噪声开始完整生成。

### 5. 侧信息的使用方式不同

- SemantiCodec 没有围绕“固定用户群/同说话人侧信息”构建恢复机制。
- PDSSC 明确利用了 **same-speaker side information**，并把它纳入统一恢复框架。

这意味着 PDSSC 的恢复模型不是无条件猜测缺失内容，而是借助说话人先验去补全更可能缺失的高层说话人相关细节。

### 6. 码率与应用点位不同

- SemantiCodec 主打的 token rate/bitrate 非常低，论文中给出 25 / 50 / 100 token/s，对应约 0.31 / 0.70 / 1.40 kbps，重点是“极低 token rate 下的通用音频 codec”。
- PDSSC 当前稿件主打的是语音通信工作点，重点讨论 1.0--4.0 kbps，尤其强调 1.5 / 3.0 kbps 下的传输质量、鲁棒性和时延。

从数值上看，SemantiCodec 的 nominal bitrate 更低；但它解决的问题不包含我们强调的 packet loss robustness、adaptive transmission 和 real-time completion。

### 7. 评估维度不同

- **SemantiCodec** 更看重：
  - reconstruction quality
  - MUSHRA
  - spectrogram distance
  - semantic richness / classification accuracy
  - 对 audio language modeling 的潜力

- **PDSSC** 更看重：
  - ViSQOL
  - UTMOS
  - PLCMOS
  - packet-loss robustness
  - ablation under packet loss
  - latency / FLOPs / deployment feasibility

这说明两篇文章的“论文价值锚点”不同。  
SemantiCodec 的锚点是“更适合语义音频 tokenization”；PDSSC 的锚点是“更适合极低码率鲁棒语音通信”。

## 三、我们的优势

下面这些点，是 PDSSC 相比 SemantiCodec 更容易成立、而且更适合在我们论文中强调的地方。

### 1. 我们解决的是更完整的通信问题，而不只是 codec 问题

SemantiCodec 的主问题仍然是“如何以更低 token rate 表示并重建音频”。  
PDSSC 则把问题扩展到了真实链路中的完整闭环：

- 编码
- 渐进发送
- 丢包后的不完整接收
- 接收端恢复
- 自适应码率控制
- 实时部署

如果你的论文定位是“speech communication system”，那么这一点是非常重要的优势。  
也就是说，我们不是只提出了一个更语义化的 codec，而是提出了一套能在不可靠链路上工作的完整传输框架。

### 2. 我们的“语义优先级”更适合渐进传输

SemantiCodec 的语义/声学分离是有效的，但它更像“两个功能块的拼接”：先 semantic token，再 acoustic token。  
PDSSC 的分层设计更进一步，强调的是：

- 哪些信息必须最先发
- 哪些信息可以后发
- 在码率被截断时，系统优先保住什么

这使得 PDSSC 天然支持 progressive transmission，而这正是通信系统里很有价值的特性。  
从论文表达上看，这也比单纯“语义更强”更贴近通信贡献。

### 3. 我们统一了两类缺失：主动降码率和被动丢包

这是 PDSSC 很强、也很有辨识度的一点。  
在我们的设计里：

- 主动码率选择带来的高层细节缺失
- 被动 packet loss 带来的高层细节缺失

都被统一成同一类 **speaker-related missing-detail restoration** 问题。

SemantiCodec 并没有建立这样的统一问题表述。  
这让我们的恢复模型不仅更有理论完整性，也更好解释为什么在不同丢包率和不同码率下都能工作。

### 4. 我们的恢复是“有靶点的补全”，不是重型生成式解码

SemantiCodec 使用 diffusion decoder，优点是生成质量强，但它本质上还是一个更重的生成式重建框架。  
PDSSC 的 flow completion 是在已有不完整 latent 上补缺失部分，恢复目标更局部、更明确：

- 语义骨架大多已经在低层保住
- 缺的主要是高层说话人相关和细粒度残差信息
- 接收端只需要针对这些缺失部分做补全

这类“targeted completion”更符合实时通信系统的需求，也更容易在时延和算力上讲出优势。

### 5. 我们显式利用 same-speaker side information

这一点让 PDSSC 在固定服务人群场景下更有工程合理性。  
SemantiCodec 追求的是跨 speech / music / sound 的通用性，因此不会围绕说话人先验来设计恢复。  
而 PDSSC 的场景更明确：用户群相对固定，因此接收端维护本地 speech library 是合理的。

这让我们的恢复过程：

- 更有针对性
- 更容易解释性能来源
- 更适合在低码率和高丢包下保持稳定

### 6. 我们有更强的“通信部署”叙事

如果从投稿叙事来看，PDSSC 比 SemantiCodec 更容易强调以下几个 deployment-oriented 卖点：

- packet-switched digital compatibility
- explicit packet-loss robustness
- bitrate adaptation
- low-step restoration
- latency evaluation
- real-time feasibility

SemantiCodec 虽然也很强，但它更偏“codec + semantic tokenization + general audio generation”路线；  
PDSSC 更偏“communication-oriented semantic system”路线。  
对于语音通信领域，这种定位通常更直接。

## 四、如果非要一句话判断“比起来如何”

如果按“通用音频语义 codec”的角度看，SemantiCodec 的视野更宽，general sound/musical audio 的泛化更强，token rate 也更激进。  
如果按“极低码率鲁棒语音通信系统”的角度看，PDSSC 的问题定义更完整、通信属性更强、系统闭环更清楚。

因此更准确的判断不是“谁完全压谁”，而是：

- **SemantiCodec 更强在：** 通用音频、语义 token richness、极低 token rate、diffusion-based reconstruction。
- **PDSSC 更强在：** 渐进传输、丢包鲁棒性、统一缺失建模、说话人侧信息恢复、自适应码率、实时部署。

## 五、写论文时可用的一句定位

如果你后面想在 related work 或 rebuttal 里一句话区分，可以用下面这个意思：

> SemantiCodec mainly targets ultra-low-rate semantic audio coding for general sound and audio language modeling, whereas PDSSC is designed for packet-switched speech communication, with explicit support for progressive transmission, packet-loss-aware recovery, adaptive bitrate control, and real-time deployment.

## 六、对我们写作上的启发

从 SemantiCodec 身上，反过来可以提醒我们在论文里进一步强调这几点：

1. 不要只说“我们也是 semantic codec”。  
要反复强调我们是 **communication system**。

2. 不要只说“语义和细节分离”。  
要强调我们是 **transmission-priority-oriented semantic layering**。

3. 不要只说“用了 flow model”。  
要强调它解决的是 **speaker-related missing-detail completion**，且是对主动截断和被动丢包的统一恢复。

4. 不要和它比谁码率更低。  
因为它本来就是 general audio ultra-low-token-rate 路线。  
我们的优势不在“更低 token rate”，而在“更完整的低码率鲁棒通信能力”。
