# Vision-Language Alignment with CLIP for video
- **Group ID**: Justgood AI
- **Project ID**: 15
- **Group members**: Edoardo Tantari and Raffaele Terracino

---

## 1. Introduction and Objective
The ability to retrieve video clips from natural language queries is a fundamental challenge in computer vision. Traditional retrieval systems rely on manual annotations or metadata — an approach that is brittle, expensive, and does not scale to the volume of video data produced today. The core question we address is: how do you find a video clip you have never labeled?

Our goal is to build a system that, given an arbitrary text query such as "person holding a flag on a mountain peak", retrieves the semantically matching clip from a large video collection — with no manual labels and no fine-tuning on the target domain. This is the text-video retrieval problem, and it sits at the intersection of vision-language alignment and zero-shot generalization.

We tackle this problem on EPIC-KITCHENS-100, a challenging egocentric dataset where clips depict fine-grained kitchen actions. This setting is particularly demanding: actions are visually similar ("take onion" vs "take skin"), the same action can appear in dozens of different clips, and the vocabulary of verbs and nouns is highly imbalanced. These properties make naive retrieval approaches fail, and push the limits of vision-language models not originally designed for egocentric video.

## 2. Contribution and Added Value
We built a text-video retrieval architecture on the EPIC-KITCHENS dataset, allowing semantic video retrieval on arbitrary text queries. We trained our model on top of pre-trained frozen foundational models, experimenting with different architectures and losses. We demonstrate that for this task our simple adapter - consisting of a standard, lightweight MLP - obtains competitive performance, allowing for robust semantic search engines on the dataset using lightweight architectures.

## 3. Data Used

### Dataset description
The project is built upon the [EPIC-KITCHENS-100 dataset](https://epic-kitchens.github.io/), containing 100 hours of egocentric footage. The dataset contains over 20 million frames, for a total of 20.5K narrations. Each narration is made up of a verb and a noun (e.g., open door), and each unique video is composed of multiple clips. Annotations for the training and validation sets are provided alongside an unannotated test set. Because of this, we use the provided validation set as a test set. 

### Sampling and zero-shot scenario
Due to storage limitations, we sample 300 unique videos from the training set. We then split it 85% for training and 15% for validation to perform model selection on various experiments.  

To validate our model's capabilities on entirely unseen verbs and nouns, we split our test set into a "seen" one, containing narrations the model saw during training, and a "zero-shot" one, containing unseen narrations. A verb/noun class is classified as a zero-shot class if it is sufficiently rare in the training set (not among the top 15–20) but sufficiently common in the test set (≥20 samples), so as to provide a meaningful test on classes that the model did not ‘dominate’ during training. Thus, we exclude from the training set all samples containing a zero-shot noun or a zero-shot verb. 

Sampling and zero-shot selection are done automatically via the script `src/dataset/sample_dataset.py`. We downloaded only these 300 videos from the academic torrent and extracted the RGB frames from the tar files using the script `src/dataset/extract_frames.py`. To replicate our experiments, we provide our annotations in `data/annotations/processed`. During training, we use the narration for each clip, consisting of the sum of the verb and noun.

### Statistics
The following table summarizes the statistics for each set.

| Split | Number of samples | 
|---|---:|
| Training set | 29619 |
| Validation set | 5802 | 
| Test seen set | 8648 |
| Test zero-shot set | 1020 | 

## 4. Methodology and Architecture

### Baseline architecture
For our baseline, we choose [TimeSformer](https://arxiv.org/abs/2102.05095) as the video encoder and [DistilBERT](https://arxiv.org/abs/1910.01108) as the text encoder. The choice is motivated by the necessity of having a powerful video encoder, which translates into a time-consuming architecture, and a text encoder capable of correctly capturing the semantics of narrations while not consuming too much time and energy. Since both encoders are frozen in the baseline, we perform offline feature extraction on sets to speed up training. 

For DistilBERT, we take the mean pooling of the last layer for each narration. This is motivated by the absence of the Next-Sentence prediction task in DistilBERT's pre-training. To extract features with TimeSformer base, which works at a temporal resolution of 8 frames, we divide each clip into 8 bins and choose the central frame as its representative. To extract the final embedding, we directly use the [CLS] token. This is done automatically in the script `src/datasets/features_extraction.py`, in which we use the default [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) model and [timesformer-base-finetuned-k600](https://huggingface.co/facebook/timesformer-base-finetuned-k600), both included within the HuggingFace transformers library. 

In our implementation, we define two Pytorch Dataset classes: EpicKitchensFramesDataset to load raw frames for performing feature extraction, and EpicKitchensFeaturesDataset to load pre-extracted features for our baseline. Lightning DataModules for these classes are also defined. This is done in `src/datasets/dataset.py`

On top of Timesformer and DistilBERT features, we train an adapter consisting of two independent MLPs with one hidden layer. The adapter's purpose is to align the text embedding space with the video one. The structure is typical of MLPs: hidden layer, layer normalization, dropout, output layer. The adapter is trained using a contrastive loss exactly as implemented in the [CLIP paper](https://arxiv.org/abs/2103.00020). As in CLIP, temperature is a learnable parameter initialized as $\log(1/0.07)$. Our baseline is implemented as a LightningModule in `src/models/adapter.py`.Hyperparameters for the baseline are provided in `experiments/configs/MLP_timesformer_config.yaml`. 

### Evaluation metrics
To perform model selection and evaluate our models, we compute Multi-Instance Recall. Standard Recall@K assumes a single relevant item per query in the gallery, which is unrealistic in EPIC-KITCHENS where multiple clips can depict the same action. We therefore adopt **Multi-Instance Recall@K**, which considers a query successful if at least one of the top-K retrieved items is semantically equivalent to it.

Two samples $i$ and $j$ are semantically equivalent if they share both verb and noun class:

$$\text{match}(i,j) = \mathbf{1}[\text{verb}_i = \text{verb}_j] \wedge \mathbf{1}[\text{noun}_i = \text{noun}_j]$$

The metric is then computed as:

$$\text{MIR@K} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left[\exists\, j \in \text{top-K}(i) : \text{match}(i,j) = 1\right]$$

The formula is translated into code in file `src/evaluation/metrics.py`

### Experiments
To improve our baseline, we perform a series of experiments. The seed is set to 42 for all experiments for reproducibility. Hyperparameters for each experiment are provided in the `experiments/configs` folder. The model with the highest MIR@1 on the validation set is finally evaluated on the test set, concluding the experimental phase.

#### Fine-tuning
In this setting, we fine-tune DistilBERT and the last layer of TimeSformer on EPIC KITCHENS. In this setting, we use a simple adapter consisting of a single linear layer to avoid losing pre-training information. TimeSformer's learning rate is set as $1/10$ of DistilBERT's. During training, we perform temporal data augmentation. Since Timesformer works with videos of 8 frames, we divide each clip into 8 bins and randomly select a frame as a representative for each bin. We also use a cosine learning rate scheduler with warmup. The full model is implemented in `src/model/clip.py`

#### Changing frozen encoders
We try to change the input features while maintaining the same Adapter layout. We first switch to original CLIP, using its text encoder and image encoder. To perform video encoding, we take the mean pooling of the encoded frames. We then try using EgoVLP+ encoders, fine-tuned on EPIC-Kitchens. 

To start each experiment concerning features, we perform two sanity checks before training:
- a zero-shot evaluation, where we evaluate Recall@K without the Adapter module
- training of a logistic classifier on verbs

Sanity checks allow us to answer the question "how good are the input features?" prior to training and are performed on the validation set. Sanity checks are performed via Jupyter Notebooks `notebooks/zeroshot_eval.py` and `notebooks/train_classifier.py 

Feature extraction with CLIP and EgoVLP+ is done via scripts `src/dataset/clip_fe.py` and `src/dataset/egovlp_fe.py`.

#### Changing the loss
Coherently with the choice of trying EgoVLP+ features, we switch from the standard CLIP loss to a contrastive loss inspired by EgoNCE, simplified to use only action-aware positive sampling, without the scene-aware negative sampling of the original work.

The standard CLIP loss (InfoNCE) treats each video-text pair as the unique positive for the other, and all other samples in the batch as negatives:

$$\mathcal{L}^{\text{CLIP}}_{v2t} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\mathbf{v}_i^T \mathbf{t}_i / \tau)}{\sum_{j=1}^{N} \exp(\mathbf{v}_i^T \mathbf{t}_j / \tau)}$$

This is problematic in egocentric datasets like EPIC-KITCHENS-100, where multiple clips can depict the same action (e.g., several videos of *"put down the knife"*). Treating semantically equivalent samples as negatives produces a noisy and misleading training signal.

Following EgoNCE, we define a positive set $\mathcal{P}_i$ for each sample $i$ as all samples in the batch that share the same verb class **and** the same noun class:

$$\mathcal{P}_i = \{j \in \mathcal{B} \mid \text{verb}(j) = \text{verb}(i) \;\wedge\; \text{noun}(j) = \text{noun}(i)\}$$

This resembles our definition of MIR@K. The loss then places probability mass on the entire positive set rather than a single pair:

$$\mathcal{L}^{\text{ego}}_{v2t} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\sum_{k \in \mathcal{P}_i} \exp(\mathbf{v}_i^T \mathbf{t}_k / \tau)}{\sum_{j=1}^{N} \exp(\mathbf{v}_i^T \mathbf{t}_j / \tau)}$$

The full loss is the average of the symmetric video-to-text and text-to-video terms:

$$\mathcal{L}^{\text{ego}} = \frac{1}{2}\left(\mathcal{L}^{\text{ego}}_{v2t} + \mathcal{L}^{\text{ego}}_{t2v}\right)$$

where $\mathbf{v}_i$ and $\mathbf{t}_i$ are L2-normalized video and text embeddings, and $\tau$ is the temperature, which we make configurable to be a hyperparameter or a learnable parameter initialized to $\log(1/0.05)$.

## 5. Results and Discussion
We report quantitative results for each experiment, including sanity checks on features and evaluation metrics of trained models.

### Sanity Checks on input video features 
| Experiment | Verb classification accuracy | Zero-shot MIR@1 | Zero-shot MIR@5 | Zero-shot MIR@10 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 32 | 0.1 | 1.4 | 3.6 |
| CLIP | 28 | 5.5 | 20.9 | 31.2 |
| EgoVLP fine-tuned features | 77 | 52.6 | 79.2 | 85.8 |

The effect of pre-extracted features is clear. Using the baseline's features, it is clear that Timesformer cannot produce a good representation of the egocentric setting of EPIC-KITCHENS, as the logistic regressor cannot produce a good boundary between classes. The same happens with mean-pooled CLIP features. However, using the latter, there is a clear gap in zero-shot retrieval performance, with an increase of +27.6% in MIR@10. This is explainable by the fact that, while CLIP (as a model for text-image retrieval) cannot model motion like TimeSformer, it is capable of producing good text representations that allow retrieval. On the other hand, EgoVLP+, fine-tuned on EPIC-KITCHENS 100 for the Multi-Instance Retrieval task, produces the best features so far, capable of high-performance zero-shot retrieval as well as being good for classification on verbs. 

### R@K on Validation set
The following table shows metrics computed on the validation set for each trained model, highlighting the best one.

| Experiment | R@1 | R@5 | R@10 |
| --- | --- | --- | --- |
| Baseline | 26.4 | 57.7 | 66.1 |
| Fine tuning | 25.4 | 54.6 | 66.1 |
| CLIP features | 21.0 | 44.3 | 55.0 |
| EgoVLP+ CLIP loss | 64.7 | 84.6 | 89.6 |
| **EgoVLP+ egonce loss** | **68.9** | **84.1** | **88.2** |

### Comments
Fine-tuning TimeSformer's last layer and full fine-tuning DistilBERT seem to destroy the encoders' capabilities, as metrics are very similar to the baseline. Furthermore, training this architecture is highly time-consuming, taking approximately 12 minutes per epoch using our machine. This experiment led us to completely abandon fine-tuning approaches. CLIP features - as opposed to their good sanity checks - perform clearly worse than our baseline. Our intuition is that while CLIP's text encoder provided good representations for our narrations, as demonstrated by the sanity check, the video embeddings are not representative and as such are easily destroyed by the adapter. 

On the other hand, EgoVLP+ demonstrates the power of its pre-training on Ego4D and fine-tuning on EPIC KITCHENS, as it produces very good features for both classification and zero-shot retrieval. Because the features are good enough, the adapter refines their geometry and obtains a +40% increase in R@1. EgoVLP+ with EgoNCE loss achieves the best R@1, showing a +4.2% increase over the CLIP loss, at the cost of a slight drop at higher K. This trade-off is consistent with what the loss is designed to do: by grouping semantically equivalent samples into a positive set $P_i$ rather than treating them as hard negatives, the model learns a more clustered embedding geometry around exact action classes. This directly optimizes for top-1 precision but reduces the sharpness of separation between semantically close — yet distinct — categories, slightly penalizing ranking quality at K=5 and K=10. This model is thus the best-trained model, having obtained the highest MIR@1 on the validation set.

### Test sets metrics for best trained model

| Split | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| Test seen | 52.39 | 70.28 | 76.34 |
| Test zero-shot | 49.61 | 73.24 | 81.18 |

Zero-shot retrieval outperforms seen at R@5 and R@10, despite a lower R@1. This is explained by the gallery density: seen classes are the most frequent in the dataset, meaning the gallery contains many semantically equivalent but visually diverse clips for the same action. This makes top-1 retrieval easier (the model has seen these classes) but ranking harder at higher K, as more candidates compete for the same slots. Zero-shot classes, being rare by construction, produce a less crowded gallery, making it easier to retrieve at least one positive in the top-5/10 — even without having seen the class during training.

### Qualitative results
The following table shows qualitative examples of misunderstood predictions by the best-trained model on the "seen" test set.

| Query (GT) | Predicted (top-1) | Cosine similarity |
| --- | --- | ---: |
| take paper | move cooking paper | 0.6457 |
| take onion | take skin | 0.6451 |
| wipe cooker | clean kitchen | 0.7037 |
| put down cloth | hang up cloth | 0.7137 |
| throw onion in | throw eggs box | 0.5152 |
| hand cloth | squeeze cloth | 0.6189 |
| throw paper into bin | throw bag | 0.6137 |
| take hand | pick up ball of dough | 0.5322 |
| throw can into bin | put rubbish in bin | 0.6179 |
| take bin | throw away bits | 0.5928 |

We group the failure cases into three categories. Semantically close errors ("wipe cooker" → "clean kitchen", "put down cloth" → "hang up cloth", "throw can into bin" → "put rubbish in bin") show the highest cosine similarities (≥0.70), suggesting the model retrieves visually and semantically plausible alternatives — errors a human might arguably consider acceptable. Verb confusion errors ("take onion" → "take skin", "hand cloth" → "squeeze cloth") indicate that the model captures the object correctly but struggles to discriminate fine-grained motion differences. Finally, complete failures ("take hand" → "pick up ball of dough", "throw onion in" → "throw eggs box") show the lowest similarities ( $\le 0.52$) and likely correspond to ambiguous or visually cluttered clips where neither the object nor the action is reliably encoded.

## 6. Conclusion and Limitations
We presented a lightweight adapter-based architecture for text-video retrieval on EPIC-KITCHENS-100, demonstrating that a simple MLP trained on top of frozen EgoVLP+ features achieves competitive performance (R@1 = 68.9% on validation). The key insight is that feature quality dominates over adapter complexity: pre-training and domain-specific fine-tuning of the encoder matter far more than the downstream architecture.

**Limitations.** Our experiments are constrained by several practical factors. First, storage and dataloader throughput forced us to work on a sampled subset of 300 videos, which may underrepresent rare verb-noun combinations. Second, EgoVLP+ was used as a frozen feature extractor only, as full end-to-end fine-tuning was infeasible on a single RTX 4090. Third, qualitative analysis reveals that the model struggles with fine-grained motion discrimination — confusing semantically similar verbs acting on the same object — suggesting that richer temporal representations would be beneficial.

**Future work.** End-to-end fine-tuning of EgoVLP+ with a multi-GPU setup would likely push R@1 significantly beyond current results. Training on the full dataset would improve the coverage of rare verb-noun pairs. Finally, implementing the complete EgoNCE loss with scene-aware negative sampling — which we simplified away — could further improve embedding geometry.

## 7. Additional Information

### 7.1 Contribution Breakdown
- **Edoardo Tantari**: data collection, video/text encoder classes, feature extraction (Baseline, CLIP, EgoVLP+), experiment configurations, presentation.
- **Raffaele Terracino**: dataset classes, alignment module, loss functions, evaluation metrics, CLI interface, report writing.

### 7.2 Use of Artificial Intelligence
During the development of this project, AI tools (Gemini and Claude) were utilized as supportive aids to optimize our workflow and accelerate development. Specifically, they assisted us in the following areas:

* **Code Integration:** Adapting and integrating the EgoVLP code from the original external repository into our project, specifically to perform feature extraction;
* **Code Quality:** Performing general code refactoring and cleanup to improve overall readability, structure, and maintainability.
* **Framework Documentation:** Acting as a rapid search and consultation tool to quickly understand specific functionalities, boilerplate requirements, and best practices within the PyTorch Lightning framework.

While these tools significantly aided in writing boilerplate code and resolving minor bugs, the core architectural design, the experimental setup, and the ultimate responsibility for the project's results and integrity remain strictly our own.