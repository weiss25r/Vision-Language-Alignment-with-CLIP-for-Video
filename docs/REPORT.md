# Vision-Language Alignment with CLIP for video
- **Group ID**: Justgood AI
- **Project ID**: 15
- **Group members**: Edoardo Tantari and Raffaele Terracino

---

## 1. Introduction and Objective
*Describe the objective of the project, why it is relevant, and what specific problem you are trying to solve. What is your main goal or initial hypothesis?*

## 2. Contribution and Added Value
We built a text-video retrieval architecture on the EPIC-KITCHENS dataset, allowing semantic video retrieval on arbitraries text queries. We trained our model on top of pre-trained frozen foundational models, experimenting with different architectures and losses. We demostrante that for this task our simple adapter - consisting of a standard, lightweight MLP - obtain competitive performance, allowing robust semantic search engines on the dataset using lightweights architectures.

## 3. Data Used
### Dataset description
The project is built upon the [EPIC-KITCHENS-100 dataset](https://epic-kitchens.github.io/), containing 100 hours of egocentric footage. The dataset contains over 20 millions frames, for a total of 20.5K narrations. Each narration is made up of a verb and a noun (e.g open door), and each unique video is composed of multiple clips, each with its corresponding  Annotations for training and validation set are provided alongisde an un-annoteded test set. Because of this, we use the provided validation set as a test set. 
### Sampling and zero-shot scenario
For storage limitations, we sample 300 unique video from the training set. We then split it 85% for training and 15% for validation, to perform model selection on various experiments.  
To validate our model capabilites on entirely unseen verbs and nouns, we split our test sets into a "seen" one, containing narrations the model saw in training, and "zero-shot" containing unseen narrations. A verb/noun class is classified as a zero-shot class if it is sufficiently rare in the training set (not among the top 15–20) but sufficiently common in the validation set (≥20 samples), so as to provide a meaningful test on classes that the model did not ‘dominate’ during training. Thus, we exclude from the training set all samples containing a zero-shot noun or a zero-shot verb. Sampling and zero-shot selection is done automatically via the script ```src/dataset/sample_dataset.py```. We downloaded only sampled rgb frames and extracted them automatically using script We downloaded only this 300 video from the academic torrent and extracted rgb frames from tar files using the script ```src/dataset/extract_frames.py```. To replicate our experiments, we provide our annotations in ```data/annotations/processed```. During training, we use the narration for each clip, consisting of the sum of the verb and noun.

## Statistics
The following table resume statistics for each set.
| Split | Number of samples | Distinct verbs | Distinct nouns |
|---|---:|---:|---:|
| Training set | 29619 | 91 | 270 |
| Validation set | 5802 | 78 | 172 |
| Test seen set | 8648 | 77 | 206 |
| Test zero-shot set | 1020 | 38 | 94 |

## 4. Methodology and Architecture
*Detail the experiments and how your system is built. What architecture did you use as a baseline? How did you modify it? Describe the network topology, key layers, the loss function used, and the training logic.*

### Baseline architecture
For our baseline, we choose [TimeSformer](paper) as video encoder and [DistilBERT](paper) as text encoder. The choice is motivated by the necessity of having a powerful video encoder, which translates into a time-consuming architecture, and a text encoder capable of correctly captures the semantic of narrations while not consuming too much time and energy. Since both encoders are frozen in the baseline, we perform offline feature extraction on sets to speed up training. For DistilBERT we take the mean pooling of the last layer for each narration, while for TimeSformer we directly use the [CLS] token. This is done automatically in script ```src/dataset/features_extraction.py```, in which we use the default "distilbarte-base-uncased" model and "timesformer-base-finetuned-k600", both included within the HuggingFace transformers library. On top of Timesformer and DistilBERT features, we train an adapter consisting of two indipendent MLP with one hidden layer. The adapter's pourpose is to align the text embedding space with the video one. The structure is one typical of MLPs: hidden layer, layer normalization, dropout, output layer. The adapter is trained using a contrastive loss exactly as implemented in [CLIP paper](paper). Hyperparameters for the baseline are provided

### Experiments
To improve our baseline, we perform the following experiments:
- Fine-tuning DistilBERT and the last layer of TimeSformer. In this setting, we use a simple adapter consisting of a single linear layer to avoid losing pre-training information. TimeSformer' learning rate is set as $1/10$ of DistilBERT one. During training, 
- Changing frozen encoders: we try to change the input features mantaining the same Adapter layout. We first switch to original CLIP, using its text encoder and image encoder. To perform video encoding, we take the mean pooling of encoded frames. We then try using EgoVLP+ encoders, fine-tuned on EPIC-Kitchens
- Coerently with the choice of trying EgoVLP+ features, we switch from the standard CLIP loss to a similar EgoVLP loss using action-positive sampling: TOOD SPIEGARE.

Hyperparameters for each experiments are provided in the experiments/configs folder. Results are commented in the next section.

### Sanity checks on features
To start each experiment concerning features, we perform two sanity checks before training:
- a zero-shot evaluation, where we evaluate Recall@K without the Adapter module
- training of a logistic classifier on verbs

Sanity checks allow us to answer the question "how good are the input features?" prior to training and are performed on validation set.

### Evaluation metrics
To perform model selection and evaluate our models, we compute Multi-Instance Recall. Standard Recall@K assumes a single relevant item per query in the gallery, which is unrealistic in EPIC-KITCHENS where multiple clips can depict the same action. We therefore adopt **Multi-Instance Recall@K**, which considers a query successful if at least one of the top-K retrieved items is semantically equivalent to it.

Two samples $i$ and $j$ are semantically equivalent if they share both verb and noun class:

$$\text{match}(i,j) = \mathbf{1}[\text{verb}_i = \text{verb}_j] \wedge \mathbf{1}[\text{noun}_i = \text{noun}_j]$$

The metric is then computed as:

$$\text{MIR@K} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left[\exists\, j \in \text{top-K}(i) : \text{match}(i,j) = 1\right]$$

The model with the highest MIR@1 on the validation set is finally evaluated on the test set, concluding the experimental phase.

## 5. Results and Discussion
Insert here the quantitative tables with the achieved results and compare your solution with the baseline. **Do not limit yourself to pasting numbers**, but comment on them:
- Why does model A perform better than model B?
- Are there classes where the model is particularly weak?
- Show qualitative examples (e.g., inserting correctly vs. incorrectly predicted images).*

We report quantitative results for each experiment, including sanity checks on features and evaluation metrics of trained

### Sanity Checks on input video features 
| Experiment | Verb classification accuracy | Zero-shot R@1 | Zero-shot R@5 | Zero-shot R@10 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 32 | 0.1 | 1.4 | 3.6 |
| CLIP | 28 | 5.5 | 20.9 | 31.2 |
| EgoVLP fine-tuned features | 77 | 52.6 | 79.2 | 85.8 |

The effect of pre-extracted features is clear. Using baseline' features, it is clear that Timesformer cannot produce a good representation of the egocentric setting of EPIC-KITCHENS, as the logistic regressor cannot produce a good boundary between classes. The same happens with mean-pooled CLIP features. However, using the latest, there is a clear gap in zero-shot retrieval performance, with an increase of +27.6% in R@10. This is explainable by the fact that, while CLIP, as it it a model for text-image retrieval, cannot model motion as TimeSformer, but it is capable of producing good text representation that allows retrieval. On the other hand, EgoVLP+, fine-tuned on EPIC-KITCHENS 100 for the Multi-Instance Retrieval task, produces the best features so far, capable of high-performance zero-shot retrival as well as being good for classification on verbs. 

### R@K on Validation set
The following tables shows metrics computed on validation set for each trained model, highlightning the best one.
Experiment | R@1 | R@5 | R@10 |
| --- | --- | --- | --- |
| Baseline | 26.4 | 57.7 | 66.1 |
| Fine tuning last layer TSF + DistilBERT | 25.4 | 54.6 | 66.1 |
| CLIP features | 21.0 | 44.3 | 55.0 |
| EgoVLP+ CLIP loss | 64.7 | 84.6 | 89.6 |
| **EgoVLP+ egonce loss** | **68.9** | **84.1** | **88.2** |

### Analysis
Fine tuning TimeSformer' last layer and full fine tuning DistilBERT seem to destroy the encoder capabilites, as metrics are very similar. Furthermore, training this architecture is highly time-consuming, as it takes almost 10 minutes per epoch using our machine. This experiment led us to abandon completely fine-tuning approaches. CLIP features - as opposed to their good sanity checks - performs clearly worst than our baseline. Our intuition is that while CLIP's text encoder provided good representations for our narrations, the absence of motion-modeling capabilites of the mean-pooling is a clear bottleneck. Our adapter then destroys extracted features.

## 6. Conclusion and Limitations
*Summarize the project's outcome. What are the current limitations (e.g., requires too much memory, fails in low-light conditions)? If you had more time, what future experiments would you run?*

## 7. Additional Information

### 7.1 Contribution Breakdown
*Detail clearly who did what within the group.*
- **Person 1**: ...
- **Person 2**: ...
- **Person 3**: ...

### 7.2 Use of Artificial Intelligence
*Declare here the possible use of tools like Copilot or ChatGPT, specifying in which phases they helped you (e.g., writing boilerplate, debugging, documentation), keeping in mind that the architectural design and the responsibility for the result are yours.*
