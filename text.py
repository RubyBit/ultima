from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from transformers import AutoTokenizer
from collections import Counter
import json
import collections

import datasets
from transformers import AutoTokenizer, LukeForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import json
from collections import Counter
from transformers import BertTokenizer,AutoTokenizer, BertForQuestionAnswering
import numpy as np
import unicodedata
from collections import Counter
from transformers import BertTokenizer, BertModel, AutoTokenizer
import torch
from torch import nn
from torch.optim import SGD
import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from tqdm import tqdm, trange
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer

import accelerate

print(accelerate.__version__)
# Load the base corpus (WikiText) and the domain corpus (SQuAD)
base_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:1%]')
domain_dataset = load_dataset("squad", split='train[:1%]')

# Function to extract texts for the unigram tokenizer training
def extract_texts(dataset):
    return "\n".join(dataset['text']).split('\n')

# Function to preprocess SQuAD dataset and return it as a list of strings
def preprocess_squad(data):
    corpus = []
    for example in data:
        combined_text = example['context'] + " " + example['question']
        corpus.append(combined_text)
    return corpus

# Initialize unigram tokenizers for base and domain
base_tokenizer = Tokenizer(Unigram())
domain_tokenizer = Tokenizer(Unigram())

# Trainer for the unigram tokenizers
trainer = UnigramTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

# Train the base tokenizer and save
base_texts = extract_texts(base_dataset)
base_tokenizer.train_from_iterator(base_texts, trainer)
base_tokenizer.save("base-unigram-tokenizer.json")

# Train the domain tokenizer and save
domain_corpus = preprocess_squad(domain_dataset)
domain_tokenizer.train_from_iterator(domain_corpus, trainer)
domain_tokenizer.save("domain-unigram-tokenizer.json")

# Load the BERT tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Load trained unigram tokenizers (if not already in memory)
# base_tokenizer = Tokenizer.from_file("base-unigram-tokenizer.json")
# domain_tokenizer = Tokenizer.from_file("domain-unigram-tokenizer.json")

# Compute sequence distributions
def compute_sequence_distribution(corpus, tokenizer):
    seq_counter = Counter()
    for sentence in corpus:
        # Tokenize the sentence using the BERT tokenizer instead of the unigram tokenizer
        tokens = bert_tokenizer.tokenize(sentence)
        for i in range(len(tokens)):
            for j in range(i+1, min(i+20, len(tokens)+1)):
                seq = tuple(tokens[i:j])
                seq_counter[seq] += 1
    return seq_counter

# Calculate base and domain sequence distributions
base_seq_distribution = compute_sequence_distribution(base_texts, bert_tokenizer)
domain_seq_distribution = compute_sequence_distribution(domain_corpus, bert_tokenizer)

# Normalize distributions
total_base = sum(base_seq_distribution.values())
total_domain = sum(domain_seq_distribution.values())
base_seq_probs = {seq: count / total_base for seq, count in base_seq_distribution.items()}
domain_seq_probs = {seq: count / total_domain for seq, count in domain_seq_distribution.items()}



def calculate_kl_divergence(base_probs, domain_probs):
    kl_divergence = {}
    for seq in base_probs.keys():
        if seq in domain_probs and base_probs[seq] > 0:
            kl_divergence[seq] = domain_probs[seq] * np.log(domain_probs[seq] / base_probs[seq])
    return kl_divergence



kl_divergence = calculate_kl_divergence(base_seq_probs, domain_seq_probs)
selected_tokens = [token for token, divergence in sorted(kl_divergence.items(), key=lambda item: item[1], reverse=True)]


F_min_base = 0.001  
F_min_domain = 0.001 

N = 100  # Number of augmentations to select
L = 5  # Max length of token sequences to consider for augmentation

augmentations = []

sorted_sequences = sorted(kl_divergence.items(), key=lambda item: item[1], reverse=True)

for seq, divergence in sorted_sequences:
    seq_str = ''.join(seq)  
    if len(augmentations) >= N:
        break
    if len(seq_str) <= L and domain_seq_probs[seq] >= F_min_domain and base_seq_probs[seq] >= F_min_base:
        augmentations.append(seq_str)


embedding_size = 768  # Typical embedding size for BERT models
num_common_tokens = 100  # Number of common tokens between source and target
num_source_tokens = 50  # Number of source domain-specific tokens
num_target_tokens = 50  # Number of target domain-specific tokens
# Assuming the same embedding sizes and initializations from before
d = 768 
model =  BertForQuestionAnswering.from_pretrained('bert-base-uncased') #BertModel.from_pretrained('bert-base-uncased')# LukeForQuestionAnswering.from_pretrained("studio-ousia/luke-base")

embedding_layer = model.get_input_embeddings()
new_tokens = augmentations
new_tokens = [bert_tokenizer.tokenize(seq) for seq in augmentations]
new_token_ids = [bert_tokenizer.convert_tokens_to_ids(tok_seq) for tok_seq in new_tokens]

flat_new_token_ids = list(set([item for sublist in new_token_ids for item in sublist]))

Xt = torch.randn((len(flat_new_token_ids), embedding_size))

common_sequences = set(base_seq_probs.keys()) & set(domain_seq_probs.keys())

common_token_ids = [bert_tokenizer.convert_tokens_to_ids(seq) for seq in common_sequences if len(seq) == 1]  
common_token_ids = [item for sublist in common_token_ids for item in sublist]  
common_token_ids = list(set(common_token_ids))  

Cs = embedding_layer.weight[common_token_ids]

source_specific_tokens = [token for token in bert_tokenizer.vocab.keys() if token not in new_tokens]
source_specific_token_ids = bert_tokenizer.convert_tokens_to_ids(source_specific_tokens)
#### NOT SURE ABOUT THIS AS I HAD TO MAKE DIMENSIONS COMPATIBLE:
source_specific_token_ids = [i for i in source_specific_token_ids if i in common_token_ids]
Xs = embedding_layer.weight[source_specific_token_ids]


M = torch.randn((d, d), requires_grad=True)
print(M.shape, Xs.shape,torch.matmul(M, Xs.T).shape)
print(Cs.T.shape)

def loss_function(M, Cs, Xs):
    return torch.norm(torch.matmul(M, Xs.T) - Cs.T, p='fro')

optimizer = SGD([M], lr=0.01)

num_iterations = 10

for _ in range(num_iterations):
    optimizer.zero_grad() 
    loss = loss_function(M, Cs, Xs)  
    loss.backward(retain_graph=True)  
    optimizer.step()  #

Ct = torch.matmul(M, Xt.T).T
Ct = Ct.detach().numpy()


bert_tokenizer.add_tokens(augmentations)

model.resize_token_embeddings(len(bert_tokenizer))
new_token_embeddings = torch.tensor(Ct)

embedding_layer = model.get_input_embeddings()
num_added_tokens = len(augmentations)

embedding_layer.weight.data[-num_added_tokens:, :] = new_token_embeddings
print(new_token_embeddings)


# tokenize the dataset
def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = bert_tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = bert_tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


dataset = load_dataset("squad")
tokenized_train_dataset = dataset["train"].map(preprocess_training_examples, batched=True,
                                               remove_columns=dataset["train"].column_names)
tokenized_validation_dataset = dataset["validation"].map(preprocess_validation_examples, batched=True,
                                                         remove_columns=dataset["validation"].column_names)

from tqdm.auto import tqdm
import numpy as np

n_best = 20
metric = datasets.load_metric("squad")

def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                            end_index < start_index
                            or end_index - start_index + 1 > 512
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

print("TRAINING")
args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True)
print("Working")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    tokenizer=bert_tokenizer,
)
print("Working2")
trainer.train()