# %% [markdown]
# # Imports

# %%
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import math

# %% [markdown]
# # Task 1: Vocabulary Creation 

# %% [markdown]
# ### Read train.json

# %%
with open('data/train.json') as f:
    train_data = json.load(f)

# %% [markdown]
# ### Count occurence of every word

# %%
vocab_dict = {}
for train in train_data:
    for word in train['sentence']:
        vocab_dict[word] = 1 + vocab_dict.get(word, 0)

# %% [markdown]
# ### Count unknown words

# %%
threshold = 2
unknown_count = 0
for key in vocab_dict.copy():
    if vocab_dict[key] < threshold:
        unknown_count += vocab_dict[key]
        del vocab_dict[key]

# %% [markdown]
# ### Sort based on occurence

# %%
sorted_vocab = sorted(vocab_dict.items(), key=lambda x:x[1], reverse=True)

# %% [markdown]
# ### Create vocab.txt

# %%
index = 1
with open('vocab.txt', 'w') as f:
    f.write('unk\t{}\t{}\n'.format(index, unknown_count))
    index += 1
    for key, val in sorted_vocab:
        f.write('%s\t%s\t%s\n' % (key, index, val))
        index += 1

# %%
print("Threshold value for identifying unknown words for replacement:", threshold)
print("The overall size of my vocabulary:", len(sorted_vocab)+1)
print(f"The special token ”< unk >” occurs {unknown_count} times following the replacement process.")

# %% [markdown]
# # Task 2: Model Learning

# %% [markdown]
# #### Reading Training data and vocab.txt

# %%
with open('data/train.json', 'r') as train_file:
    training_data = json.load(train_file)

# Read vocab data from vocab.txt
vocab_data = [line.split()[0] for line in open('vocab.txt') if line.strip()]

# %% [markdown]
# ### Generating transition and emission parameters

# %%
transition, emission, label_dict = {}, {}, {}
unique_label = set()

for record in training_data:
    labels = record['labels']
    sentence = record['sentence']

    for i in range(len(labels)):
        label = labels[i]
        word = sentence[i]

        unique_label.add(label)
        label_dict[label] = 1 + label_dict.get(label, 0)

        if word.isdigit():
            word = '<isdigit>'

        if word not in vocab_data:
            word = '<unk>'

        emission[(label, word)] = 1 + emission.get((label, word), 0)

        if i == 0:
            transition[('.', label)] = 1 + transition.get(('.', label), 0)

        else:
            prev_label = labels[i-1]
            transition[(prev_label, label)] = 1 + transition.get((prev_label, label), 0)


# %% [markdown]
# #### Performing laplace_smoothing to remove any 0 probability from the transition parameter

# %%
label_total = {}
for t in transition:
    label_total[t[0]] = transition[t] + label_total.get(t[0], 0)

# %%
def laplace_smoothing(transition_dict_counts, k, states, label_total):
    trans_probs = {}
    for current_state in states:
        for next_state in states:
            if (current_state,next_state) in transition_dict_counts:
                tg=transition_dict_counts[current_state,next_state]
            else:
                tg=0
            trans_probs[current_state, next_state] = (tg + k) / (label_total[current_state] + k * len(states))
    return trans_probs

transition = laplace_smoothing(transition, 1, list(unique_label), label_total)

# %% [markdown]
# ### Emission Parameter

# %%
for i in emission:
    emission[i] /= label_dict[i[0]]

# %% [markdown]
# ### Generating hmm.json

# %%
transition_parameter = {}
for key, val in transition.items():    
    k1, k2 = key[0], key[1]
    transition_parameter[str(k1)+","+str(k2)] = val

emission_parameter = {}
for key, val in emission.items():    
    k1, k2 = key[0], key[1]
    emission_parameter[str(k1)+","+str(k2)] = val

hmm = {}
hmm['transition'] = transition_parameter
hmm['emission'] = emission_parameter

with open('hmm.json', 'w') as json_file:
    json.dump(hmm, json_file, indent=4, sort_keys=True)    

# %% [markdown]
# ### How many transition and emission parameters in your HMM?

# %%
with open('hmm.json', 'r') as json_file:
    data = json.load(json_file)

print("Number of transition parameters", len(data['transition']))
print("Number of emission parameters", len(data['emission']))

# %% [markdown]
# # Task 3: Greedy Decoding with HMM

# %%
with open('data/dev.json', 'r') as dev_file:
    dev_data = json.load(dev_file)

dev_words = [word for dev in dev_data for word in dev['sentence']]

unknown_words = {}
for word in train['sentence']:
    if word not in sorted_vocab:
        unknown_words[word] = 1 + unknown_words.get(word, 0)

# %%
def greedy_decoding(words, unique_label):
    res = []
    all_state = []
    T = list(unique_label)
    k = 1

    for key, word in enumerate(words):        
        t = ""
        max_pr = 0

        if word not in vocab_data:
            res.append((word, 'NNP'))
            all_state.append('NNP')
            continue

        for tag in T:
            stateval = 0
            if key == 0:
                trans_val = transition[('.', tag)]
            else:
                trans_val = transition[(all_state[-1], tag)]

            for z in emission:
                if z[1] == word and z[0] == tag:
                    yi = trans_val*emission[z]
                    stateval += yi

            if stateval >= max_pr:
                max_pr = stateval
                t = tag
                
        all_state.append(t)
        res.append((word, t))

        k=k+1
        if k%500==0:
          print(k)

    return res    

# %%
res = greedy_decoding(dev_words, unique_label)

# %%
ground_truth = [(word, label) for entry in dev_data for word, label in zip(entry["sentence"], entry["labels"])]

correct_predictions = sum(1 for (wordA, labelA), (wordB, labelB) in zip(res, ground_truth) if labelA == labelB)
total_predictions = len(ground_truth)

accuracy = (correct_predictions / total_predictions) * 100
incorrect = 100 - accuracy

print("Accuracy for Greedy Decoding using HMM:", accuracy)
print("Error Rate:", incorrect)

# %% [markdown]
# ### Predicting on Test Data

# %%
with open('data/test.json', 'r') as test_file:
    test_data = json.load(test_file)

test_words = [word for test in test_data for word in test['sentence']]

test_res = greedy_decoding(test_words, unique_label)

# %% [markdown]
# ### Generating greedy.json

# %%
with open('data/test.json', 'r') as test_file:
    greedy_data = json.load(test_file)

k = 0
for i in greedy_data:
    test_labels = [label for word, label in zip(i['sentence'], test_res[k:k+len(i['sentence'])])]
    k += len(i['sentence'])
    i['labels'] = test_labels

output_file = 'greedy.json'
with open(output_file, "w") as outfile:
    json.dump(greedy_data, outfile, indent=4)

# %% [markdown]
# # Task 4: Viterbi Decoding with HMM

# %%
with open('data/dev.json', 'r') as dev_file:
    dev_data = json.load(dev_file)

dev_words = [word for dev in dev_data for word in dev['sentence']]

dev_sentences = [[word for word in dev['sentence']] for dev in dev_data]

dev_sentences2 = [[f"{word}/{label}" for word, label in zip(dev['sentence'], dev['labels'])] for dev in dev_data]


# %%
NN_SUFFIX = ["action", "age", "ance", "cy", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling",
               "ment", "ness", "or", "ry", "scape", "ship", "dom", "ty"]
VB_SUFFIX = ["ed", "ify", "ise", "ize", "ate", "ing"]
JJ_SUFFIX = ["ous", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "able","wise"]
ADV_SUFFIX = ["ward"]
VBG_suffix="ing"
VBN_suffix="ed"
NNS_suffix=["s","ies","wards","es"]

def viterbi(sentence_list, unique_label, emission, transition):
    def get_emission_probability(tag, word):
        return emission.get((tag, word), SMALL_PROB)

    def get_transition_probability(prev_tag, current_tag):
        return transition.get((prev_tag, current_tag), SMALL_PROB)

    def handle_unknown_word(word):
        if any(char.isdigit() for char in word):
            if word.startswith('$'):
                return "CD"
            return "CD"
        elif any(char.isupper() for char in word):
            return "NNP"
        elif any(word.endswith(suffix) for suffix in NN_SUFFIX):
            return "NN"
        elif any(word.endswith(suffix) for suffix in VB_SUFFIX):
            return "VB"
        elif any(word.endswith(suffix) for suffix in JJ_SUFFIX):
            return "JJ"
        elif any(word.endswith(suffix) for suffix in ADV_SUFFIX):
            return "RB"
        elif any(word.endswith(suffix) for suffix in NNS_suffix):
            return "NNS"
        elif word.endswith("ing"):
            return "VBG"
        elif word.endswith("ed"):
            return "VBN"
        elif word.istitle():
            return "NNP"
        elif word.endswith("'s"):
            return"POS"
        elif '-' in word:
            return "JJ"
        return "NNP"

    result = []
    SMALL_PROB = 1e-10  # A small value to avoid zero probabilities

    for sentences in sentence_list:
        V = []
        V_BP = []

        for wno, word in enumerate(sentences):
            V.append({})
            V_BP.append({})

            if wno == 0:
                for t2 in unique_label:
                    et = get_emission_probability(t2, word)
                    tt = get_transition_probability('.', t2)
                    V[wno][t2] = et * tt
                    V_BP[wno][t2] = '.'

            else:
                for t2 in unique_label:
                    max_prob = -math.inf
                    best_prev_tag = None

                    for t1 in V_BP[wno - 1]:
                        et = get_emission_probability(t2, word)
                        tt = get_transition_probability(t1, t2)
                        curr_prob = V[wno - 1][t1] + math.log(et) + math.log(tt)

                        if curr_prob > max_prob:
                            max_prob = curr_prob
                            best_prev_tag = t1

                    V[wno][t2] = max_prob
                    V_BP[wno][t2] = best_prev_tag

                # Handle unknown words and digits
                if all(get_emission_probability(t2, word) == SMALL_PROB for t2 in unique_label):
                    if word.isdigit():
                        unknown_tag = '<isdigit>'
                    else:
                        unknown_tag = handle_unknown_word(word)
                    V[wno][unknown_tag] = max(V[wno].values())
                    V_BP[wno][unknown_tag] = max(V_BP[wno].values())

        best_tag = max(V[-1], key=V[-1].get)
        tagged_sentence = [sentences[-1] + '/' + best_tag]

        for i in range(len(V) - 2, -1, -1):
            best_tag = V_BP[i + 1][best_tag]
            tagged_sentence.append(sentences[i] + '/' + best_tag)

        result.append(tagged_sentence[::-1])  # Reverse the list for the correct order

    return result


# %%
dev_vit=viterbi(dev_sentences, unique_label, emission, transition)

# %%
def accuracy_calculate(gold_standard, predicted):
    correct = sum(1 for gold, pred in zip(gold_standard, predicted) if gold == pred)
    total = len(gold_standard)
    incorrect = total - correct
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, incorrect / total if total > 0 else 0.0

ground = [x.split('/') for sentence in dev_sentences2 for x in sentence]
pred = [x.split('/') for sentence in dev_vit for x in sentence]

accuracy, error_rate = accuracy_calculate(ground, pred)
print(f"Accuracy for Viterbi Decoding using HMM: {accuracy * 100:.2f}%")
print(f"Error Rate: {error_rate * 100:.2f}%")

# %% [markdown]
# # Predicting on test.json

# %%
with open('data/test.json', 'r') as test_file:
    test_data = json.load(test_file)

test_sentences = [[word for word in test['sentence']] for test in test_data]

test_vit=viterbi(test_sentences, unique_label, emission, transition)

# %% [markdown]
# ### Creating viterbi.json

# %%
with open('data/test.json', 'r') as test_file:
    greedy_data = json.load(test_file)

for index, data in enumerate(greedy_data):
    data['labels'] = [test_vit[index][i].split('/')[1] for i in range(len(data['sentence']))]

output_file = 'viterbi.json'
with open(output_file, "w") as outfile:
    json.dump(greedy_data, outfile, indent=4)        


