import cPickle as pickle
import os
import numpy as np
from utils import read_json_lines, load_json, save_json, save_pickle
from tqdm import tqdm

def line_to_words(line, eos=True, downcase=True):
    eos_word = "<eos>"
    words = line.lower().split() if downcase else line.split()
    # !!!! remove comma here, since they are too many of them
    words = [w for w in words]
    words = words + [eos_word] if eos else words
    return words

glove_path = "./glove.840B.300d.txt"
word2idx_path = "."
idx2word_path = '.'
raw_train_path = "./tvqa_train_processed.json"
raw_val_path = "./tvqa_val_processed.json"
raw_test_path = "./tvqa_test_public_processed.json"

word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
idx2word = {0: "<pad>", 1: "<unk>", 2: "<eos>"}
offset = len(word2idx)
text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "located_sub_text"]
all_sentences = []
raw_train = load_json(raw_train_path)
for k in text_keys:
    all_sentences.extend(ele[k] for ele in raw_train)
raw_val = load_json(raw_val_path)
for k in text_keys:
    all_sentences.extend(ele[k] for ele in raw_val)
raw_test = load_json(raw_test_path)
for k in text_keys:
    all_sentences.extend(ele[k] for ele in raw_test)

word_counts = {}
for sentence in all_sentences:
    for w in line_to_words(sentence, eos=False):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= 1 and w not in word2idx.keys()]
print("Vocabulary Size %d (<pad> <unk> <eos> excluded) using word_count_threshold %d.\n" % (len(vocab), 1))

for idx, w in enumerate(vocab):
    word2idx[w] = idx + offset
    idx2word[idx + offset] = w
print("word2idx size: %d, idx2word size: %d.\n" % (len(word2idx), len(idx2word)))

glove_model = {}
with open(glove_path, 'r') as glove:
    print "Loading Glove Model"
    for line in glove:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        glove_model[word] = embedding
print("Glove Loaded, building word2idx, idx2word mapping. This may take a while.\n")

glove_matrix = np.zeros([len(idx2word), 300])
glove_keys = glove_model.keys()
k = 0
for i in tqdm(range(len(idx2word))):
    w = idx2word[i]
    if w in glove_keys:
        w_embed = glove_model[w]
    else:
        w_embed = np.random.randn(300) * 0.4
        k += 1
    glove_matrix[i, :] = w_embed
print str(k) + "words do not have word embedding"
vocab_embedding = glove_matrix
print("Vocab embedding size is :", glove_matrix.shape)
save_pickle(word2idx, "./word2idx.pkl")
save_pickle(idx2word, "./idx2word.pkl")
save_pickle(glove_matrix, "./vocab_embedding_matrix.pkl")