import json
import numpy as np


### ---- SCORING FUNCTIONS ---- ###

def bert_scores(dataset):
    raise NotImplementedError

def rarity_scores(dataset, vocab):
    """ Less likely (more rare) sentences are more difficult.
    @returns rarity_scores: scores to be ranked by rank_scores
    """
    train_data, dev_data = dataset
    src_vocab, tgt_vocab = vocab.src, vocab.tgt

    train_scores = []
    for (src_sent, tgt_sent) in train_data:
        src_word_scores, src_sent_score = src_vocab.words2rarity(src_sent)
        tgt_word_scores, tgt_sent_score = tgt_vocab.words2rarity(tgt_sent)
        train_scores.append((src_sent_score, tgt_sent_score))
    dev_scores = []
    for (src_sent, tgt_sent) in dev_data:
        src_word_scores, src_sent_score = src_vocab.words2rarity(src_sent)
        tgt_word_scores, tgt_sent_score = tgt_vocab.words2rarity(tgt_sent)
        dev_scores.append((src_sent_score, tgt_sent_score))
    
    return (train_scores, dev_scores)

def length_scores(dataset):
    raise NotImplementedError

### -------------------------- ###

def get_difficulty_scores(order_name, dataset, vocab):
    """ Returns the difficulty scores given a specificied ordering function.
    @returns difficulty_scores
    """
    if order_name == "bert": 
        raise NotImplementedError
    elif order_name == "rarity":
        scores = rarity_scores(dataset, vocab)
    elif order_name == "length":
        raise NotImplementedError
    else:
        raise ValueError("Order name {} is not supported!".format(order_name))
    return scores

def rank_scores(difficulty_scores):
    """ Ranks the sentences of a dataset according to the difficulty scores.
    @param difficulty_scores
    @returns order_ids (list) a list of ints specifying the ordering 
    """
    # TODO: for now, the difficulty is for the *src* sentence only
    train_scores, dev_scores = difficulty_scores
    train_scores = [src_tgt_pair[0] for src_tgt_pair in train_scores]
    dev_scores = [src_tgt_pair[0] for src_tgt_pair in dev_scores]
    train_order_ids = sorted(range(len(train_scores)), key=train_scores.__getitem__)
    dev_order_ids = sorted(range(len(dev_scores)), key=dev_scores.__getitem__)
    return (train_order_ids, dev_order_ids)

def load_order(order_name, dataset, vocab):
    """ Takes in a dataset and returns an ordering (ranking) 
    based on a scoring (difficulty) function.
    @param order_name (str): the scoring function to use
    @param dataset (tuple): a tuple of lists (train_data, dev_data)
    @param vocab (Vocab): a Vocab object containing (vocab.src, vocab.tgt)
    @returns order (Dataset): an ordered dataset of (train_data, dev_data) 
        in increasing order (easy difficulty to hard difficulty) 
    """
    orderings = ["none", "bert", "rarity", "length"]
    print("order name is:", order_name)
    if order_name not in orderings:
        raise ValueError("Order name {} is not supported!".format(order_name))

    if order_name == "none":
        return dataset

    difficulty_scores = get_difficulty_scores(order_name, dataset, vocab)
    order_ids = rank_scores(difficulty_scores)

    # Sort the dataset according to the ordering
    train_order_ids, dev_order_ids = order_ids
    train_data, dev_data = dataset
    ordered_train = [train_data[i] for i in train_order_ids]
    ordered_dev = [dev_data[i] for i in dev_order_ids]
    return (ordered_train, ordered_dev)
    
def balance_order(order, dataset):
    """ Ensures that the ordered dataset according to difficulty 
    is balanced, i.e. each class has equal number of samples.
    """
    pass

def visualize_scoring(ordered_dataset, vocab):
    """ Visualizes difficulty of sentences.
    @param ordered_dataset (Dataset): dataset of (train_data, dev_data)
    @param vocab (Vocab): Vocabulary with vocab.src and vocab.tgt
    """
    train_data, dev_data = ordered_dataset
    src_vocab, tgt_vocab = vocab.src, vocab.tgt
    print("Creating scoring visualizations.")
    num_easy = 1
    skip = 30
    num_medium = 1
    num_hard = 1
    
    easy_examples = train_data[skip:skip+num_easy]
    medium_examples = train_data[skip + int(len(train_data) / 2):skip + int(len(train_data) / 2) + num_medium]
    hard_examples = train_data[-num_hard:]
    
    print("\n***** [ Easy examples ] *****")
    print("Source:\n", " ".join(easy_examples[0][0]))
    print("Target:\n", " ".join(easy_examples[0][1]))
    src_word_scores, src_sent_score = src_vocab.words2rarity(easy_examples[0][0])
    tgt_word_scores, tgt_sent_score = tgt_vocab.words2rarity(easy_examples[0][1])
    print("src_word_scores:", src_word_scores)
    print("src_sent_score:", src_sent_score)
    print("tgt_word_scores:", tgt_word_scores)
    print("tgt_sent_score:", tgt_sent_score)

    print("\n***** [ Medium examples ] *****")
    print("Source:\n", " ".join(medium_examples[0][0]))
    print("Target:\n", " ".join(medium_examples[0][1]))
    src_word_scores, src_sent_score = src_vocab.words2rarity(medium_examples[0][0])
    tgt_word_scores, tgt_sent_score = tgt_vocab.words2rarity(medium_examples[0][1])
    print("src_word_scores:", src_word_scores)
    print("src_sent_score:", src_sent_score)
    print("tgt_word_scores:", tgt_word_scores)
    print("tgt_sent_score:", tgt_sent_score)

    print("\n***** [ Hard examples ] *****")
    print("Source:\n", " ".join(hard_examples[0][0]))
    print("Target:\n", " ".join(hard_examples[0][1]))
    src_word_scores, src_sent_score = src_vocab.words2rarity(hard_examples[0][0])
    tgt_word_scores, tgt_sent_score = tgt_vocab.words2rarity(hard_examples[0][1])
    print("src_word_scores:", src_word_scores)
    print("src_sent_score:", src_sent_score)
    print("tgt_word_scores:", tgt_word_scores)
    print("tgt_sent_score:", tgt_sent_score)
    
    

