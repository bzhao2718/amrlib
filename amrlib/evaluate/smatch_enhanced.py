import re
import logging
from collections import defaultdict, OrderedDict
from multiprocessing import Pool
import smatch
from amr import AMR
from .smatch_reentracy_srl import compute_reentracy_srl

logger = logging.getLogger(__name__)


###################################################################################################
# Code to compute the smatch score using multithreading plus additional tests to score modified
# graphs for debugging specific perforance issues (ie.. scoring with and without wiki tags)
#
# Code from From https://github.com/ChunchuanLv/amr-evaluation-tool-enhanced
# which pulls from https://github.com/mdtux89/amr-evaluation
# Smatch library at https://github.com/snowblink14/smatch
#
# Smatch Subscore definitions:
# Unlabeled:          Compute on the predicted graphs after removing all edge labels
# No WSD.             Compute while ignoring Propbank senses (e.g., duck-01 vs duck-02)
# Non_sense_frames    F-score on Propbank frame identification without sense (e.g. duck-00)
# Wikification        F-score on the wikification (:wiki roles)
# Named Ent.          F-score on the named entity recognition (:name roles)
# Negations           F-score on the negation detection (:polarity roles)
# IgnoreVars          Computed by replacing variables with their concepts
# Concepts            F-score on the concept identification task
# Frames              F-score on Propbank frame identification without sense (e.g. duck-01)
# Reentrancy          Computed on reentrant edges only
# SRL                 Computed on :ARG-i roles only
###################################################################################################


###############################################################################
#### Public functions for computing smatch
###############################################################################

# Score a list of entry pairs
# The entries should be a list of single line strings.
def compute_smatch(test_entries, gold_entries):
    pairs = zip(test_entries, gold_entries)
    mum_match = mum_test = mum_gold = 0
    pool = Pool()
    for (n1, n2, n3) in pool.imap_unordered(match_pair, pairs):
        mum_match += n1
        mum_test += n2
        mum_gold += n3
    pool.close()
    pool.join()
    precision, recall, f_score = smatch.compute_f(mum_match, mum_test, mum_gold)
    return precision, recall, f_score


def compute_scores(test_fn, gold_fn):
    # Get the graph from each entry in each file
    test_entries = get_entries(test_fn)
    gold_entries = get_entries(gold_fn)
    assert len(test_entries) == len(gold_entries), '%d != %d' % (len(test_entries), len(gold_entries))
    # Compute standard smatch scores
    precision, recall, f_score = compute_smatch(test_entries, gold_entries)
    output_score('Smatch', precision, recall, f_score)
    # Compute unlabeled data
    tes = [unlabel(e) for e in test_entries]
    ges = [unlabel(e) for e in gold_entries]
    precision, recall, f_score = compute_smatch(tes, ges)
    output_score('Unlabeled', precision, recall, f_score)
    # Compute withough Word Sense Disambiguation
    tes = [remove_wsd(e) for e in test_entries]
    ges = [remove_wsd(e) for e in gold_entries]
    precision, recall, f_score = compute_smatch(tes, ges)
    output_score('No WSD', precision, recall, f_score)
    # get the other misc sub-scores
    score_dict = compute_subscores(test_entries, gold_entries)
    for stype, (pr, rc, f) in score_dict.items():
        output_score(stype, pr, rc, f)
    # Get the Reentracies and SRL scores
    score_dict = compute_reentracy_srl(test_entries, gold_entries)
    for stype, (pr, rc, f) in score_dict.items():
        output_score(stype, pr, rc, f)


# Read in an AMR file and return the graph as a string
def get_entries(fname):
    with open(fname) as f:
        data = f.read()
    entries = []
    for e in data.split('\n\n'):
        lines = [l.strip() for l in e.splitlines()]
        lines = [l for l in lines if (l and not l.startswith('#'))]
        string = ' '.join(lines)
        string = string.replace('\t', ' ')  # replace tabs with a space
        string = re.sub(' +', ' ', string)  # squeeze multiple spaces into a single
        if string:
            entries.append(string)
    return entries


###############################################################################
#### Internally used functions for computation, etc..
###############################################################################

def output_score(stype, precision, recall, f_score):
    print('%-16s -> P: %.3f,  R: %.3f,  F: %.3f' % (stype, precision, recall, f_score))


# Process a single pair (function separated for multiprocessing)
def match_pair(pair):
    amr1, amr2 = pair
    smatch.match_triple_dict.clear()  # clear the matching triple dictionary
    try:
        ret = smatch.get_amr_match(amr1, amr2)
        return ret
    except:
        return 0, 0, 0


# Returns a dictionary of variables to concepts
def var2concept(amr):
    v2c = {}
    for n, v in zip(amr.nodes, amr.node_values):
        v2c[n] = v
    return v2c


# Modify graphs and compute scores on the variations
def compute_subscores(pred, gold):
    inters = defaultdict(int)
    golds = defaultdict(int)
    preds = defaultdict(int)
    # Loop through all entries
    for amr_pred, amr_gold in zip(pred, gold):
        # Create the predicted data
        amr_pred = AMR.parse_AMR_line(amr_pred.replace("\n", ""))
        if amr_pred is None:
            logger.error('Empty amr_pred entry')
            continue
        dict_pred = var2concept(amr_pred)
        triples_pred = [t for t in amr_pred.get_triples()[1]]
        triples_pred.extend([t for t in amr_pred.get_triples()[2]])
        # Create the gold data
        amr_gold = AMR.parse_AMR_line(amr_gold.replace("\n", ""))
        if amr_gold is None:
            logger.error('Empty amr_gold entry')
            continue
        dict_gold = var2concept(amr_gold)
        triples_gold = [t for t in amr_gold.get_triples()[1]]
        triples_gold.extend([t for t in amr_gold.get_triples()[2]])
        # Non_sense_frames scores
        list_pred = non_sense_frames(dict_pred)
        list_gold = non_sense_frames(dict_gold)
        inters["Non_sense_frames"] += len(list(set(list_pred) & set(list_gold)))
        preds["Non_sense_frames"] += len(set(list_pred))
        golds["Non_sense_frames"] += len(set(list_gold))
        # Wikification scores
        list_pred = wikification(triples_pred)
        list_gold = wikification(triples_gold)
        inters["Wikification"] += len(list(set(list_pred) & set(list_gold)))
        preds["Wikification"] += len(set(list_pred))
        golds["Wikification"] += len(set(list_gold))
        # Named entity scores
        list_pred = namedent(dict_pred, triples_pred)
        list_gold = namedent(dict_gold, triples_gold)
        inters["Named Ent."] += len(list(set(list_pred) & set(list_gold)))
        preds["Named Ent."] += len(set(list_pred))
        golds["Named Ent."] += len(set(list_gold))
        # Negation scores
        list_pred = negations(dict_pred, triples_pred)
        list_gold = negations(dict_gold, triples_gold)
        inters["Negations"] += len(list(set(list_pred) & set(list_gold)))
        preds["Negations"] += len(set(list_pred))
        golds["Negations"] += len(set(list_gold))
        # Ignore Vars scores
        list_pred = everything(dict_pred, triples_pred)
        list_gold = everything(dict_gold, triples_gold)
        inters["IgnoreVars"] += len(list(set(list_pred) & set(list_gold)))
        preds["IgnoreVars"] += len(set(list_pred))
        golds["IgnoreVars"] += len(set(list_gold))
        # Concepts scores
        list_pred = concepts(dict_pred)
        list_gold = concepts(dict_gold)
        inters["Concepts"] += len(list(set(list_pred) & set(list_gold)))
        preds["Concepts"] += len(set(list_pred))
        golds["Concepts"] += len(set(list_gold))
        # Frames scores
        list_pred = frames(dict_pred)
        list_gold = frames(dict_gold)
        inters["Frames"] += len(list(set(list_pred) & set(list_gold)))
        preds["Frames"] += len(set(list_pred))
        golds["Frames"] += len(set(list_gold))
    # Create the return dictionary
    rdict = OrderedDict()
    for score in preds:
        pr = 0 if preds[score] <= 0 else inters[score] / float(preds[score])
        rc = 0 if golds[score] <= 0 else inters[score] / float(golds[score])
        f = 0 if pr + rc <= 0 else 2 * (pr * rc) / (pr + rc)
        rdict[score] = (pr, rc, f)
    return rdict


###############################################################################
#### Internally used graph / string manipulation
###############################################################################

# Unlabel data
match_of = re.compile(r":[0-9a-zA-Z]*-of")
match_no_of = re.compile(r":[0-9a-zA-Z]*(?!-of)")


def unlabel(amr):
    amr = re.sub(match_no_of, ":label", amr)
    amr = re.sub(match_of, ":label-of", amr)
    return amr


# Remove Word Sense Disambiguation (ie.. -01) from words
match_wsd = re.compile(r'(\/ [a-zA-Z0-9\-][a-zA-Z0-9\-]*)-[0-9][0-9]*')


def remove_wsd(text):
    return re.sub(match_wsd, r'\1-01', text)


RE_FRAME_NUM = re.compile(r'-\d\d$')


def concepts(v2c_dict):
    return [str(v) for v in list(v2c_dict.values())]


def frames(v2c_dict):
    return [str(v) for v in list(v2c_dict.values()) if RE_FRAME_NUM.search(v) is not None]


def non_sense_frames(v2c_dict):
    return [re.sub(RE_FRAME_NUM, '', str(v)) for v in list(v2c_dict.values()) if \
            RE_FRAME_NUM.search(v) is not None]


def namedent(v2c_dict, triples):
    return [str(v2c_dict[v1]) for (l, v1, v2) in triples if l == "name"]


def negations(v2c_dict, triples):
    return [v2c_dict[v1] for (l, v1, v2) in triples if l == "polarity"]


def wikification(triples):
    return [v2 for (l, v1, v2) in triples if l == "wiki"]


def everything(v2c_dict, triples):
    lst = []
    for t in triples:
        c1 = t[1]
        c2 = t[2]
        if t[1] in v2c_dict:
            c1 = v2c_dict[t[1]]
        if t[2] in v2c_dict:
            c2 = v2c_dict[t[2]]
        lst.append((t[0], c1, c2))
    return lst


####### modified functions

def clean_entry(entry):
    if entry:
        if isinstance(entry, float):
            print(entry)
            return ''
        lines = [l.strip() for l in entry.splitlines()]
        lines = [l for l in lines if (l and not l.startswith('#'))]
        string = ' '.join(lines)
        string = string.replace('\t', ' ')  # replace tabs with a space
        string = re.sub(' +', ' ', string)  # squeeze multiple spaces into a single
        return string


def get_splitted_entry(sents):
    if sents and isinstance(sents, str):
        sent_list = sents.split('\n\n')
        if sent_list:
            entry_list = []
            for sent in sent_list:
                if sent:
                    entry_list.append(clean_entry(sent))
            return entry_list


def get_entries_sent_splitted(df, col_cand='cand_amr', col_ref='ref_amr'):
    cand_entries = []
    ref_entries = []
    for cand, ref in zip(df[col_cand], df[col_ref]):
        cand_list = get_splitted_entry(sents=cand)
        ref_list = get_splitted_entry(sents=ref)
        cand_entries.append(cand_list)
        ref_entries.append(ref_list)
    return cand_entries, ref_entries


# Read in an AMR file and return the graph as a string
def get_cand_ref_entries_from_df(df, col_cand='cand_amr', col_ref='ref_amr', sent_splitted=False):
    cand_entries = []
    ref_entries = []
    if sent_splitted:  # each entry is a list of sentences
        return get_entries_sent_splitted(df, col_cand=col_cand, col_ref=col_ref)

    for cand, ref in zip(df[col_cand], df[col_ref]):
        cand_entry = clean_entry(cand)
        ref_entry = clean_entry(ref)
        # if cand_entry and ref_entry:
        cand_entries.append(cand_entry)
        ref_entries.append(ref_entry)
    return cand_entries, ref_entries


def smatch_scores_single_str(cands, refs):
    """
    return a list of smatch scores for each cand, ref entry
    """
    assert len(cands) == len(refs), '%d != %d' % (len(cands), len(refs))
    p_list = []
    r_list = []
    f_list = []
    for cand, ref in zip(cands, refs):
        if cand and ref and isinstance(cand, str) and isinstance(ref, str):
            precision, recall, f_score = compute_smatch([cand], [ref])
        else:
            precision, recall, f_score = 0.0, 0.0, 0.0
        p_list.append(precision)
        r_list.append(recall)
        f_list.append(f_score)
    return p_list, r_list, f_list


def smatch_scores_splitted(cands, refs):
    """
    return a list of smatch scores for each cand, ref entry
    the each cand and ref is a list of sentences so we compare each cand sentence with ref sentences and pick the max smatch scores
    the final score is the average: (s1+s2+...+sn)/n, where s_i is the smatch score for each individual sentence in a cand summary
    """
    assert len(cands) == len(refs), '%d != %d' % (len(cands), len(refs))
    p_list = []
    r_list = []
    f_list = []
    idx = 0
    print_every = 50
    for cand, ref in zip(cands, refs):
        if cand and ref and not isinstance(cand, float):
            precision, recall, f_score = single_smatch_score_splitted(cand, ref)
        else:
            precision, recall, f_score = 0.0, 0.0, 0.0
        p_list.append(precision)
        r_list.append(recall)
        f_list.append(f_score)
        if idx % print_every == 0:
            print(f'idx: {idx}')
        idx += 1
    return p_list, r_list, f_list


def single_smatch_score_splitted(cand_list, ref_list):
    p_scores = []
    r_scores = []
    f_scores = []
    for cand in cand_list:
        if cand and not isinstance(cand, float):
            p_max, r_max, f_max = 0.0, 0.0, 0.0
            p_list, r_list, f_list = [], [], []
            # cands = [cand] * len(ref_list)  # compare each individual cand amr against ref amrs and pick the max score
            # p, r, f = compute_smatch(cands, ref_list)
            # p_scores.append(max(p))
            # r_scores.append(max(r))
            # f_scores.append(max(f))
            for ref in ref_list:
                if ref:
                    p, r, f = compute_smatch([cand], [ref])
                    p_list.append(p)
                    r_list.append(r)
                    f_list.append(f)
                    # p_max = p if p > p_max else p_max
                    # r_max = r if r > r_max else r_max
                    # f_max = f if f > f_max else f_max
            p_scores.append(max(p_list))
            r_scores.append(max(r_list))
            f_scores.append(max(f_list))
    s_p = sum(p_scores) / len(p_scores) if p_scores else 0.0
    s_r = sum(r_scores) / len(r_scores) if r_scores else 0.0
    s_f = sum(f_scores) / len(f_scores) if f_scores else 0.0
    return s_p, s_r, s_f


def smatch_scores_from_entries(test_entries, gold_entries):
    assert len(test_entries) == len(gold_entries), '%d != %d' % (len(test_entries), len(gold_entries))
    # Compute standard smatch scores
    precision, recall, f_score = compute_smatch(test_entries, gold_entries)
    output_score('Smatch', precision, recall, f_score)
    # Compute unlabeled data
    tes = [unlabel(e) for e in test_entries]
    ges = [unlabel(e) for e in gold_entries]
    precision, recall, f_score = compute_smatch(tes, ges)
    output_score('Unlabeled', precision, recall, f_score)
    # Compute withough Word Sense Disambiguation
    tes = [remove_wsd(e) for e in test_entries]
    ges = [remove_wsd(e) for e in gold_entries]
    precision, recall, f_score = compute_smatch(tes, ges)
    output_score('No WSD', precision, recall, f_score)
    # get the other misc sub-scores
    score_dict = compute_subscores(test_entries, gold_entries)
    for stype, (pr, rc, f) in score_dict.items():
        output_score(stype, pr, rc, f)
    # Get the Reentracies and SRL scores
    score_dict = compute_reentracy_srl(test_entries, gold_entries)
    for stype, (pr, rc, f) in score_dict.items():
        output_score(stype, pr, rc, f)
