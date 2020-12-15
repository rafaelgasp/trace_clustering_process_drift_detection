import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
import itertools

def get_traces_as_tokens(traces_df, col_ref="Activity"):
    """
        Groups activities executions into traces
        as a string of tokens
        
        Ex:
             Activity |  Timestamp
            -----------------------
            START     | 2019-09-01
            A         | 2019-09-01
            B         | 2019-09-01
            C         | 2019-09-01
            END-A     | 2019-09-01
            
       into: "START A B C END-A"  
    """
    return traces_df.groupby("Trace_order")[col_ref].apply(
        lambda x: " ".join(x.values)
    )

def get_count_representation(tokens, binary=True, tfidf=False, ngram_range=(1, 1)):
    """
        Generic method to represent traces as vectors by counting
        activities or transitions.

        Parameters:
        ------------
            tokens (pd.Series): Trace represented as tokens (series of strings)
            binary (bool): Count binary or frequency
            tfidf (bool): Use tf-idf to normalize frequency
            ngram_range (tuple): Range of ngrams to obtain representation
    """
    
    if tfidf:
        cv = TfidfVectorizer(
            norm = None,
            smooth_idf = False,
            tokenizer=str.split, 
            lowercase=False,
            use_idf=True,
            ngram_range=ngram_range,
            min_df=0,
            max_df=1.0
        )
    else:
        cv = CountVectorizer(
            tokenizer=str.split, 
            lowercase=False,
            ngram_range=ngram_range,
            min_df=0,
            max_df=1.0,
            binary=binary
        )
    
    cv_result = cv.fit_transform(tokens)
    
    return pd.DataFrame(
        cv_result.todense(), 
        columns=cv.get_feature_names()
    )

# # # # # # # #
# Transitions #
# # # # # # # #
def get_binary_transitions_representation(tokens):    
    """
        Binary Transistions representation of traces
        (1 or 0 if the transition occur in traces)
    """
    return get_count_representation(tokens, True, False, (2,2))

def get_frequency_transitions_representation(tokens):    
    """
        Frequency Transistions representation of traces
        (# of occurences of a transition on the trace)
    """
    return get_count_representation(tokens, False, False, (2,2))

def get_tfidf_transitions_representation(tokens):    
    """
        TF-IDF Transistions representation of traces
        (frequency of the transition occur in traces 
        weighted by inverse document frequency)
    """
    return get_count_representation(tokens, False, True, (2,2))


# # # # # # 
# Activity #
# # # # # # 
def get_binary_representation(tokens):    
    """
        Binary representation of traces
        (1 or 0 if an activity occur in traces)
    """
    return get_count_representation(tokens, True, False, (1,1))

def get_frequency_representation(tokens):    
    """
        Frequency representation of traces
        (# of times an activity occur in traces)
    """
    return get_count_representation(tokens, False, False, (1,1))

def get_tfidf_representation(tokens):    
    """
        TF-IDF representation of traces
        (frequency of the occurence of activity in traces 
        weighted by inverse document frequency)
    """
    return get_count_representation(tokens, False, True, (1,1))


# # # # # # # # # #
# Extra Functions #
# # # # # # # # # #
def reinverse_tokens(tokens, inv_aliases, ret_string=True):
    """
        Invert aliases back to full activities names
    """
    r = []
    
    if isinstance(tokens, str):
        t = tokens.split()
    else:
        t = tokens
    
    for token in t:
        if token in inv_aliases:
            r.append(inv_aliases[token])
        else:
            r.append(token)
    
    if ret_string:
        return " ".join(r)
    
    return r