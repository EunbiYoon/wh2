from utils import load_training_set, load_test_set
from itertools import chain
import pandas as pd
import openpyxl
from collections import Counter


# Prioir Probability :: pr(yi) = N(yi)/N
def prior_prob(pos_data_len, neg_data_len):
    prior_pos = pos_data_len / ( pos_data_len + neg_data_len )
    prior_neg = neg_data_len / ( pos_data_len + neg_data_len )
    return prior_pos, prior_neg
        
# find unique word in test dataset
def find_unique_words(nested_list):
    word_counts = {}
    # 모든 단어의 빈도수 계산
    for sublist in nested_list:
        for word in sublist:
            word_counts[word] = word_counts.get(word, 0) + 1
    # 빈도수가 1인 단어만 필터링
    unique_words = [word for word, count in word_counts.items() if count == 1]
    return unique_words

# find matching index between condition_table index and test dataset
def find_matching_indices(vocab, test_unique):
    pos_test_index = []
    
    for word in test_unique:
        if word in vocab:
            pos_test_index.append(vocab.index(word))  # 해당 단어의 인덱스를 저장
    
    return pos_test_index


# Conditioinal Probability :: pr(wk|yi) = n(wk, yi)/ sigma (n(ws, yi)) -> s = 1 ~ |V|
def condition_prob(vocab, pos_train, neg_train, test_data):
    ###### train ######
    # make condition_table
    condition_table=pd.DataFrame(index=range(len(vocab)), columns=["pos_frequency","neg_frequency"])

    # assign all the vocab in pos and neg with value 0
    pos_word_counts = {word: 0 for word in vocab}
    neg_word_counts = {word: 0 for word in vocab}

    # calcualte pos_train word frequency
    for review in pos_train:
        for word in review:
            if word in pos_word_counts:
                pos_word_counts[word] += 1

    # calcualte neg_train word frequency
    for review in neg_train:
        for word in review:
            if word in neg_word_counts:
                neg_word_counts[word] += 1

    # fill out the value in condition table
    for i, word in enumerate(vocab):
        condition_table.at[i, "pos_frequency"] = pos_word_counts[word]
        condition_table.at[i, "neg_frequency"] = neg_word_counts[word]
    
    # add sum end of table
    sum_row=condition_table.sum()
    condition_table.loc['Sum']=sum_row

    # get the portion 
    condition_table["pos_portion"] = condition_table["pos_frequency"] / sum_row.iloc[0]
    condition_table["neg_portion"] = condition_table["neg_frequency"] / sum_row.iloc[1]


    ###### test ######
    # find unique word from test set
    test_unique=find_unique_words(test_data)

    # get the index matching with condition_table index
    test_index=find_matching_indices(vocab, test_unique)

    # get the portion from condition_table
    test_condition_table=condition_table.iloc[test_index]

    # multiply pos_portion and neg_portion
    multiply_row=test_condition_table.prod()
    test_condition_table.loc['Multiply']=multiply_row


    ####### check result
    print(test_condition_table)

    return multiply_row["pos_portion"], multiply_row["neg_portion"]



# Standard Multinomial Naive Bayes
def multinomial_standard(prior_pos, prior_neg, condition_pos, condition_neg):
    posterior_pos = prior_pos * condition_pos
    posterior_neg = prior_neg * condition_neg
    if posterior_pos>posterior_neg:
        posterior_result="Positive"
        return posterior_result
    elif posterior_pos<posterior_neg:
        posterior_result="Negative"
        return posterior_result
    elif posterior_pos==posterior_neg:
        if posterior_pos==0:
            print("-> Error :: Multiply 0 made 0")
        else:
            print("-> Error :: Positive and Negative Have Same Posterior Probability")

                                                                                                                            
# main function
if __name__ == '__main__':
    ### preprocessign data
    percentage_positive_instances_train = 0.02
    percentage_negative_instances_train = 0.02
    percentage_positive_instances_test = 0.02
    percentage_negative_instances_test = 0.02
    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
    print("\nTrain and Test dataset are completed to preprocessing data\n")


    ### Question 1 : standard Multinominal Naive Bayes
    print("[Question 1] Standard Multinomial Naive Bayes with both 20 % Train and Test Dataset\n")
    prior_pos_train,prior_neg_train=prior_prob(len(pos_train), len(neg_train)) # prior probability
    # test positive dataset
    condition_pos_test, condition_neg_test=condition_prob(list(vocab), pos_train, neg_train, pos_test) # conditional probability 
    q1_test_pos=multinomial_standard(prior_pos_train, prior_neg_train, condition_pos_test, condition_neg_test) # posterior probability
    print(f"=> Test Positive Dataset Predicted to '{q1_test_pos}' class\n")
    # test negative dataset
    condition_pos_test, condition_neg_test=condition_prob(list(vocab), pos_train, neg_train, neg_test) # conditional probability 
    q1_test_neg=multinomial_standard(prior_pos_train, prior_neg_train, condition_pos_test, condition_neg_test) # posterior probability
    print(f"=> Test Negative Dataset Predicted to '{q1_test_pos}' class\n")
     