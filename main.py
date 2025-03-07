from utils import load_training_set, load_test_set
from itertools import chain
import pandas as pd
import openpyxl
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


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
    test_index = []
    
    for word in test_unique:
        if word in vocab:
            test_index.append(vocab.index(word))  # 해당 단어의 인덱스를 저장

    return test_index

def stand_portion(condition_table, sum_row):
     # get the portion 
    condition_table["pos_portion"] = condition_table["pos_frequency"] / sum_row.iloc[0]
    condition_table["neg_portion"] = condition_table["neg_frequency"] / sum_row.iloc[1]
    return condition_table


# Conditioinal Probability :: pr(wk|yi) = n(wk, yi)/ sigma (n(ws, yi)) -> s = 1 ~ |V|
def condition_prob(pos_train, neg_train):
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
    return condition_table


# compare final two class's posterior value 
def compare_posterior(posterior_pos,posterior_neg):
    # compare bewtween two class
    if posterior_pos>posterior_neg:
        posterior_result="Positive"
        return posterior_result
    elif posterior_pos<posterior_neg:
        posterior_result="Negative"
        return posterior_result
    elif posterior_pos==posterior_neg:
        if posterior_pos==0:
            print("-> Error :: Multiply 0 made 0")
            return None
        else:
            print("-> Error :: Positive and Negative Have Same Posterior Probability")
            return None


# Standard Multinomial Naive Bayes
def multinomial_standard(prior_pos, prior_neg, condition_table, vocab, test_data):
    # get the portion 
    condition_table["pos_portion"] = condition_table["pos_frequency"] / condition_table.at["Sum","pos_frequency"]
    condition_table["neg_portion"] = condition_table["neg_frequency"] / condition_table.at["Sum","neg_frequency"]
    
    ##### sort with test dataset 
    # find unique word from test set
    test_unique=find_unique_words(test_data)
    # get the index matching with condition_table index
    test_index=find_matching_indices(vocab, test_unique)
    # get the portion from condition_table
    print(condition_table)
    condition_table=condition_table.iloc[test_index]
    print(condition_table)
    #################################################

    # multiply pos_portion and neg_portion
    multiply_row=condition_table.prod()
    condition_table.loc['Multiply']=multiply_row
    condition_pos=multiply_row["pos_portion"]
    condition_neg=multiply_row["neg_portion"]
    print(condition_table)

    # multiply prior probability to final value of posterior probability
    posterior_pos = prior_pos * condition_pos
    posterior_neg = prior_neg * condition_neg

    # compare posterior probability between class
    predicted_class=compare_posterior(posterior_pos, posterior_neg)
    return predicted_class


# Laplace Smoothing log(prior probability) + sum (log(conditional))
def laplace_smoothing(prior_pos, prior_neg, condition_table, vocab, test_data, alpha):
    # if the probability is 0, numerator+alpha & denominator+alpha*(len(vocab))
    condition_table['pos_frequency_alpha'] = condition_table['pos_frequency'].apply(lambda x: alpha if x == 0 else x)
    condition_table['neg_frequency_alpha'] = condition_table['neg_frequency'].apply(lambda x: alpha if x == 0 else x)
    condition_table.at['Sum','pos_frequency_alpha'] = condition_table.at['Sum','pos_frequency']+alpha*(len(vocab))
    condition_table.at['Sum','neg_frequency_alpha'] = condition_table.at['Sum','neg_frequency']+alpha*(len(vocab))

    # get the portion 
    condition_table["pos_portion_log"] = condition_table["pos_frequency_alpha"] / condition_table.at["Sum","pos_frequency_alpha"]
    condition_table["neg_portion_log"] = condition_table["neg_frequency_alpha"] / condition_table.at["Sum","neg_frequency_alpha"]

    ##### sort with test dataset 
    # find unique word from test set
    test_unique=find_unique_words(test_data)
    # get the index matching with condition_table index
    test_index=find_matching_indices(vocab, test_unique)
    # get the portion from condition_table
    condition_table=condition_table.iloc[test_index]
    #################################################

    # get log summation of both portion
    condition_log_pos = np.sum(condition_table['pos_portion_log'].apply(np.log))
    condition_log_neg = np.sum(condition_table['neg_portion_log'].apply(np.log))
    print(condition_table)

    # add with prior probability for final value of posterior probability
    posterior_pos=np.log(prior_pos)+condition_log_pos
    posterior_neg=np.log(prior_neg)+condition_log_neg
    
    # compare posterior probability between class
    predicted_class=compare_posterior(posterior_pos, posterior_neg)
    return predicted_class

# make graph for Question2
def draw_graph(alpha_list, accuracy_list):
    log_alpha_list = np.log(alpha_list) 
    plt.figure(figsize=(8, 6))
    plt.plot(log_alpha_list, accuracy_list, marker='o', linestyle='-', label='Log Scale Plot')
    plt.xlabel('X values (log scale)')
    plt.ylabel('Y values')
    plt.title('Log-Log Plot')
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig('accuracy_graph.png')

# select alpha value
def select_alpha(alpha_list, accuracy_list):
    # find max_index in accuracy_list
    max_index = np.argmax(accuracy_list)

    # find alpha value located in max_index
    best_alpha = alpha_list[max_index]

    return best_alpha

# proprocessing data
def preprocessing_data(train_percen_pos, train_percen_neg, test_percen_pos, test_percen_neg):
    percentage_positive_instances_train = train_percen_pos
    percentage_negative_instances_train = train_percen_neg
    percentage_positive_instances_test = test_percen_pos
    percentage_negative_instances_test = test_percen_neg
    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
    print("\nTrain and Test dataset are completed to preprocessing data\n")
    return pos_train, neg_train, vocab, pos_test, neg_test
                                                                                                            
# main function
if __name__ == '__main__':
    ### Question 1 : Standard Multinominal Naive Bayes
    print("[Question 1] Standard Multinomial Naive Bayes with both 20 % Train and Test Dataset\n")
    ### preprocessign data
    pos_train, neg_train, vocab, pos_test, neg_test=preprocessing_data(0.2, 0.2, 0.2, 0.2)
    # prior probability
    prior_pos_train,prior_neg_train=prior_prob(len(pos_train), len(neg_train)) 
    # condition probability
    condition_table=condition_prob(pos_train, neg_train)
    # posterior probability
    q1_test_pos=multinomial_standard(prior_pos_train, prior_neg_train, condition_table, list(vocab), pos_test)
    q1_test_neg=multinomial_standard(prior_pos_train, prior_neg_train, condition_table, list(vocab), neg_test) 
    # message
    print(f"=> Test Positive Dataset Predicted to '{q1_test_pos}' class")
    print(f"=> Test Negative Dataset Predicted to '{q1_test_pos}' class\n")
    

    ### Question 2 : Laplace Smoothing alpha = 0.0001 ~ 1000
    print("\n[Question 2] Apply Laplace Smoothing with both 20 % Train and Test Dataset\n")
    # posterior probability
    alpha_list=[0.001, 0.01, 1, 10, 100, 1000]
    accuracy_list=list()
    for i in range(len(alpha_list)):
        accuracy=0
        q2_test_pos=laplace_smoothing(prior_pos_train, prior_neg_train, condition_table, list(vocab), pos_test, i) 
        q2_test_neg=laplace_smoothing(prior_pos_train, prior_neg_train, condition_table, list(vocab), neg_test, i) 
        if q2_test_pos=="Positive":
            accuracy=accuracy+0.5
        if q2_test_neg=="Negative":
            accuracy=accuracy+0.5
        accuracy_list.append(accuracy)
    # message
    print(f"=> Test Positive Dataset Predicted to '{q2_test_pos}' class")
    print(f"=> Test Negative Dataset Predicted to '{q2_test_neg}' class")
    print(f"==> Accuracy : {accuracy_list}")
    # make graph
    draw_graph(alpha_list,accuracy_list)


    ### Question 3 : use all training and test dataset with accuracy maximize alpha
    print("\n[Question 3] Apply Laplace Smoothing with both 100 % Train and Test Dataset\n")
    ### preprocessign data
    pos_train, neg_train, vocab, pos_test, neg_test=preprocessing_data(1, 1, 1, 1)
    # select alpha maximize accuracy
    best_alpha=select_alpha(alpha_list,accuracy_list)
    # prior probability
    prior_pos_train,prior_neg_train=prior_prob(len(pos_train), len(neg_train)) 
    # condition probability
    condition_table=condition_prob(pos_train, neg_train)
    # get alpha value that maximizes accuracy
    best_alpha=select_alpha(alpha_list,accuracy_list)
    # posterior probability
    q3_test_pos=laplace_smoothing(prior_pos_train, prior_neg_train, condition_table, list(vocab), pos_test, best_alpha) 
    q3_test_neg=laplace_smoothing(prior_pos_train, prior_neg_train, condition_table, list(vocab), neg_test, best_alpha) 
    # message
    print(f"=> Test Positive Dataset Predicted to '{q3_test_pos}' class")
    print(f"=> Test Negative Dataset Predicted to '{q3_test_neg}' class")


    ### Question 4 : training 30% < test 100% dataset
    print("\n[Question 4] Apply Laplace Smoothing with 30% Train and 100% Test Dataset\n")
    ### preprocessign data
    pos_train, neg_train, vocab, pos_test, neg_test=preprocessing_data(0.3, 0.3, 1, 1)
    # prior probability
    prior_pos_train,prior_neg_train=prior_prob(len(pos_train), len(neg_train)) 
    # condition probability
    condition_table=condition_prob(pos_train, neg_train)
    # get alpha value that maximizes accuracy
    best_alpha=select_alpha(alpha_list,accuracy_list)
    # posterior probability
    q4_test_pos=laplace_smoothing(prior_pos_train, prior_neg_train, condition_table, list(vocab), pos_test, best_alpha) 
    q4_test_neg=laplace_smoothing(prior_pos_train, prior_neg_train, condition_table, list(vocab), neg_test, best_alpha) 
    # message
    print(f"=> Test Positive Dataset Predicted to '{q4_test_pos}' class")
    print(f"=> Test Negative Dataset Predicted to '{q4_test_neg}' class")


    ### Question 6 : unbalanced dataset => train(10%, 50%) / test 100%
    print("\n[Question 46 Apply Laplace Smoothing with Unbalanced 10%,50% Train and 100% Test Dataset\n")
    ### preprocessign data
    pos_train, neg_train, vocab, pos_test, neg_test=preprocessing_data(0.1, 0.5, 1, 1)
    # prior probability
    prior_pos_train,prior_neg_train=prior_prob(len(pos_train), len(neg_train)) 
    # condition probability
    condition_table=condition_prob(pos_train, neg_train)
    # get alpha value that maximizes accuracy
    best_alpha=select_alpha(alpha_list,accuracy_list)
    # posterior probability
    q6_test_pos=laplace_smoothing(prior_pos_train, prior_neg_train, condition_table, list(vocab), pos_test, best_alpha) 
    q6_test_neg=laplace_smoothing(prior_pos_train, prior_neg_train, condition_table, list(vocab), neg_test, best_alpha) 
    # message
    print(f"=> Test Positive Dataset Predicted to '{q6_test_pos}' class")
    print(f"=> Test Negative Dataset Predicted to '{q6_test_neg}' class")


    
