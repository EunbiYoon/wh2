import numpy as np
import pandas as pd
from itertools import chain
import time
import matplotlib.pyplot as plt
from utils import load_training_set, load_test_set



########### 0. Basic Calculation ###########
# log function
def log_func(num):
    return np.log10(num)

# Efficient unique word extraction from the test dataset using NumPy
def find_unique_words(nested_list):
    words = np.array(list(chain(*nested_list)))  # Flatten list and convert to NumPy array
    unique_words, counts = np.unique(words, return_counts=True)
    return unique_words[counts == 1]  # Return only unique words

# Efficient matching of indices between likelihood_table index and test dataset
def find_matching_indices(vocab, test_unique):
    vocab_set = set(vocab)  # Using set for faster lookup
    return np.array([i for i, word in enumerate(test_unique) if word in vocab_set])

# Vectorized portion calculation for the likelihood table
def stand_portion(likelihood_table, sum_row):
    pos_sum = sum_row["pos_frequency"]
    neg_sum = sum_row["neg_frequency"]
    likelihood_table["pos_portion"] = likelihood_table["pos_frequency"] / pos_sum
    likelihood_table["neg_portion"] = likelihood_table["neg_frequency"] / neg_sum
    return likelihood_table

# Compare probability
def compare_probability(posterior_pos, posterior_neg):
    if posterior_pos > posterior_neg:
        result="Positive"
    elif posterior_pos < posterior_neg:
        result="Negative"
    elif posterior_pos == posterior_neg:
        result=np.random.choice(["Positive", "Negative"])
    else:
        print("Error : compare_probability function")
    print(result)
    return result
                                

# draw table for Q1~Q6
def draw_table(confusion_table, accuracy, precision, recall, image_name):
    fig, ax = plt.subplots(figsize=(5, 3))  # Create figure and axes
    ax.axis('off')  # Hide axes

    # Display the table
    table = ax.table(cellText=confusion_table.values, 
                     colLabels=confusion_table.columns, 
                     rowLabels=confusion_table.index, 
                     cellLoc='center',  # Center-align cell text
                     loc='center')

    # Adjust table scale for better visibility
    table.scale(1.5, 1.5)

    # Add accuracy, recall, and precision above the table
    text_y = 1.2  # Adjust text position (1.2 means above the table)
    ax.text(0.5, text_y, f"Accuracy: {accuracy:.2f}", fontsize=12, ha='center', transform=ax.transAxes)
    ax.text(0.5, text_y - 0.1, f"Precision: {precision:.2f}", fontsize=12, ha='center', transform=ax.transAxes)
    ax.text(0.5, text_y - 0.2, f"Recall: {recall:.2f}", fontsize=12, ha='center', transform=ax.transAxes)

    # Add title to the table
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')

    # Adjust layout to fit content properly
    plt.tight_layout()

    plt.savefig(f"{image_name}.png")

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



########### 1. Preprocessing Data ###########
def preprocessing_data(train_percen_pos, train_percen_neg, test_percen_pos, test_percen_neg):
    percentage_positive_instances_train = train_percen_pos
    percentage_negative_instances_train = train_percen_neg
    percentage_positive_instances_test = test_percen_pos
    percentage_negative_instances_test = test_percen_neg
    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
    print("\nTrain and Test dataset are completed to preprocessing data\n")
    return pos_train, neg_train, vocab, pos_test, neg_test


########### 2. Prior Probability => train data probability ###########
# Standard : Q1
def prior_standard(pos_data_len, neg_data_len):
    total_len = pos_data_len + neg_data_len
    return pos_data_len / total_len, neg_data_len / total_len

# Laplace : Q2 ~ Q6
def prior_log(pos_data_len, neg_data_len):
    pos_prior, neg_prior = prior_standard(pos_data_len, neg_data_len)
    return log_func(pos_prior), log_func(neg_prior)


########### 3. Likelihood ###########
# likelihood_table 
def likelihood_common(pos_train, neg_train, vocab):
     ###### train ######
    # make likelihood_table
    likelihood_table=pd.DataFrame(0,index=range(len(vocab)), columns=["pos_frequency","neg_frequency"])

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
        likelihood_table.at[i, "pos_frequency"] = pos_word_counts[word]
        likelihood_table.at[i, "neg_frequency"] = neg_word_counts[word]
    
    # add sum end of table
    sum_row=likelihood_table.sum()
    likelihood_table.loc['Sum']=sum_row
    return likelihood_table


# Standard : Q1
def likelihood_standard(pos_train, neg_train, vocab, test_data):
    # get likelihood table
    likelihood_table=likelihood_common(pos_train, neg_train, vocab)
    
    # Calculate portions
    pos_sum = likelihood_table.at["Sum", "pos_frequency"]
    neg_sum = likelihood_table.at["Sum", "neg_frequency"]
    likelihood_table["pos_portion"] = likelihood_table["pos_frequency"] / pos_sum
    likelihood_table["neg_portion"] = likelihood_table["neg_frequency"] / neg_sum

    # Multiply portions for the likelihood table
    multiply_row = likelihood_table.prod(axis=0)

    # calculate likely_pos, likely_neg
    likely_pos=multiply_row["pos_portion"]
    likely_neg=multiply_row["neg_portion"]
    return likely_pos, likely_neg



# Laplace Smoothing : Q2 ~ Q6
def likelihood_laplace(likelihood_table):
    return 4


########### 4. Posterior Probability ###########
# Standard multinomial Naive Bayes
def posterior_standard(pos_train, neg_train, vocab, test_data):
    prior_pos, prior_neg=prior_standard(len(pos_train), len(neg_train))
    likely_pos, likely_neg=likelihood_standard(pos_train, neg_train, vocab, test_data)
    posterior_result=compare_probability(prior_pos*likely_pos, prior_neg*likely_neg)
    return posterior_result



# Laplace Smoothing log(prior probability) + sum (log(likelihoodal))
def laplace_smoothing(prior_pos, prior_neg, likelihood_table, vocab, test_data, alpha):
    # if the probability is 0, numerator+alpha & denominator+alpha*(len(vocab))
    likelihood_table['pos_frequency_alpha'] = likelihood_table['pos_frequency'].apply(lambda x: alpha if x == 0 else x)
    likelihood_table['neg_frequency_alpha'] = likelihood_table['neg_frequency'].apply(lambda x: alpha if x == 0 else x)
    likelihood_table.at['Sum','pos_frequency_alpha'] = likelihood_table.at['Sum','pos_frequency']+alpha*(len(vocab))
    likelihood_table.at['Sum','neg_frequency_alpha'] = likelihood_table.at['Sum','neg_frequency']+alpha*(len(vocab))

    # get the portion 
    likelihood_table["pos_portion_log"] = likelihood_table["pos_frequency_alpha"] / likelihood_table.at["Sum","pos_frequency_alpha"]
    likelihood_table["neg_portion_log"] = likelihood_table["neg_frequency_alpha"] / likelihood_table.at["Sum","neg_frequency_alpha"]

    ##### sort with test dataset 
    # find unique word from test set
    test_unique=find_unique_words(test_data)
    # get the index matching with likelihood_table index
    test_index=find_matching_indices(vocab, test_unique)
    # get the portion from likelihood_table
    likelihood_table=likelihood_table.iloc[test_index]
    #################################################

    # get log summation of both portion
    likelihood_log_pos = np.sum(likelihood_table['pos_portion_log'].apply(np.log))
    likelihood_log_neg = np.sum(likelihood_table['neg_portion_log'].apply(np.log))

    # add with prior probability for final value of posterior probability
    posterior_pos=np.log(prior_pos)+likelihood_log_pos
    posterior_neg=np.log(prior_neg)+likelihood_log_neg
    
    # compare posterior probability between class
    predicted_class=compare_probability(posterior_pos, posterior_neg)
    return predicted_class


########### 5. Confusion Metrix ###########
def confusion_matrix(posterior_pos, posterior_neg):
    TP=posterior_pos.count("Positive")
    TN=posterior_neg.count("Negative")
    FP=posterior_neg.count("Positive")
    FN=posterior_pos.count("Negative")
    print(f"TP:{TP}")
    print(f"FN:{FN}")
    print(f"TN:{TN}")
    print(f"FP:{FP}")
    # calcualte accuracy
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    # calcualte precision
    precision=(TP)/(TP+FP)
    # calculate reacall
    recall=(TP)/(TP+FN)
    # make confusion table
    confusion_table = pd.DataFrame(
        index=["True Positive", "True Negative"],  
        columns=["Predicted Positive", "Predicted Negative"], 
        data=[[TP, FN], [FP, TN]]
    )

    # draw table 
    draw_table(confusion_table,accuracy,precision,recall,"Q1")    

    return confusion_table


# main function
if __name__ == '__main__':
    start = time.time()  # 시작 시간
    ### Question 1 : Standard Multinominal Naive Bayes
    print("[Question 1] Standard Multinomial Naive Bayes with both 20 % Train and Test Dataset\n")
    ### preprocessign data
    pos_train, neg_train, vocab, pos_test, neg_test=preprocessing_data(0.0004, 0.0004, 0.0004, 0.0004)
    print("pos_train")
    print(len(pos_train))
    print("neg_train")
    print(len(neg_train))
    print("pos_test")
    print(len(pos_test))
    print("neg_test")
    print(len(neg_test))
    print("")
    # prior probability
    prior_pos_train,prior_neg_train=prior_standard(len(pos_train), len(neg_train)) 
    ####### input each of data #####
    posterior_pos=list()
    posterior_neg=list()
    for i in range(len(pos_test)):
        # likelihood probability
        likelihood_pos=likelihood_standard(pos_train, neg_train, list(vocab), pos_test)
        likelihood_neg=likelihood_standard(pos_train, neg_train, list(vocab), neg_test)
        print("i")
        print(i)
        # posterior probability
        posterior_pos.append(posterior_standard(pos_train, neg_train, vocab, pos_test))
        posterior_neg.append(posterior_standard(pos_train, neg_train, vocab, neg_test))
    
    # confusion matrix
    confusion_table=confusion_matrix(posterior_pos, posterior_neg)
 
