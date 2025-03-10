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
    print("\n[Question 6 Apply Laplace Smoothing with Unbalanced 10%,50% Train and 100% Test Dataset\n")
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