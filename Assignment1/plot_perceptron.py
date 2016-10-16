"""
Plots information about a perceptron classifier on a 2-dimensional dataset.
"""

import matplotlib.pyplot as plt

def plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, w, w_dist_history):
    """
    % The top-left plot shows the dataset and the classification boundary given by
    % the weights of the perceptron. The negative examples are shown as circles
    % while the positive examples are shown as squares. If an example is colored
    % green then it means that the example has been correctly classified by the
    % provided weights. If it is colored red then it has been incorrectly classified.
    % The top-right plot shows the number of mistakes the perceptron algorithm has
    % made in each iteration so far.
    % The bottom-left plot shows the distance to some generously feasible weight
    % vector if one has been provided (note, there can be an infinite number of these).
    % Points that the classifier has made a mistake on are shown in red,
    % while points that are correctly classified are shown in green.
    % The goal is for all of the points to be green (if it is possible to do so).
    % Inputs:
    %   neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
    %       num_neg_examples is the number of examples for the negative class.
    %   pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
    %       num_pos_examples is the number of examples for the positive class.
    %   mistakes0 - A vector containing the indices of the datapoints from class 0 incorrectly
    %       classified by the perceptron. This is a subset of neg_examples.
    %   mistakes1 - A vector containing the indices of the datapoints from class 1 incorrectly
    %       classified by the perceptron. This is a subset of pos_examples.
    %   num_err_history - A vector containing the number of mistakes for each
    %       iteration of learning so far.
    %   w - A 3-dimensional vector corresponding to the current weights of the
    %       perceptron. The last element is the bias.
    %   w_dist_history - A vector containing the L2-distance to a generously
    %       feasible weight vector for each iteration of learning so far.
    %       Empty if one has not been provided.
    %%
    """

    fig = plt.figure()
    ax1 = fig.add_subplot(221)

    neg_correct_ind = [i for i in range(len(neg_examples)) if i not in mistakes0]
    pos_correct_ind = [i for i in range(len(pos_examples)) if i not in mistakes1]

    if neg_examples.any():
        ax1.scatter(neg_examples[neg_correct_ind,0], neg_examples[neg_correct_ind,1], marker='o', s=80, color='green')

    if pos_examples.any():
        ax1.scatter(pos_examples[pos_correct_ind,0], pos_examples[pos_correct_ind,1], marker='s', s=80, color='green')

    if mistakes0:
        ax1.scatter(neg_examples[mistakes0,0], neg_examples[mistakes0,1], marker='o', s=80, color='red')

    if mistakes1:
        ax1.scatter(pos_examples[mistakes1,0], pos_examples[mistakes1,1], marker='s', s=80, color='red')

    ax1.plot([-5,5], [(-w[-1]+5*w[0])/w[1], (-w[-1]-5*w[0])/w[1]])
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-1,1])
    
    ax1.set_title('Classifier')

    ax2 = fig.add_subplot(222)
    ax2.plot(num_err_history)
    ax2.set_xlim([-1,max(15,len(num_err_history))]);
    ax2.set_ylim([0,neg_examples.shape[0]+pos_examples.shape[0]+1]);
    ax2.set_title('Number of errors');
    ax2.set_xlabel('Iteration');
    ax2.set_ylabel('Number of errors');

    ax3 = fig.add_subplot(223)
    ax3.plot(w_dist_history)
    ax3.set_xlim([-1,max(15,len(num_err_history))]);
    ax3.set_ylim([0,15]);
    ax3.set_title('Distance')
    ax3.set_xlabel('Iteration');
    ax3.set_ylabel('Distance');


    plt.show()
