import numpy as np
import nltk
import csv
import string
import operator
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
nltk.download('stopwords')

def process_raw(path):
    """Read raw file and convert content to a list of reviews.

    Parameters
    ----------
    path : str
        File path to review dataset.

    Returns
    -------
    data : list
        List of reviews contained within dataset.

    """
    data = []
    with open(path, 'r', encoding='utf8') as f:
        contents = csv.reader(f, delimiter='\n')
        for row in contents:
            data.append(row)

    return data

def stratify_datasets(pos, neg):
    """Combines positive and negative reviews in to a single list along
    with their respective label.

    Parameters
    ----------
    pos : list
        List of positive reviews for a particular dataset.
    neg : list
        List of negative reviews for a particular dataset.

    Returns
    -------
    combined : list
        List of (review, label) pairs.

    """
    pos_tuple = [(''.join(i), 1) for i in pos]
    neg_tuple = [(''.join(i), 0) for i in neg]
    
    combined = pos_tuple + neg_tuple
    
    return combined

def get_labels(dataset):
    """Get labels for a particular dataset. Not essential but saves 
    repeating the same list comprehension throughout the code.

    Parameters
    ----------
    dataset : list
        List of (review, label) pairs.

    Returns
    -------
    labels : list
        List of labels for each review.

    """
    labels = [i[1] for i in dataset]
    
    return labels

def tokenise_and_lemmatise(review):    
    """Splits a review in to individual tokens, and then lemmatises said
    tokens.

    Parameters
    ----------
    review : tuple
        A single review from the dataset.

    Returns
    -------
    tokenised_and_lemmatised : list
        List of tokenised and lemmatised words in the review.

    """
    tokenised_review = nltk.tokenize.word_tokenize(review[0])
    tokenised_and_lemmatised = []
    for word in tokenised_review:
        tokenised_and_lemmatised.append(lemmatiser.lemmatize(word).lower())

    return tokenised_and_lemmatised

def get_vocabulary(dataset, vocab_length):
    """Create an ordered list of a set length of the most frequent words
     in a whole dataset, referred to as the 'vocabulary'. Utilises list
     of stopwords specified by the default nltk set as well as custom
     additions.

    Parameters
    ----------
    dataset : list
        List of reviews.
    vocab_length : int
        Specified length of the vocabulary.

    Returns
    -------
    vocabulary : list
        List of {vocab_length} most frequent words in the dataset.

    """
    frequency_dict = {}
    for review in dataset:
        tokenised_review = tokenise_and_lemmatise(review)
        for token in tokenised_review:
            if token in all_stopwords:
                continue
            if token not in frequency_dict:
                frequency_dict[token] = 1
            elif token in frequency_dict:
                frequency_dict[token] += 1
    
    sorted_frequency_list = sorted(frequency_dict.items(), 
                                   key = operator.itemgetter(1),
                                   reverse = True)
    
    vocabulary = [word for word, _ in sorted_frequency_list[:vocab_length]]
    
    return vocabulary

def get_vocab_vector(vocabulary, review):
    """Create a vector for a review based on the vocabulary. The vector
    is the same length as the vocabulary, and each element of the vector
    corresponds to the number of occurences of that word in the review. 
    Used as a feature for fitting the model to.

    Parameters
    ----------
    vocabulary : list
        List of most frequent words.
    review : tuple
        A single review from the dataset.

    Returns
    -------
    final_vector : ndarray
        Array of number of occurences of words in vocabulary.

    """
    vector = []
    token_list = tokenise_and_lemmatise(review)
    
    for _, word in enumerate(vocabulary):
        if word in token_list:
            vector.append(token_list.count(word))
        else: 
            vector.append(0)
    
    final_vector = np.asarray(vector, dtype = np.int8)

    return final_vector

def get_number_of_sentences(review):
    """Find the number of sentences in a review. Used as a feature for
    fitting the model to.

    Parameters
    ----------
    review : tuple
        A single review from the dataset.

    Returns
    -------
    number_of_sentences : int
        The number of sentences that {review} is composed of.

    """
    sentences = nltk.tokenize.sent_tokenize(review[0])
    number_of_sentences = len(sentences)
    
    return number_of_sentences

def get_mean_sentence_length(review):
    """Get the mean number of words per sentence in each review.  Used
    as a feature for fitting the model to.

    Parameters
    ----------
    review : tuple
        A single review from the dataset.

    Returns
    -------
    mean_length : int
        The mean word length of each sentence in {review}. Rounded to
        the nearest integer.

    """
    sentences = nltk.tokenize.sent_tokenize(review[0])
    word_split = [nltk.tokenize.word_tokenize(i) for i in sentences]
    mean_length = int(round(np.mean([len(i) for i in word_split])))
    
    return mean_length

def combine_features(vocabulary, review):
    """Call each of the 3 feature selecting functions and combine them 
    in to a single vector. The resulting vector is therefore of the
    length of the vocabulary + 2, which are the results of the two
    additional features mean_sentence_length and number_of_sentences.

    Parameters
    ----------
    vocabulary : list
        List of the most frequent words in the dataset.
    review : tuple
        A single review from the dataset.

    Returns
    -------
    x : ndarray
        A vector containing the vocabulary vector derived from
        vocab_vector, with mean sentence length and number of sentences
        appended to the end.

    """
    v = get_vocab_vector(vocabulary, review)
    s = get_mean_sentence_length(review)
    n = get_number_of_sentences(review)
    
    x = np.append(v, [s, n])
    
    return x

def create_feature_arrays(dataset, vocabulary):
    """Create arrays X and y for the model to fit. X is created by
    successively appending x arrays for each review, while y is an array
    composed of the labels of each review.

    Parameters
    ----------
    dataset : list
        List of reviews.
    vocabulary : list
        List of the most frequent words in the dataset.

    Returns
    -------
    X : ndarray
        A 2d vector containing x (feature) vectors for each review.
    y : ndarray
        A 1d vector containing the label for each review.

    """
    X = []
    y = []
    
    for review in dataset:
        x = combine_features(vocabulary, review)
        X.append(x)
        y.append(review[1])

    return X, y

def reduce_dimensions(X, y, target_dimensions):
    """Use a chi-squared test to reduce the X vector down to a select 
    number of features by eliminating words that are common to both
    positive and negative reviews.

    Parameters
    ----------
    X : ndarray
        2d vector of features for each review.
    y : ndarray
        1d vector containing the label for each review.

    Returns
    -------
    reducer : SelectKBest
        An SelectKBest class which is fitted on to the X, y arrays.
    new_X : ndarray
        A 2d vector of features for each review, reduced town to a
        length of {target_dimensions} for each review.

    """
    reducer = SelectKBest(chi2, k = target_dimensions).fit(X, y)
    new_X = reducer.transform(X)
    X_array = np.asarray(new_X, dtype = np.int8)
    
    return reducer, X_array

def get_predictions(classifier, reducer, test, vocabulary):
    """Use the model to make predictions on reviews in the test set.

    Parameters
    ----------
    classifier : SVC
        An instance of the SVC class of sklearn.svm.
    reducer : SelectKBest
        An instance of SelectKBest.fit.
    test : list
        The test dataset.
    vocabulary : list
        List of the most frequent words in the dataset.

    Returns
    -------
    predictions : list
        List of predictions {0, 1} for each review in the dataset.

    """
    predictions = []
    for review in test:
        vector = combine_features(vocabulary, review)
        prediction = classifier.predict(reducer.transform([vector]))
        predictions.append(int(prediction))
        
    return predictions

def metrics(test, predictions):
    """Obtain precision, recall, f-measure and accuracy scores for the 
    model based on the test dataset labels and the predictions.

    Parameters
    ----------
    test : list
        The test dataset.
    predictions : list
        List of predictions.

    Returns
    -------
    precision : float
        The precision of the model, TP/(TP + FP)
    recall : float
        The recall of the model, TP/(TP + FN)
    f1 : float
        The f-measure score of the model, 2TP/(2TP + FP + FN).
    accuracy : float
        The accuracy of the model, (TP + TN)/(P + N).

    """

    gold = get_labels(test)
    precision = precision_score(gold, predictions, average='macro')
    recall = recall_score(gold, predictions, average='macro')
    f1 = f1_score(gold, predictions, average='macro')
    accuracy = accuracy_score(gold, predictions)
    
    return precision, recall, f1, accuracy

def tune_parameters(dev, test, vocabulary_lengths):
    """Train a number of different models based on different vocabulary 
    lengths, returning the accuracy score for each. This function uses
    the dev set in order to find the optimum number of features to train
    the final model on. Writes output to a file "IMDB_fine_tuning.txt"
    for plotting, rather than programmatically returning best accuracy.
    Contains print messages as this can take a long time.

    NOT NECESSARY FOR FINAL TRAINING BUT WAS USED EARLIER TO OBTAIN DATA
    FOR PLOTTING FROM WHICH AN OPTIMUM VALUE FOR VOCABULARY LENGTH WAS 
    CHOSEN.

    Parameters
    ----------
    dev : list
        The development dataset.
    test : list
        The test dataset.
    vocabulary_lengths : list
        List of int. Vocabulary lengths to test.

    Returns
    -------
    accuracies : list
        List of accuracy scores for models trained on each vocabulary
        length.

    """
    print("Testing vocabulary lengths: {}".format(', '.join([str(i) for i in vocabulary_lengths])))
    accuracies = []
    
    for _, length in enumerate(vocabulary_lengths):
        print("Testing length: {}".format(length))
        print("Getting vocabulary...")
        vocabulary = get_vocabulary(dev, length)
        print("Creating feature vectors...")
        X, y = create_feature_arrays(dev, vocabulary)
        print("Performing dimensionality reduction...")
        reducer, reduced_X = reduce_dimensions(X, y, int(length/2))
        classifier = SVC(kernel='linear', gamma='auto')
        print("Fitting...")
        classifier.fit(reduced_X, y)
        print("Making predictions...")
        predictions = get_predictions(classifier, reducer, test, vocabulary)
        _, _, _, accuracy = metrics(test, predictions)
        accuracies.append(accuracy)
        print("Finished iteration.")
        
    print("Finished all testing.")
    
    with open("IMDB_fine_tuning.txt", 'w') as f:
        f.write("{}".format(', '.join([str(i) for i in accuracies])))
    f.close()
    
    return accuracies

def full_pipeline(train, test, vocabulary_length, save=True):
    """Final model training routine using specified vocabulary length.
    
    Parameters
    ----------
    train : list
        The training dataset.
    test : list
        The test dataset.
    vocabulary_length : int
        Specify the length of the vocabulary to create.
    save : bool
        If True, write a file containing the test labels along with the
        predictions.

    Returns
    -------
    precision : float
        The precision of the model.
    recall : float
        The recall of the model.
    f1 : float
        The f-measure score of the model.
    accuracy : float
        The accuracy of the model.

    """
    vocabulary = get_vocabulary(train, vocabulary_length)
    X, y = create_feature_arrays(train, vocabulary)
    reducer, reduced_X = reduce_dimensions(X, y, target_dimensions = 375)
    classifier = SVC(kernel='linear', gamma='auto') # Initialise classifier
    classifier.fit(reduced_X, y) # Fit function
    predictions = get_predictions(classifier, reducer, test, vocabulary)
    precision, recall, f1, accuracy = metrics(test, predictions)
    
    if save:
        with open("IMDb model training results.txt", 'w') as f:
            f.write(', '.join([str(r[1]) for r in test]))
            f.write('\n')
            f.write(', '.join([str(p) for p in predictions]))
        f.close()
    
    return precision, recall, f1, accuracy

if __name__ == '__main__':
	
    # Process all datasets
    train_pos = process_raw('IMDb/train/imdb_train_pos.txt')
    train_neg = process_raw('IMDb/train/imdb_train_neg.txt')
    test_pos = process_raw('IMDb/test/imdb_test_pos.txt')
    test_neg = process_raw('IMDb/test/imdb_test_neg.txt')
    dev_pos = process_raw('IMDb/dev/imdb_dev_pos.txt')
    dev_neg = process_raw('IMDb/dev/imdb_dev_neg.txt')
    
    # Combine and add labels
    training = stratify_datasets(train_pos, train_neg)
    dev = stratify_datasets(dev_pos, dev_neg)
    test = stratify_datasets(test_pos, test_neg)
    
    lemmatiser = nltk.stem.WordNetLemmatizer()

    # Initialise nltk stopwords along with custom list
    nltk_stopwords = set(nltk.corpus.stopwords.words('english'))
    punctuation_stopwords = {i for i in string.punctuation}
    custom_stopwords = {'br', "'s", 'wa', "''", '``', "n't", '...'}
    all_stopwords = nltk_stopwords.union(punctuation_stopwords, custom_stopwords)
    
    # Train model and print results!
    p, r, f1, a = full_pipeline(training, test, 750)

    print("========== Results ==========\n")
    print("    Precision: {:.3f}".format(p))
    print("       Recall: {:.3f}".format(r))
    print("    F-measure: {:.3f}".format(f1))
    print("     Accuracy: {:.3f}\n".format(a))
    print("=============================")
