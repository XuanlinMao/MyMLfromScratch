# Based on Stanford CS224N Assignment2
import numpy as np
from ..utils.utils import softmax, sigmoid


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors (rows of matrix) for all words in vocab
                      (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)
    """

    y_hat = softmax(centerWordVec @ outsideVectors.T).squeeze()
    loss = - np.log(y_hat[outsideWordIdx])
    y = np.zeros((outsideVectors.shape[0],1))
    y[outsideWordIdx] = 1
    gradCenterVec = (y_hat.reshape(-1,1) - y).T @ outsideVectors
    gradOutsideVecs = centerWordVec * (y_hat.reshape(-1,1) - y)

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    # indices = [outsideWordIdx] + negSampleWordIndices

    dotProduct = (centerWordVec @ outsideVectors.T).squeeze()
    loss = - np.log(sigmoid(dotProduct[outsideWordIdx])).sum() - np.log(sigmoid(- dotProduct[negSampleWordIndices])).sum()
    gradCenterVec = - (1 - sigmoid(dotProduct[outsideWordIdx])) * outsideVectors[outsideWordIdx] + \
        ((1 - sigmoid(- dotProduct[negSampleWordIndices])).reshape(-1,1) * outsideVectors[negSampleWordIndices]).sum(axis=0)
    
    gradOutsideVecs = np.zeros_like(outsideVectors)
    gradOutsideVecs[outsideWordIdx,:] += (- (1 - sigmoid(dotProduct[outsideWordIdx])) * centerWordVec).squeeze()
    for negIdx in negSampleWordIndices:
        gradOutsideVecs[negIdx,:] += ((1 - sigmoid(- dotProduct[negIdx])) * centerWordVec).squeeze()

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) for all words in vocab
                        (V in pdf handout)
    outsideVectors -- outside word vectors (as rows) for all words in vocab
                    (U in pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vectors
            (dJ / dV in the pdf handout)
    gradOutsideVectors -- the gradient with respect to the outside word vectors
                        (dJ / dU in the pdf handout)
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    centerWordIdx = word2Ind[currentCenterWord]
    centerWordVec = centerWordVectors[centerWordIdx,:].reshape(1,-1)
    outsideWordIdxs = [word2Ind[outsideWord] for outsideWord in outsideWords]
    for outsideWordIdx in outsideWordIdxs:
        lossi, gradCenterVec, gradOutsideVecs = word2vecLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors, dataset)
        loss += lossi
        gradCenterVecs[centerWordIdx, :] += gradCenterVec.squeeze()
        gradOutsideVectors += gradOutsideVecs

    return loss, gradCenterVecs, gradOutsideVectors
