import sys
import numpy as np
import lxmls.sequences.discriminative_sequence_classifier as dsc
import pdb

class StructuredPerceptron(dsc.DiscriminativeSequenceClassifier):
    ''' Implements a first order CRF'''

    def __init__(self, observation_labels, state_labels, feature_mapper,
                 num_epochs = 10, learning_rate = 1.0, averaged = True):
        dsc.DiscriminativeSequenceClassifier.__init__(self, observation_labels, state_labels, feature_mapper)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.averaged = averaged
        self.params_per_epoch = []

    def train_supervised(self, dataset):
        self.parameters = np.zeros(self.feature_mapper.get_num_features())
        num_examples = dataset.size()
        for epoch in xrange(self.num_epochs):
             num_labels_total = 0
             num_mistakes_total = 0
             for i in xrange(num_examples):
                sequence = dataset.seq_list[i]
                num_labels, num_mistakes = self.perceptron_update(sequence)
                num_labels_total += num_labels
                num_mistakes_total += num_mistakes
             self.params_per_epoch.append(self.parameters.copy())
             acc = 1.0 - float(num_mistakes_total)/float(num_labels_total)
             print "Epoch: %i Accuracy: %f" %(epoch, acc)
        self.trained = True

        if(self.averaged == True):
            new_w = 0
            for old_w in self.params_per_epoch:
                new_w += old_w
            new_w = new_w / len(self.params_per_epoch)
            self.parameters = new_w



    def perceptron_update(self, sequence):
        ## Exercise 3.3
        y_hat = self.viterbi_decode(sequence)[0].y
        y_true = sequence.y
        num_labels = len(sequence.y)
        num_mistakes = sum([int(res) for res in np.array(y_true, dtype=np.int) != np.array(y_hat, dtype=np.int)])
        
        # Update initial features
        true_initial_features = self.feature_mapper.get_initial_features(sequence, y_true[0])
        pred_initial_features = self.feature_mapper.get_initial_features(sequence, y_hat[0])
        self.parameters[true_initial_features] += 1
        self.parameters[pred_initial_features] -= 1

        # Update emission features
        for pos in xrange(len(sequence.x)):
            y_t_true = y_true[pos]
            true_emission_features = self.feature_mapper.get_emission_features(sequence, pos, y_t_true)
            self.parameters[true_emission_features] += 1
            y_t_hat = y_hat[pos]
            pred_emission_features = self.feature_mapper.get_emission_features(sequence, pos, y_t_hat)
            self.parameters[pred_emission_features] -= 1

            # Update transition features.
            if pos > 0:
                prev_y_t_true = y_true[pos-1]
                true_transition_features = self.feature_mapper.get_transition_features(sequence, pos-1, y_t_true, prev_y_t_true)
                self.parameters[true_transition_features] += 1
                prev_y_t_hat = y_hat[pos-1]
                pred_transition_features = self.feature_mapper.get_transition_features(sequence, pos-1, y_t_hat, prev_y_t_hat)
                self.parameters[pred_transition_features] -= 1

        # Update final features
        true_initial_features = self.feature_mapper.get_initial_features(sequence, y_true[-1])
        pred_initial_features = self.feature_mapper.get_initial_features(sequence, y_hat[-1])
        self.parameters[true_initial_features] += 1
        self.parameters[pred_initial_features] -= 1

        return num_labels, num_mistakes


    def save_model(self,dir):
        fn = open(dir+"parameters.txt",'w')
        for p_id,p in enumerate(self.parameters):
            fn.write("%i\t%f\n"%(p_id,p))
        fn.close()

    def load_model(self,dir):
        fn = open(dir+"parameters.txt",'r')
        for line in fn:
            toks = line.strip().split("\t")
            p_id = int(toks[0])
            p = float(toks[1])
            self.parameters[p_id] = p
        fn.close()
