"""
train.py
Vikram Narayan
Trains a flattened hierarchical hidden markov model (HHMM) in a manner
similar to how normal HMMs are trained.
"""
# system imports
import pdb
import random
import numpy
import numpy.random
import argparse
from collections import defaultdict
import music21
import copy

# local import
import hhmm

def initialize_emission_probs(note, prob_of_note):
    """given note, initialize an emission probability dictionary
    so that the given note occurs with likelihood prob_of_note, and
    the remaining probability is divided among the remaining notes.
    Assumes that prob_of_note is between 0 and 1 non-inclusive.
    Assumes note is specified in a way analogoue with hhmm.notes"""
    emission_probs={}
    emission_probs[note]=prob_of_note
    remainder=1-prob_of_note
    other_notes_probs = remainder/(len(hhmm.notes)-1)
    for n in hhmm.notes:
        if n==note:
            continue
        emission_probs[n]=other_notes_probs
    return emission_probs

def read_corpus(filename):
    """reads corpus file"""
    observations=[]
    f=open(filename, 'r')
    for line in f:
        observations.append(line[:len(line)-1])
    f.close()
    return observations

class HMM:
    def __init__(self, hierarchicalHMM, filename):
        """reads HMM structure from flattened hhmm.
        no error checking: assumes the hhmm is flattened."""
        self.transitions={}
        self.emissions={}
        for state in hierarchicalHMM.root.vertical_transitions:
            self.transitions[state]=copy.copy(state.horizontal_transitions)
            self.emissions[state]=initialize_emission_probs(state.note, 0.99)
        self.states = hierarchicalHMM.root.vertical_transitions.keys()
        # probabilities transitioning from root state
        self.transitions[hierarchicalHMM.root]= copy.copy(hierarchicalHMM.root.vertical_transitions)
        self.start = (hierarchicalHMM.root)

        # domain specific phrase information 
        # self.phrase_begs=[]
        # self.phrase_ends=[]

        # print len(self.transitions[self.start])
        # for transition in self.transitions[self.start]:
        #     print transition.note
        #     if transition.note=='(':
        #         self.phrase_begs.append(transition)
        #     elif transition.note==')':
        #         self.phrase_ends.append(transition)
        # self.transitions[self.start][self.phrase_begs[0]]=1
        # self.transitions[self.phrase_ends[0]][self.phrase_begs[1]]=1
        # self.transitions[self.phrase_ends[1]][self.phrase_begs[1]]=0.75
        # self.transitions[self.phrase_ends[1]][self.phrase_begs[2]]=0.25
        # self.transitions[self.phrase_ends[2]][self.start]=1

        self.observations = read_corpus(filename)

    def best_state_sequence(self, observation):
        """given an observation as a list of symbols,
        find the most likely state sequence that generated it."""
        observation=observation.split()

        viterbi_path = []
        for i in range(len(observation)):
            viterbi_path.append('')

        # initialize table for viterbi algorithm
        viterbi_table={}
        back_pointers={}
        for state in self.states:
            viterbi_table[state]=[]
            back_pointers[state]=[]
            for i in range(len(observation)):
                viterbi_table[state].append(0)
                back_pointers[state].append('')

        # initialize first column of viterbi table
        actual_max=-float('inf')
        for state in self.states:
            viterbi_table[state][0] = numpy.log10(self.transitions[self.start][state] * self.emissions[state][observation[0]] ) 
            back_pointers[state][0]=self.start

            if viterbi_table[state][0] > actual_max:
                actual_max = viterbi_table[state][0]
                viterbi_path[0] = state

        # fill in rest of viterbi table and the viterbi path
        for output in range(1,len(observation)):
            for state in self.states:
                possible_max={}
                for prev_state in self.states:
                    possible_max[prev_state] = (viterbi_table[prev_state][output-1] + numpy.log10(self.transitions[prev_state][state]*self.emissions[state][observation[output]]))
   
                actual_max=-float('inf')
                actual_prevstate=''
                for value in possible_max:
                    if possible_max[value] > actual_max:
                        actual_max = possible_max[value]
                        actual_prevstate=value

                viterbi_table[state][output] = actual_max
                back_pointers[state][output] = actual_prevstate
                viterbi_path[output-1] = actual_prevstate

        # get the final state in the viterbi path
        actual_max=-float('inf')
        actual_prevstate=''
        backtrace_starter=''
        for state in self.states:
            if viterbi_table[state][len(observation)-1] > actual_max:
                actual_max = viterbi_table[state][len(observation)-1]
                backtrace_starter=state
                viterbi_path[len(observation)-1] = state

        # follow the backtrace to get the viterbi path
        stack=[backtrace_starter]
        iterator=backtrace_starter
        for i in range(len(observation)-1,0,-1):
            stack.append(back_pointers[iterator][i])
            iterator=back_pointers[iterator][i]
            back_pointers
        viterbi_path2=[]
        while len(stack)>0:
            viterbi_path2.append(stack.pop())

        return (viterbi_path,viterbi_path2)

    # def forward_algorithm(self, observation):
    #     """given an observation as a list of symbols,
    #     run the forward algorithm"""

    #     # initialize forward algorithm table
    #     fwd_table={}
    #     fwd_table['scaling factor']=[]
    #     for i in xrange(len(observation)):
    #         fwd_table['scaling factor'].append(0)

    #     for state in self.states:
    #         fwd_table[state]=[]
    #         for i in xrange(len(observation)):
    #             fwd_table[state].append(0)

    #     # initialize first col of fwd algorithm table
    #     for state in self.states:
    #         # logs will be taken at the end 
    #         fwd_table[state][0] = (self.transitions[self.start][state] * self.emissions[state][observation[0]] ) 

    #     # fill in the rest of the forward table
    #     for output in xrange(1,len(observation)):
    #         for state in self.states:
    #             fwd=0
    #             for prev_state in self.states:
    #                 fwd+=fwd_table[prev_state][output-1] * self.transitions[prev_state][state] * self.emissions[state][observation[output]]
    #             fwd_table[state][output] = fwd

    #     return fwd_table
    def forward_algorithm(self, observation):
        """given an observation as a list of symbols,
        run the forward algorithm"""
        
        # initialize forward algorithm table
        fwd_table={}
        fwd_table['scaling factor']=[]
        for i in range(len(observation)):
            fwd_table['scaling factor'].append(0)

        for state in self.states:
            fwd_table[state]=[]
            for i in range(len(observation)):
                fwd_table[state].append(0)

        # initialize first col of fwd algorithm table
        for state in self.states:
            # logs will be taken at the end 
            fwd_table[state][0] = (self.transitions[self.start][state] * self.emissions[state][observation[0]] ) 

        # fill in the rest of the forward table
        for output in range(1,len(observation)):
            for state in self.states:
                fwd=0
                for prev_state in self.states:
                    fwd+=fwd_table[prev_state][output-1] * self.transitions[prev_state][state] * self.emissions[state][observation[output]]
                fwd_table[state][output] = fwd

        return fwd_table
    def total_probability(self, observation):
        """compute the probability of the observation under the model"""
        observation=observation.split()
        fwd_table = self.forward_algorithm(observation)

        # forward_prob = numpy.log10(numpy.prod(fwd_table['scaling factor']))

        forward_prob=0
        for state in self.states:
            forward_prob+=fwd_table[state][len(observation)-1]

        return numpy.log10(forward_prob)



    def backward_algorithm(self, observation):
        """given an observation as a list of symbols,
        find the probability of the observation under this HMM,
        using the backward algorithm"""
        # initialize backward algorithm table
        bk_table={}
        bk_table['scaling factor']=[]
        observation2 = [self.start]+observation
        for i in range(len(observation2)):
            bk_table['scaling factor'].append(0)

        for state in self.states:
            bk_table[state]=[]
            for i in range(len(observation2)):
                bk_table[state].append(0)

        # initialize and scale last column
        for state in self.states:
            bk_table[state][len(observation2)-1]=1.0

        output=len(observation2)-2
        while output>=1:
            for state in self.states:
                back=0
                for after_state in self.states:
                    back+=self.transitions[state][after_state] * self.emissions[after_state][observation2[output+1]] * bk_table[after_state][output+1]
                bk_table[state][output] = back

            output=output-1

        back=0
        for state in self.states:
            back+= self.transitions[self.start][state] * self.emissions[state][observation2[1]] * bk_table[state][1]
        for state in self.states:
            bk_table[state][0]=back

        return bk_table

    def total_probability_bk(self, observation):
        """compute the probability of the observation under the model"""
        observation=observation.split()
        bk_table = self.backward_algorithm(observation)
        for state in self.states:
            bk_prob = bk_table[state][0]
        
        return numpy.log10(bk_prob)


    def expectation_maximization(self, corpus, convergence, iterations):
        """given a corpus, which is a list of observations, and
        apply EM to learn the HMM parameters that maximize the corpus likelihood.
        stop when the log likelihood changes less than the convergence threhshold.
        update self.transitions and self.emissions, and return the log likelihood
        of the corpus under the final updated parameters."""
        prev_log_likelihood=-float('inf')
        epochs=0
        while (True):
            log_likelihood=0

            # store emission soft counts
            soft_count={}
            for state in self.states:
                soft_count[state]={}
                for i in hhmm.notes:
                    soft_count[state][i]=0

            # store soft counts for transitions
            soft_count_trans={}
            soft_count_trans[self.start]={}
            for state in self.states:
                soft_count_trans[state]={}
                soft_count_trans[self.start][state]=0
                for state2 in self.states:
                    soft_count_trans[state][state2]=0

            for observation in corpus:
                total_prob = self.total_probability(observation)
                log_likelihood+=total_prob
                fwd_matrix = self.forward_algorithm(observation.split())
                bk_matrix = self.backward_algorithm(observation.split())
                print "total_prob =",total_prob
                # new_emissions stores the counts for observation
                new_emissions = {}
                # new_transitions={}

                for state in self.states:
                    new_emissions[state]=[]
                    for i in range(len(observation.split())):
                        new_emissions[state].append(0)
                # emission soft counts
                for i in range(len(observation.split())):
                    for state in self.states:
                        new_emissions[state][i] = fwd_matrix[state][i] * bk_matrix[state][i+1] 
                        new_emissions[state][i] = new_emissions[state][i]/(10**total_prob)
                        if soft_count[state].has_key(observation.split()[i]):
                            soft_count[state][observation.split()[i]]+=new_emissions[state][i]
                        else:
                            soft_count[state][observation.split()[i]]=new_emissions[state][i]

                # transition soft counts
                for i in range(len(observation.split())-1):
                    for state in self.states:
                        for state2 in self.states:
                            soft_count_trans[state][state2]+=(fwd_matrix[state][i] * self.transitions[state][state2] * self.emissions[state2][observation.split()[i+1]] * bk_matrix[state2][i+2])/(10**total_prob)

                # update transition probabilities from start
                for state in self.states:
                    soft_count_trans[self.start][state]+= (self.transitions[self.start][state] * self.emissions[state][observation.split()[0]]* bk_matrix[state][1])/(10**total_prob) 
                # bss = self.best_state_sequence(observation)
                # for state in self.states:
                #     # if bss[0]==state:
                #     soft_count_trans[self.start][state]+=total_prob * self.emissions[state][observation[0]]

            pdb.set_trace()
            #normalize emission soft counts
            for state in self.states:
                running_sum=0
                for letter in soft_count[state]:
                    running_sum+=soft_count[state][letter]
                for letter in soft_count[state]:
                    soft_count[state][letter] =soft_count[state][letter]/running_sum



        #     #update emission probabilities
        #     for state in self.states:
        #         for letter in soft_count[state]:
        #             if soft_count[state][letter]!=0:
        #                 self.emissions[state][letter] = soft_count[state][letter]

        #     #normalize transition soft counts
        #     for state in self.states:
        #         running_sum=0
        #         for state2 in self.states:
        #             running_sum+= soft_count_trans[state][state2]
        #         for state2 in self.states:
        #             soft_count_trans[state][state2] = soft_count_trans[state][state2]/running_sum

        #     running_sum=0
        #     for state in self.states:
        #         running_sum+=soft_count_trans[self.start][state]
        #     for state in self.states:
        #         soft_count_trans[self.start][state] = soft_count_trans[self.start][state]/running_sum
            
        #     #update transition probabilities
        #     for state in self.states:
        #         for state2 in self.states:
        #             self.transitions[state][state2] = soft_count_trans[state][state2]

        #     for state in self.states:
        #         self.transitions[self.start][state] =soft_count_trans[self.start][state]
            
        #     epochs+=1
        #     if epochs>iterations:
        #         break
        #     print "EM: epoch",epochs
        #     print "EM: log_likelihood-prev_log_likelihood =",log_likelihood-prev_log_likelihood

        #     if (log_likelihood - prev_log_likelihood) < convergence:
        #         return log_likelihood

        #     prev_log_likelihood=log_likelihood

        # return log_likelihood

    # def post_processing(self):
    #     """after an hmm has been trained, the probabilities of certain states
    #     must be adjusted (as per the Weiland paper)"""

    def generate(self):
        """after an hmm has been trained, use it to generate songs
        REWRITE THIS"""
        current=self.start
        emission_notes=[]
        current = hhmm.probabilistic_choice(self.transitions[current])
        emission_notes.append(hhmm.probabilistic_choice(self.emissions[current]))
        while True:
            current = hhmm.probabilistic_choice(self.transitions[current])
            if current.type==hhmm.EOF_STATE or current==self.start:
                break
            emission_notes.append(hhmm.probabilistic_choice(self.emissions[current]))

        hhmm.write_midi(emission_notes)

if __name__=='__main__':
    print "making hierarchicalHMM..."
    hierarchicalHMM = hhmm.HHMM()

    # # create sub-states for beginning, middle, and end,
    # # create production states for each note
    parent = hierarchicalHMM.root
    # for i in xrange(3):
    #     new_child = hierarchicalHMM.create_child(parent)
    #     for note in hhmm.notes:
    #         hierarchicalHMM.create_child(new_child, internal=False, note=note)
    #     hhmm.normalize(new_child.vertical_transitions)

    # hhmm.normalize(parent.vertical_transitions)
    # hierarchicalHMM.initialize_horizontal_probs(parent)

    # create sub-states for beginning, middle, and end,
    # create production states for each note

    for i in ['beginning', 'middle', 'end']:
        new_child = hierarchicalHMM.create_child(parent, name=i)
        for note in hhmm.notes:
            hierarchicalHMM.create_child(new_child, internal=False, note=note)
        hierarchicalHMM.initialize_horizontal_probs(new_child)
        hhmm.normalize(new_child.vertical_transitions)

    hierarchicalHMM.initialize_horizontal_probs(parent)
    hhmm.normalize(parent.vertical_transitions)

    beginning_node=hierarchicalHMM.node_dict['beginning']
    middle_node=hierarchicalHMM.node_dict['middle']
    end_node=hierarchicalHMM.node_dict['end']
    # change probabilities as follows: 

    # P(root->beginning_node)=1
    parent.vertical_transitions[beginning_node]=1
    for node in parent.vertical_transitions:
        if node==beginning_node:
            continue
        parent.vertical_transitions[node]=0

    # P(beginning->middle)=1
    beginning_node.horizontal_transitions[middle_node]=1
    for node in beginning_node.horizontal_transitions:
        if node==middle_node:
            continue
        beginning_node.horizontal_transitions[node]=0


    # P(middle->middle)=0.7 (note this is just preliminary, and may change)
    middle_node.horizontal_transitions[middle_node]=0.7
    # P(middle->end)=0.3
    middle_node.horizontal_transitions[end_node]=0.3
    for node in middle_node.horizontal_transitions:
        if node==middle_node or node==end_node:
            continue
        middle_node.horizontal_transitions[node]=0

    # P(end->EOF)=1
    eof_node=hierarchicalHMM.get_eof_state(end_node)
    end_node.horizontal_transitions[eof_node]=1
    for node in end_node.horizontal_transitions:
        if node==eof_node:
            continue
        end_node.horizontal_transitions[node]=0

    # within each of [beginning, middle, end], p(internalstate->'(')=1
    # i.e., each phrase must begin with a start-of-phrase symbol
    # P(')'->EOF)=1
    for i in [beginning_node, middle_node, end_node]:
        for pstate in hierarchicalHMM.get_pstates(i):
            if pstate.note=='(':
                i.vertical_transitions[pstate]=1
                for node in i.vertical_transitions:
                    if node!=pstate:
                        i.vertical_transitions[node]=0
            elif pstate.note==')':
                eof_node=hierarchicalHMM.get_eof_state(pstate)
                pstate.horizontal_transitions[eof_node]=1
                for node in pstate.horizontal_transitions:
                    if node!=eof_node:
                        pstate.horizontal_transitions[node]=0

    # testing self referential loop stuff
    for internal_node in parent.vertical_transitions:
        if internal_node.type==hhmm.INTERNAL_STATE:
            sr=hierarchicalHMM.is_SR(internal_node)
            if sr:
                # print internal_node.horizontal_transitions, "\n\n"
                hierarchicalHMM.convert_to_minSR(internal_node)

    print "flattening hierarchicalHMM..."
    hierarchicalHMM.flatten()
    for i in hierarchicalHMM.root.vertical_transitions:
        if hierarchicalHMM.root.vertical_transitions[i]==1:
            print i.note
    pdb.set_trace()

    print "converting flattened hierarchicalHMM to normal hmm..."
    hmm = HMM(hierarchicalHMM, 'bach_chorales_cmajor_aminor_midi.data')
    # x=hmm.best_state_sequence(hmm.observations[340])
    # y=hmm.total_probability(hmm.observations[329])
    # z=hmm.total_probability_bk(hmm.observations[329])
    print "beginning expectation maximization..."
    alpha=hmm.expectation_maximization(hmm.observations[:3],convergence=0.1, iterations=200)
    for i in xrange(4):
        hmm.generate()
