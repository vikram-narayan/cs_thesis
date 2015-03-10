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

        self.observations = read_corpus(filename)

    def best_state_sequence(self, observation):
        """given an observation as a list of symbols,
        find the most likely state sequence that generated it."""
        observation=observation.split()

        viterbi_path = []
        for i in range(len(observation)):
            viterbi_path.append('')

        # -- TODO --
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

        # -- TODO end
        return (viterbi_path,viterbi_path2)




if __name__=='__main__':
	hierarchicalHMM = hhmm.HHMM()

	# create sub-states for beginning, middle, and end,
	# create production states for each note
	parent = hierarchicalHMM.root
	for i in xrange(3):
		new_child = hierarchicalHMM.create_child(parent)
		for note in hhmm.notes:
			hierarchicalHMM.create_child(new_child, internal=False, note=note)
		hhmm.normalize(new_child.vertical_transitions)

	hhmm.normalize(parent.vertical_transitions)
	hierarchicalHMM.initialize_horizontal_probs(parent)

	# testing self referential loop stuff
	for internal_node in parent.vertical_transitions:
		if internal_node.type==hhmm.INTERNAL_STATE:
			sr=hierarchicalHMM.is_SR(internal_node)
			if sr:
				# print internal_node.horizontal_transitions, "\n\n"
				hierarchicalHMM.convert_to_minSR(internal_node)

	hierarchicalHMM.flatten()
	hmm = HMM(hierarchicalHMM, 'bach_chorales_a4.data')
	x=hmm.best_state_sequence(hmm.observations[340])
	pdb.set_trace()
