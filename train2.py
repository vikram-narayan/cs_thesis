"""
train2.py
Vikram Narayan
Uses a flat HMM to train a hierarchical HMM. 
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
    the remaining (1-prob_of_note) is divided among remaining notes.
    Assumes that prob_of_note is between 0 and 1 non-inclusive.
    Assumes note is specified in a way analogous with hhmm.notes"""
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

class HMM_node:
    def __init__(self, note):
        self.note=note


class HMM:
    def __init__(self, hierarchicalHMM, filename):
        """converts a hierarchical HMM to a flat HMM"""
        self.hierarchicalHMM=hierarchicalHMM
        self.transitions={}
        self.emissions={}
        self.states={}
        self.start=HMM_node(None)
        self.states[self.start]=1

        self.corresponding_flat_node={}
        self.corresponding_hierarchical_node={}

        # form dictionaries to create lookup dicts for flat and 
        # hierarchical production states
        production_states=hierarchicalHMM.get_pstates(hierarchicalHMM.root)
        for i in production_states:
            hmm_node=HMM_node(i.note)
            self.corresponding_flat_node[i]=hmm_node
            self.corresponding_hierarchical_node[hmm_node]=i
            self.states[hmm_node]=1
            # print "sanity check:", i.horizontal_transitions[hierarchicalHMM.get_eof_state(i)]


        # copy values from i.horizontal_transitions
        for i in production_states:
                i_flat = self.corresponding_flat_node[i]
                self.transitions[i_flat]={}

                for k in i.horizontal_transitions:
                    if k.type==hhmm.EOF_STATE:
                        continue
                    self.transitions[i_flat][self.corresponding_flat_node[k]]=i.horizontal_transitions[k]
        # probability of going from production state i to production state j =
        # (i -> EOF state) * (i's parent -> j's parent) * (j's parent -> j)
        c=0
        for i in production_states:
            for j in production_states:
                i_flat = self.corresponding_flat_node[i]
                j_flat = self.corresponding_flat_node[j]

                # now add transition probabilities for nodes whose parent state is different
                # than i's parent state
                temp=0
                if i.parent==j.parent:
                    continue
                else:
                    i_to_end = i.horizontal_transitions[hierarchicalHMM.get_eof_state(i)]
                    iparent_to_jparent = i.parent.horizontal_transitions[j.parent]
                    jparent_to_j = j.parent.vertical_transitions[j]

                    # i.horizontal_transitions[j] = i_to_end * iparent_to_jparent * jparent_to_j
                    self.transitions[i_flat][j_flat] = i_to_end * iparent_to_jparent * jparent_to_j
                    temp+=i_to_end * iparent_to_jparent * jparent_to_j
                    temp2=i_flat
                    # if temp>0:
                    #     print "temp>0", temp
                    #     print self.transitions[i_flat][j_flat]
                    #     print sum(self.transitions[i_flat].values())
                    #     c+=1
                    #     print c
                    # print "i_to_end:",i_to_end, "     iparent_to_jparent:", iparent_to_jparent,"      jparent_to_j:",jparent_to_j


        # vertical transition probabilities transformed into initial activation probabilities
        # by computing the product of vertical probs from root state to production state
        self.transitions[self.start]={}
        self.transitions[self.start][self.start]=0
        for i in production_states:
            i_flat = self.corresponding_flat_node[i]
            self.transitions[self.start][i_flat]=hierarchicalHMM.ps_to_root(i,1)
            self.transitions[i_flat][self.start]=0
            # self.root.vertical_transitions[i] = self.ps_to_root(i,1)

        # hard code the concept that the last ending phrase symbol transitions
        # back to the start
        for i in hierarchicalHMM.get_pstates(hierarchicalHMM.node_dict['end']):
            if i.note==')':
                i_flat=self.corresponding_flat_node[i]
                self.transitions[i_flat][self.start]=1

        # prevent key error problems with the start state
        self.emissions[self.start]={}
        for note in hhmm.notes:
            self.emissions[self.start][note]=0

        # each production state emits its assigned note with probability 1
        for state in production_states:
        #     self.transitions[state]=copy.copy(state.horizontal_transitions)
            self.emissions[self.corresponding_flat_node[state]]=initialize_emission_probs(state.note, 1)
        # self.states = hierarchicalHMM.root.vertical_transitions.keys()

        # probabilities transitioning from root state
        # self.transitions[hierarchicalHMM.root]= copy.copy(hierarchicalHMM.root.vertical_transitions)
        # self.start = (hierarchicalHMM.root)


        # get corpus from file
        self.observations = read_corpus(filename)


        # print "--------=-=-=-=-=-"
        # for i in self.states:
        #     for j in self.states:
        #         if j not in self.transitions[i]:
        #         print self.transitions[i][j]


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
                    # print "state in fwd_table",state in fwd_table
                    # print "prev_state in self.transitions",prev_state in self.transitions
                    # print "state in self.transitions[prev_state]",state in self.transitions[prev_state]
                    if (state not in self.transitions[prev_state]):
                        pdb.set_trace()
                    # print "state in self.emissions",state in self.emissions
                    # print "state==prev_state",state==prev_state
                    # print "\n\n"
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


    def sigma(self, hierarchical_node):
        """returns the set of production states that are descendants of hierarchical_node.
        returns hierarchical_node if it is a production state"""
        q=[]
        ps=[]
        q.append(hierarchical_node)

        if hierarchical_node.type==hhmm.INTERNAL_STATE:
            while len(q)>0:
                n=q.pop()
                for child in n.vertical_transitions:
                    if child.type==hhmm.PRODUCTION_STATE:
                        ps.append(child)
                    elif child.type==hhmm.INTERNAL_STATE:
                        q.append(child)
            return ps
        elif hierarchical_node.type==hhmm.PRODUCTION_STATE:
            return [hierarchical_node]

    def all_nodes_from_hhmm(self):
        """ retrieves all non-eof nodes from the hhmm. """
        allnodes=[]
        allnodes.append(self.hierarchicalHMM.root)
        q=[]
        q.append(self.hierarchicalHMM.root)
        while len(q)>0:
            n=q.pop()
            for child in n.vertical_transitions:
                if child.type==hhmm.PRODUCTION_STATE:
                    allnodes.append(child)
                elif child.type==hhmm.INTERNAL_STATE:
                    allnodes.append(child)
                    q.append(child)
        return allnodes



    def expectation_maximization(self, corpus, convergence, iterations):
        """given a corpus, which is a list of observations, and
        - apply EM to learn the HMM parameters that maximize the corpus likelihood.
        - stop when log likelihood changes less than the convergence threhshold, or the algorithm has completed the specified number of iterations.
        - update self.transitions and self.emissions, and return the log likelihood
        of the corpus under the final updated parameters."""
        prev_log_likelihood=-float('inf')
        epochs=0
        while (True):
            log_likelihood=0
            print "expectation_maximization: epochs:",epochs

            trans_counts={}
            pi={}
            end_trans={}
            for i in self.all_nodes_from_hhmm():
                trans_counts[i]={}
                pi[i]={}
                end_trans[i]=0
                for j in self.all_nodes_from_hhmm():
                    trans_counts[i][j]=0
                    pi[i][j]=0

            allnodes=self.all_nodes_from_hhmm()
            for observation in corpus:
                print "expectation_maximization: observation:",observation
                alpha = self.forward_algorithm(observation.split())
                beta = self.backward_algorithm(observation.split())
                prob_of_obs = self.total_probability(observation)
                print "expectation_maximization: sanity check that alpha==beta:",prob_of_obs==self.total_probability_bk(observation)


                # gamma[t][i]: prob that node i was active at time t
                gamma={} 

                # compute gamma for production states in the hierarchical HMM by using
                # corresponding alpha and beta values in the flat HMM
                for t in range(len(observation.split())):
                    gamma[t]={}
                    for i in self.states:
                        # the start state of the flat HMM has no hierarchical analogue, so skip it
                        if i==self.start:
                            continue
                        # beta[i] is indexed at time t+1 because the table is 1 longer than the alpha table
                        # print "alpha[i].has_key(t)",alpha[i].has_key(t), 'beta[i].has_key(t+1)',beta[i].has_key(t+1)
                        try:
                            gamma[t][self.corresponding_hierarchical_node[i]] = (alpha[i][t] * beta[i][t+1])/(10**prob_of_obs)
                        except KeyError as ke:
                            pdb.set_trace()

                # xi[t][i][j]: prob that at time t there was a transition from state i to state j
                xi={} 
                # compute xi
                for t in range(len(observation.split())-1):
                    xi[t]={}
                    for i in self.states:

                        if i==self.start:
                            continue
                        xi[t][self.corresponding_hierarchical_node[i]]={}

                        for j in self.states:
                            if j==self.start:
                                continue
                            try:
                                xi[t][self.corresponding_hierarchical_node[i]][self.corresponding_hierarchical_node[j]] = (alpha[i][t] * self.transitions[i][j]  * self.emissions[j][observation.split()[t+1]] * beta[i][t+2])/(10**prob_of_obs)
                            except (KeyError,IndexError) as ke: 
                                pdb.set_trace()


                # calculate gamma for internal states 
                for t in range(len(observation.split())-1):
                    gamma_sum=0
                    for i in allnodes:

                        # skip production states and eof states
                        if i.type!=hhmm.INTERNAL_STATE:
                            continue

                        set_of_i=set(self.sigma(i))
                        not_set_of_i=set(self.sigma(self.hierarchicalHMM.root)) - set_of_i
                        for k in not_set_of_i:
                            for l in set_of_i:
                                gamma_sum+=xi[t][k][l]
                        gamma[t][i]=gamma_sum
                    

                # now re-estimate Tij between all nodes i and j that are not end states

                print "GOT TO TIJ RE_ESTIMATION"
                for i in allnodes:
                    xi_sum_at_each_t=0
                    gamma_sum_at_each_t=0
                    for t in range(len(observation.split())-2):
                        gamma_sum_at_each_t+=gamma[t][i]

                    for j in allnodes:
                        for t in range(len(observation.split())-2):
                            try:
                                sigma_i=set(self.sigma(i))
                                sigma_j=set(self.sigma(j))
                            except TypeError as te:
                                pdb.set_trace()
                            for k in sigma_i:
                                for l in sigma_j:
                                    xi_sum_at_each_t+=xi[t][k][l]
                        # RuntimeWarning: invalid value encountered in double_scalars
                        trans_counts[i][j]+=xi_sum_at_each_t/gamma_sum_at_each_t


                # fill pi[i][j] to estimate hierarchical transitions
                for i in allnodes:
                    for j in allnodes:
                        sigma_j = set(self.sigma(j))
                        sigma_i=set(self.sigma(i))
                        not_sigma_i=set(self.sigma(self.hierarchicalHMM.root)) - set_of_i

                        gamma_sum_pi_numerator=0
                        for k in sigma_j:
                            gamma_sum_pi_numerator+=gamma[0][k]

                        xi_numerator=0
                        for t in range(len(observation.split())-2):
                            for k in not_sigma_i:
                                for l in sigma_j:
                                    xi_numerator+=xi[t][k][l]

                        gamma_denominator=0
                        for k in sigma_i:
                            gamma_denominator+=gamma[0][k]

                        xi_denominator=0
                        for t in range(len(observation.split())-2):
                            for k in not_sigma_i:
                                for l in sigma_i:
                                    xi_denominator+=xi[t][k][l]
                        pi[i][j]+=(gamma_sum_pi_numerator + xi_numerator)/(gamma_denominator + xi_denominator)

                # re-estimate transitions from state i to i's eof state
                for i in allnodes:
                    # skip root because root is the 1 internal state that doesn't have an eof state
                    if i==self.hierarchicalHMM.root:
                        continue
                    i_eof = self.hierarchicalHMM.get_eof_state(i)
                    if i_eof not in trans_counts[i]:
                        trans_counts[i][i_eof]=0

                    sigma_i=set(self.sigma(i))

                    # all production states descended from i's parent 
                    sigma_pi = set(self.sigma(i.parent))

                    xi_numerator=0
                    gamma_denominator=0
                    for t in range(len(observation.split())-2):
                        for k in sigma_i:
                            for l in sigma_pi:
                                xi_numerator+=xi[t][k][l]

                        gamma_denominator+=gamma[t][i]

                    end_trans[i]+=xi_numerator/gamma_denominator
                    trans_counts[i][i_eof]+=xi_numerator/gamma_denominator

                pdb.set_trace()

            # normalize trans_counts, pi, and end_trans
            for i in trans_counts:
                if i==self.hierarchicalHMM.root:
                    continue
                hhmm.normalize(i)
            for i in pi:
                hhmm.normalize(i)

            # transfer values from trans_counts and pi to the hhmm
            for i in allnodes:
                i.vertical_transitions = pi[i]

                # if i==root, only copy the new vertical transitions
                if i==self.hierarchicalHMM.root:
                    continue                   

                i.horizontal_transitions[j] = trans_counts[i]

            print "expectation_maximization: prev_log_likelihood-log_likelihood:", prev_log_likelihood-log_likelihood

            epochs+=1
            if (epochs>iterations) or (prev_log_likelihood-log_likelihood < convergence):
                break

            prev_log_likelihood=log_likelihood

        return log_likelihood
        #     # store emission soft counts
        #     soft_count={}
        #     for state in self.states:
        #         soft_count[state]={}
        #         for i in hhmm.notes:
        #             soft_count[state][i]=0

        #     # store soft counts for transitions
        #     soft_count_trans={}
        #     soft_count_trans[self.start]={}
        #     for state in self.states:
        #         soft_count_trans[state]={}
        #         soft_count_trans[self.start][state]=0
        #         for state2 in self.states:
        #             soft_count_trans[state][state2]=0

        #     for observation in corpus:
        #         total_prob = self.total_probability(observation)
        #         log_likelihood+=total_prob
        #         fwd_matrix = self.forward_algorithm(observation.split())
        #         bk_matrix = self.backward_algorithm(observation.split())
        #         print "total_prob =",total_prob
        #         # new_emissions stores the counts for observation
        #         new_emissions = {}
        #         # new_transitions={}

        #         for state in self.states:
        #             new_emissions[state]=[]
        #             for i in range(len(observation.split())):
        #                 new_emissions[state].append(0)
        #         # emission soft counts
        #         for i in range(len(observation.split())):
        #             for state in self.states:
        #                 new_emissions[state][i] = fwd_matrix[state][i] * bk_matrix[state][i+1] 
        #                 new_emissions[state][i] = new_emissions[state][i]/(10**total_prob)
        #                 if soft_count[state].has_key(observation.split()[i]):
        #                     soft_count[state][observation.split()[i]]+=new_emissions[state][i]
        #                 else:
        #                     soft_count[state][observation.split()[i]]=new_emissions[state][i]

        #         # transition soft counts
        #         for i in range(len(observation.split())-1):
        #             for state in self.states:
        #                 for state2 in self.states:
        #                     soft_count_trans[state][state2]+=(fwd_matrix[state][i] * self.transitions[state][state2] * self.emissions[state2][observation.split()[i+1]] * bk_matrix[state2][i+2])/(10**total_prob)

        #         # update transition probabilities from start
        #         for state in self.states:
        #             soft_count_trans[self.start][state]+= (self.transitions[self.start][state] * self.emissions[state][observation.split()[0]]* bk_matrix[state][1])/(10**total_prob) 
        #         # bss = self.best_state_sequence(observation)
        #         # for state in self.states:
        #         #     # if bss[0]==state:
        #         #     soft_count_trans[self.start][state]+=total_prob * self.emissions[state][observation[0]]

        #     #normalize emission soft counts
        #     for state in self.states:
        #         running_sum=0
        #         for letter in soft_count[state]:
        #             running_sum+=soft_count[state][letter]
        #         for letter in soft_count[state]:
        #             soft_count[state][letter] =soft_count[state][letter]/running_sum


        # #     #update emission probabilities
        # #     for state in self.states:
        # #         for letter in soft_count[state]:
        # #             if soft_count[state][letter]!=0:
        # #                 self.emissions[state][letter] = soft_count[state][letter]

        # #     #normalize transition soft counts
        # #     for state in self.states:
        # #         running_sum=0
        # #         for state2 in self.states:
        # #             running_sum+= soft_count_trans[state][state2]
        # #         for state2 in self.states:
        # #             soft_count_trans[state][state2] = soft_count_trans[state][state2]/running_sum

        # #     running_sum=0
        # #     for state in self.states:
        # #         running_sum+=soft_count_trans[self.start][state]
        # #     for state in self.states:
        # #         soft_count_trans[self.start][state] = soft_count_trans[self.start][state]/running_sum
            
        # #     #update transition probabilities
        # #     for state in self.states:
        # #         for state2 in self.states:
        # #             self.transitions[state][state2] = soft_count_trans[state][state2]

        # #     for state in self.states:
        # #         self.transitions[self.start][state] =soft_count_trans[self.start][state]
            
        # #     epochs+=1
        # #     if epochs>iterations:
        # #         break
        # #     print "EM: epoch",epochs
        # #     print "EM: log_likelihood-prev_log_likelihood =",log_likelihood-prev_log_likelihood

        # #     if (log_likelihood - prev_log_likelihood) < convergence:
        # #         return log_likelihood

        #     prev_log_likelihood=log_likelihood

        # return log_likelihood


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
    for i in ['beginning', 'middle', 'end']:
        new_child = hierarchicalHMM.create_child(parent, name=i)
        # create production states for each note
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
            eof_node=hierarchicalHMM.get_eof_state(pstate)
            if pstate.note=='(':
                i.vertical_transitions[pstate]=1
                for node in i.vertical_transitions:
                    if node!=pstate:
                        i.vertical_transitions[node]=0
                pstate.horizontal_transitions[eof_node]=0

            elif pstate.note==')':
                pstate.horizontal_transitions[eof_node]=1
                for node in pstate.horizontal_transitions:
                    if node!=eof_node:
                        pstate.horizontal_transitions[node]=0
            else:
                pstate.horizontal_transitions[eof_node]=0
            hhmm.normalize(pstate.horizontal_transitions)
            # if pstate.horizontal_transitions[eof_node]!=0.0:
            #     print pstate.note



    # testing self referential loop stuff
    for internal_node in parent.vertical_transitions:
        if internal_node.type==hhmm.INTERNAL_STATE:
            sr=hierarchicalHMM.is_SR(internal_node)
            if sr:
                # print internal_node.horizontal_transitions, "\n\n"
                hierarchicalHMM.convert_to_minSR(internal_node)

    print "flattening hierarchicalHMM..."
    # hierarchicalHMM.flatten()
    # for i in hierarchicalHMM.root.vertical_transitions:
    #     if hierarchicalHMM.root.vertical_transitions[i]==1:
    #         print i.note

    print "converting flattened hierarchicalHMM to normal hmm..."
    hmm = HMM(hierarchicalHMM, 'temp.data')

    # import sys
    # # hhmm.write_midi(hierarchicalHMM.traverse(hierarchicalHMM.root))
    # for i in hierarchicalHMM.traverse(hierarchicalHMM.root):
    #     print i,
    # sys.exit(0)

    # x=hmm.best_state_sequence(hmm.observations[340])
    # y=hmm.total_probability(hmm.observations[329])
    # z=hmm.total_probability_bk(hmm.observations[329])
    print "beginning expectation maximization..."
    alpha=hmm.expectation_maximization(hmm.observations[:3],convergence=0.1, iterations=200)
    for i in xrange(4):
        hmm.generate()
