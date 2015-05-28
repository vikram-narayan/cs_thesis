"""
train3.py
Vikram Narayan
Uses a new, simpler hierarchical EM algorithm to train. Currently only trains transitions.
Applicable to hierarchical sequential data. 
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
import math

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
    # for n in ['0','1','2','3']:
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
        """converts a minimally self referential hierarchical HMM to a flat HMM"""
        self.hierarchicalHMM=hierarchicalHMM
        self.transitions={}
        self.emissions={}
        self.states={}
        self.start=HMM_node(None)
        self.states[self.start]=1

        self.corresponding_flat_node={}
        self.corresponding_hierarchical_node={}

        # form dictionaries to create lookup dicts for flat and hierarchical production states
        production_states=hierarchicalHMM.get_pstates(hierarchicalHMM.root)
        for i in production_states:
            hmm_node=HMM_node(i.note)
            self.corresponding_flat_node[i]=hmm_node
            self.corresponding_hierarchical_node[hmm_node]=i
            self.states[hmm_node]=1

        # add a state to the flat HMM corresponding to the 2nd level's eof state
        # (necessary for the flattened probabilities to equal 1)
        # for i in production_states:
        #     if i.depth==2: 
        #         eof_node=hierarchicalHMM.get_eof_state(i)
        #         hmm_node=HMM_node('end_state')
        #         self.corresponding_flat_node[eof_node]=hmm_node
        #         self.corresponding_hierarchical_node[hmm_node]=eof_node              
        #         self.states[hmm_node]=0
        #         break


        # copy values from i.horizontal_transitions
        for i in production_states:
                i_flat = self.corresponding_flat_node[i]
                self.transitions[i_flat]={}

                for k in i.horizontal_transitions:
                    if k.type==hhmm.EOF_STATE:
                        continue
                    self.transitions[i_flat][self.corresponding_flat_node[k]]=i.horizontal_transitions[k]


        """=================================== OK UP TO HERE ==================================="""
        for i in self.states:
            if i==self.start or self.states[i]==0:
                continue
            else:
                for j in self.states:
                    if j==self.start:
                        continue
                    if self.corresponding_hierarchical_node[i].parent==self.corresponding_hierarchical_node[j].parent:
                        continue
                    i_to_end = self.corresponding_hierarchical_node[i].horizontal_transitions[self.hierarchicalHMM.get_eof_state(self.corresponding_hierarchical_node[i])]
                    iparent_to_jparent = self.corresponding_hierarchical_node[i].parent.horizontal_transitions[self.corresponding_hierarchical_node[j].parent]
                    jparent_to_j = self.corresponding_hierarchical_node[j].parent.vertical_transitions[self.corresponding_hierarchical_node[j]]
                    self.transitions[i][j] = i_to_end * iparent_to_jparent * jparent_to_j


        # vertical transition probabilities transformed into initial activation probabilities
        # by computing the product of vertical probs from root state to production state
        self.transitions[self.start]={}
        self.transitions[self.start][self.start]=0
        for i in production_states:
            i_flat = self.corresponding_flat_node[i]
            self.transitions[self.start][i_flat]=hierarchicalHMM.ps_to_root(i,1)
            self.transitions[i_flat][self.start]=0
            # self.root.vertical_transitions[i] = self.ps_to_root(i,1)



        # prevent key error problems with the start state
        self.emissions[self.start]={}
        # for note in ['0','1','2','3']:
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

        for t in self.transitions:
            hhmm.normalize(self.transitions[t])

        # get corpus from file
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
            try:
                fwd_table[state][0] = (self.transitions[self.start][state] * self.emissions[state][observation[0]] ) 
            except (KeyError, IndexError) as ke:
                pdb.set_trace()


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
                    try:
                        fwd+=fwd_table[prev_state][output-1] * self.transitions[prev_state][state] * self.emissions[state][observation[output]]
                    except KeyError as ke:
                        pdb.set_trace()
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

    def print_flat(self):
        for x in self.states:
            for y in self.states:
                print self.transitions[x][y] ,
            print ""


    def reflatten(self):

        # pdb.set_trace()

        # make sure self.hierarchicalHMM is minimally self referential
        for internal_node in self.hierarchicalHMM.root.vertical_transitions:
            if internal_node.type==hhmm.INTERNAL_STATE:
                sr=self.hierarchicalHMM.is_SR(internal_node)
                if sr:
                    # print internal_node.horizontal_transitions, "\n\n"
                    self.hierarchicalHMM.convert_to_minSR(internal_node)

        # reflatten the hierarchical HMM
        for i in self.states:
            if i==self.start or self.states[i]==0:
                continue
            else:
                for j in self.states:
                    if j==self.start:
                        continue
                    if self.corresponding_hierarchical_node[i].parent==self.corresponding_hierarchical_node[j].parent:
                        # self.transitions[i][j] = self.corresponding_hierarchical_node[i].horizontal_transitions[self.corresponding_hierarchical_node[j]]
                        continue
                    # if self.corresponding_hierarchical_node[i].note=='72' and self.corresponding_hierarchical_node[i].parent.name=='beginning':
                    #     pdb.set_trace()
                    i_to_end = self.corresponding_hierarchical_node[i].horizontal_transitions[self.hierarchicalHMM.get_eof_state(self.corresponding_hierarchical_node[i])]
                    iparent_to_jparent = self.corresponding_hierarchical_node[i].parent.horizontal_transitions[self.corresponding_hierarchical_node[j].parent]
                    jparent_to_j = self.corresponding_hierarchical_node[j].parent.vertical_transitions[self.corresponding_hierarchical_node[j]]
                    self.transitions[i][j] = i_to_end * iparent_to_jparent * jparent_to_j


        # transition values within the same parent state must be updated
        for i in self.states:
            if i==self.start or self.states[i]==0:
                continue
            else:
                for j in self.states:
                    if j==self.start:
                        continue
                    if self.corresponding_hierarchical_node[i].parent==self.corresponding_hierarchical_node[j].parent:
                        self.transitions[i][j] = self.corresponding_hierarchical_node[i].horizontal_transitions[self.corresponding_hierarchical_node[j]]


        # vertical transition probabilities transformed into initial activation probabilities
        # by computing the product of vertical probs from root state to production state
        production_states=self.hierarchicalHMM.get_pstates(self.hierarchicalHMM.root)

        self.transitions[self.start]={}
        self.transitions[self.start][self.start]=0
        for i in production_states:
            i_flat = self.corresponding_flat_node[i]
            self.transitions[self.start][i_flat]=hierarchicalHMM.ps_to_root(i,1)
            self.transitions[i_flat][self.start]=0
            # self.root.vertical_transitions[i] = self.ps_to_root(i,1)


        for t in self.transitions:
            hhmm.normalize(self.transitions[t])

        # pdb.set_trace()

    def expectation_maximization(self, corpus, convergence, iterations):
        """given a corpus, which is a list of observations, and
        - apply EM to learn the HMM parameters that maximize the corpus likelihood.
        - stop when log likelihood changes less than the convergence threhshold, or the algorithm has completed the specified number of iterations.
        - update self.transitions and self.emissions, and return the log likelihood
        of the corpus under the final updated parameters."""
        prev_log_likelihood=-float('inf')
        epochs=0
        allnodes=self.all_nodes_from_hhmm()
        while (True):
            # pdb.set_trace()
            log_likelihood=0
            print "EM: epoch:",epochs
            trans_counts={}
            pi={}
            end_trans={}
            for i in allnodes:
                trans_counts[i]={}
                end_trans[i]=0
                for j in i.horizontal_transitions:
                    # if j.type==hhmm.EOF_STATE:
                    #     continue
                    trans_counts[i][j]=0# used to be 0

            for i in allnodes:
                if i.type!=hhmm.INTERNAL_STATE:
                    continue
                pi[i]={}
                for j in i.vertical_transitions:
                    if j.type==hhmm.EOF_STATE:
                        continue

                    pi[i][j]=0

            for observation in corpus:
                print "EM: observation:",observation
                alpha = self.forward_algorithm(observation.split())
                beta = self.backward_algorithm(observation.split())
                prob_of_obs = self.total_probability(observation)
                if prob_of_obs==float('-inf'):
                    pdb.set_trace()
                    # return
                print "EM: prob_of_obs:", prob_of_obs
                log_likelihood+=prob_of_obs


                if math.isnan(log_likelihood):
                    pdb.set_trace()
                print "EM: sanity check that alpha==beta:",prob_of_obs==self.total_probability_bk(observation)



                print 'gamma '
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

                # normalize gamma values
                for t in range(len(observation.split())):
                    hhmm.normalize(gamma[t])
                print 'xi'
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

                                # print "alpha[i][t]", alpha[i][t] 
                                # print "beta[i][t+2]",beta[i][t+2]
                                # print "self.emissions[i][j]",sum(self.emissions[i].values())
                                # print "self.emissions[j][observation.split()[t+1]]",self.emissions[j][observation.split()[t+1]]
                                xi[t][self.corresponding_hierarchical_node[i]][self.corresponding_hierarchical_node[j]] = (alpha[i][t] * self.transitions[i][j]  * self.emissions[j][observation.split()[t+1]] * beta[j][t+2])/(10**prob_of_obs)
                                if self.corresponding_hierarchical_node[i].parent!=self.corresponding_hierarchical_node[j].parent:
                                    trans_counts[self.corresponding_hierarchical_node[i]][self.hierarchicalHMM.get_eof_state(self.corresponding_hierarchical_node[i])] += (alpha[i][t] * self.transitions[i][j]  * self.emissions[j][observation.split()[t+1]] * beta[j][t+2])/(10**prob_of_obs)
                                else:
                                    trans_counts[self.corresponding_hierarchical_node[i]][self.corresponding_hierarchical_node[j]] += (alpha[i][t] * self.transitions[i][j]  * self.emissions[j][observation.split()[t+1]] * beta[j][t+2])/(10**prob_of_obs)
                            except (KeyError,IndexError) as ke: 
                                pdb.set_trace()

                            # if epochs==1 and gamma[t][self.corresponding_hierarchical_node[i]]>0.9 and gamma[t+1][self.corresponding_hierarchical_node[j]]>0.9:
                            #     print self.corresponding_hierarchical_node[i].name, '->', self.corresponding_hierarchical_node[j].name
                            #     pdb.set_trace()


                # normalize xi values 
                for t in range(len(observation.split())-1):
                    for i in xi[t]:
                        hhmm.normalize(xi[t][i])
                print 'gamma internal'
                # gamma for internal states
                for t in range(len(observation.split())):
                    for i in allnodes:
                        # skip production states and eof states
                        if i.type!=hhmm.INTERNAL_STATE:
                            continue
                        gamma[t][i] = 0 # used to be 0
                        set_of_i=set(self.sigma(i))
                        for l in set_of_i:
                            gamma[t][i] += gamma[t][l]
                print 'xi internal'
                # xi for internal states
                for t in range(len(observation.split())-1):

                    for i in allnodes:
                        # skip production states and eof states
                        if i.type!=hhmm.INTERNAL_STATE:
                            continue 

                        xi[t][i]={}

                        for j in i.horizontal_transitions:
                            xi[t][i][j]=0 # used to be 0 
                            if j.type==hhmm.EOF_STATE or i==j:
                                continue
                            # if j.type==hhmm.EOF_STATE:
                            #     set_of_i=set(self.sigma(i))
                            #     xi[t][i][j] += gamma[t][i] * i.horizontal_transitions[j]
                            #     trans_counts[i][j] += gamma[t][i] * i.horizontal_transitions[j]
                            #     # for k in set_of_i:
                            #     #     # probability of "emitting" an eof symbol is 1, so don't need to multply that
                            #     #     xi[t][i][j]+=gamma[t][k]*k.horizontal_transitions[self.hierarchicalHMM.get_eof_state(k)]
                            #     #     trans_counts[i][j]+=gamma[t][k]*k.horizontal_transitions[self.hierarchicalHMM.get_eof_state(k)]
                            #     continue

                            set_of_i=set(self.sigma(i))
                            set_of_j=set(self.sigma(j))
                            for k in set_of_i:
                                for l in set_of_j:

                                    xi[t][i][j]+=xi[t][k][l] #* gamma[t][i]
                                    trans_counts[i][j] +=xi[t][k][l] #* gamma[t][i]
                                    pi[l.parent][l] += xi[t][k][l] #* gamma[t][i]

                print 'internal eof states'
                for i in allnodes:
                    if i.type==hhmm.INTERNAL_STATE and i!=self.hierarchicalHMM.root:
                        try:

                            # xi[len(observation.split())-1][i][self.hierarchicalHMM.get_eof_state(i)]=gamma[len(observation.split())-1][i] * i.horizontal_transitions[self.hierarchicalHMM.get_eof_state(i)]
                            trans_counts[i][self.hierarchicalHMM.get_eof_state(i)]=gamma[len(observation.split())-1][i] * i.horizontal_transitions[self.hierarchicalHMM.get_eof_state(i)]
                        except KeyError as e:
                            pdb.set_trace()

                # now re-estimate Tij between all nodes i and j that are not end states

                # # fill pi[i][j] to estimate hierarchical transitions

                # # re-estimate transitions from state i to i's eof state
                # for i in allnodes:

            # normalize trans_counts, pi, and end_trans
            print "normalize trans_counts"
            for i in trans_counts:
                if i==self.hierarchicalHMM.root:
                    continue
                try:
                    hhmm.normalize(trans_counts[i])
                except ZeroDivisionError as e:
                    pdb.set_trace()
            for i in pi:
                try:
                    hhmm.normalize(pi[i])
                except ZeroDivisionError as e:
                    pdb.set_trace()


            # transfer values from trans_counts and pi to the hhmm
            # for i in allnodes:
            #     if i.type!=hhmm.INTERNAL_STATE:
            #         continue
            #     if sum(pi[i].values())==0:
            #         continue
            #     for j in i.vertical_transitions:
            #         if j.type==hhmm.EOF_STATE:
            #             continue
            #         i.vertical_transitions[j] = pi[i][j]

            print "transfer values from trans_counts"
            for i in allnodes:
                if i==self.hierarchicalHMM.root:
                    continue
                if sum(trans_counts[i].values())==0:
                    # print "zero sum we hv a problem"
                    # pdb.set_trace()
                    continue
                for j in i.horizontal_transitions:
                    # print "transferring values..."
                    # if j.type==hhmm.EOF_STATE:
                    #     continue
                    # i.horizontal_transitions[j] += (trans_counts[i][j] - i.horizontal_transitions[j])*0.1
                    i.horizontal_transitions[j] = trans_counts[i][j]

            # for i in allnodes:
            #     if i.type==hhmm.INTERNAL_STATE:
            #         if sum(pi[i].values())==0:
            #             continue
            #         for j in i.vertical_transitions:
            #             if j.type==hhmm.EOF_STATE:
            #                 continue
            #             i.vertical_transitions[j] = pi[i][j]
                # try:
                #     for j in pi[i]:
                #         i.vertical_transitions[j] = pi[i][j]
                # except KeyError as e:
                #     pdb.set_trace()
            #     # if i==root, only copy the new vertical transitions
            #     # if i==self.hierarchicalHMM.root:
            #     #     continue                   
            #     for j in trans_counts[i]:
            #         i.horizontal_transitions[j] = trans_counts[i][j]

            print "EM: prev_log_likelihood-log_likelihood:", prev_log_likelihood-log_likelihood

            epochs+=1
            if (epochs>iterations) or (abs(prev_log_likelihood-log_likelihood) < convergence):
                break

            prev_log_likelihood=log_likelihood

            self.reflatten()
            # pdb.set_trace()
        return log_likelihood


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

        print emission_notes
        # hhmm.write_midi(emission_notes)

if __name__=='__main__':
    print "making hierarchicalHMM..."

    hierarchicalHMM = hhmm.HHMM()

    # create sub-states for beginning, middle, and end,
    # create production states for each note
    parent = hierarchicalHMM.root

    # create sub-states for beginning, middle, and end,
    for i in ['beginning', 'middle', 'end']:
        new_child = hierarchicalHMM.create_child(parent, name=i)
        # create production states for each note
        for note in hhmm.notes:
            hierarchicalHMM.create_child(new_child, internal=False, note=note, name=i+note)
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


    # testing self referential loop stuff
    print "making hierarchicalHMM minimaly self referential..."
    for internal_node in parent.vertical_transitions:
        if internal_node.type==hhmm.INTERNAL_STATE:
            sr=hierarchicalHMM.is_SR(internal_node)
            if sr:
                # print internal_node.horizontal_transitions, "\n\n"
                hierarchicalHMM.convert_to_minSR(internal_node)


    print "converting flattened hierarchicalHMM to normal hmm..."
    hmm = HMM(hierarchicalHMM, 'bach_chorales_cmajor_only.data')

    print "beginning expectation maximization..."
    alpha=hmm.expectation_maximization(hmm.observations[:10],convergence=0.001, iterations=100)
    pdb.set_trace()
    for i in xrange(10):
        hmm.hierarchicalHMM.traverse(hmm.hierarchicalHMM.root,30)
