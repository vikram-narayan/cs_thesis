"""
train2.py
Vikram Narayan
Trains sub-HMMs within a HHMM as their own separate HMMs, combines them separately (meant for the poster presentation). 
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



def read_corpus(filename):
    """reads corpus file"""
    observations=[]
    f=open(filename, 'r')
    for line in f:
        observations.append(line[:len(line)-1])
    f.close()
    return observations


class HMM:
    def __init__(self, filename):
        """converts a hierarchical HMM to a flat HMM"""

        self.transitions={}
        self.emissions={}
        self.states=['0','1','2','3','#']
        # self.states=hhmm.notes+['#']

        # initialize transition probabilities
        for state in self.states:
            self.transitions[state]=self.initialize_transition_probs_random()
            # self.transitions[state]['(']=0
                # if state2=='#':
                #     self.transitions[state][state2]=0
                # self.transitions[state][state2]=1.0/(len(self.states)-1)

        for state in self.states:
            hhmm.normalize(self.transitions[state])

        for state in self.states:
            # self.emissions[state]=self.initialize_emission_probs(state,0.9)
            self.emissions[state]=self.initialize_emission_probs_random()
            # for note in hhmm.notes:
            #     self.emissions[state][note]=1
            # hhmm.normalize(self.emissions[state])
        self.emissions['#'] = self.initialize_emission_probs('(',0.99)

        # get corpus from file
        self.observations = read_corpus(filename)


    def initialize_emission_probs(self, note, prob_of_note):
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

    def initialize_emission_probs_random(self):
        emission_probs={}
        remainder=1
        for n in hhmm.notes:
            emission_probs[n]=random.uniform(0,remainder)
            remainder = remainder - emission_probs[n]
        hhmm.normalize(emission_probs)
        return emission_probs

    def initialize_transition_probs_random(self):
        transition_probs={}
        remainder=1
        for n in self.states:
            transition_probs[n]=random.uniform(0,remainder)
            remainder = remainder - transition_probs[n]
        hhmm.normalize(transition_probs)
        return transition_probs

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
            viterbi_table[state][0] = numpy.log10(self.transitions['#'][state] * self.emissions[state][observation[0]] ) 
            back_pointers[state][0]='#'

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
            fwd_table[state][0] = (self.transitions['#'][state] * self.emissions[state][observation[0]] ) 

        # fill in the rest of the forward table
        for output in range(1,len(observation)):
            for state in self.states:
                fwd=0
                for prev_state in self.states:
                    # print "state in fwd_table",state in fwd_table
                    # print "prev_state in self.transitions",prev_state in self.transitions
                    # print "state in self.transitions[prev_state]",state in self.transitions[prev_state]
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

        fwd_table = self.forward_algorithm(observation)

        # forward_prob = numpy.log10(numpy.prod(fwd_table['scaling factor']))

        forward_prob=0
        for state in self.states:
            forward_prob+=fwd_table[state][len(observation)-1]

        return numpy.log10(forward_prob)

    def forward_algorithm_scale(self, observation):
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
            normalizer=0
            for state2 in self.states:
                normalizer+=(self.transitions['#'][state2] * self.emissions[state][observation[0]] )
                # print self.transitions['#'][state2] , ",", self.emissions[state][observation[0]] , ",", observation[0]
                # print state, state2
                print state, state2

            # logs will be taken at the end 
            fwd_table[state][0] = (self.transitions['#'][state] * self.emissions[state][observation[0]] ) / normalizer

        # fill in the rest of the forward table
        for output in range(1,len(observation)):
            for state in self.states:
                fwd=0
                for prev_state in self.states:
                    fwd+=fwd_table[prev_state][output-1] * self.transitions[prev_state][state] 

                denominator1=0
                for k in self.states:
                    denominator1+=self.emissions[k][observation[output]]
                    denominator2=0
                    for j in self.states:
                        denominator2+=fwd_table[j][output-1] * self.transitions[j][k] 

                    denominator1 = denominator1 * denominator2

                fwd_table['scaling factor'][output] = 1/denominator1
                fwd_table[state][output] = (fwd * self.emissions[state][observation[output]])/denominator1

        return fwd_table

    def total_probability_scale(self, observation):
        """compute the probability of the observation under the model"""

        fwd_table = self.forward_algorithm_scale(observation)

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
        observation2 = ['#']+observation
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
            back+= self.transitions['#'][state] * self.emissions[state][observation2[1]] * bk_table[state][1]
        for state in self.states:
            bk_table[state][0]=back

        return bk_table

    def backward_algorithm_scale(self, observation):
        """given an observation as a list of symbols,
        find the probability of the observation under this HMM,
        using the backward algorithm"""
        # initialize backward algorithm table
        fwd_table = self.forward_algorithm_scale(observation)
        bk_table={}
        bk_table['scaling factor']=[]
        observation2 = ['#']+observation
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
                bk_table[state][output] = back * fwd_table['scaling factor'][output+1]

            output=output-1

        back=0
        for state in self.states:
            back+= self.transitions['#'][state] * self.emissions[state][observation2[1]] * bk_table[state][1]
        for state in self.states:
            bk_table[state][0]=back

        return bk_table


    def test(self, corpus):


        for observation in corpus:
            observation = observation.split()
            alpha_scale = self.forward_algorithm_scale(observation)
            beta_scale = self.backward_algorithm_scale(observation)
            alpha = self.forward_algorithm(observation)
            beta = self.backward_algorithm(observation)

            gamma ={}
            gamma_scale={}
            for state in self.states:
                gamma[state]=[]
                gamma_scale[state]=[]
                for t in range(len(observation)-1):
                    gamma[state].append(0)
                    gamma_scale[state].append(0)


            for t in range(len(observation)-1):

                for i in self.states:
                    gamma_denominator=0
                    for j in self.states:
                        gamma_denominator += alpha[j][t] * beta[j][t+1]
                    gamma[i][t]  = (alpha[i][t] * beta[i][t+1])/gamma_denominator

                    gamma_denominator_scale=0
                    for j in self.states:
                        gamma_denominator_scale += alpha_scale[j][t] * beta_scale[j][t+1]
                    gamma_scale[i][t]  = (alpha_scale[i][t] * beta_scale[i][t+1])/gamma_denominator_scale

                    print gamma[i][t], gamma_scale[i][t], gamma[i][t]==gamma_scale[i][t]


    def total_probability_bk(self, observation):
        """compute the probability of the observation under the model"""

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
        # -- TODO -- fill in EM implementation
        prev_log_likelihood=-float('inf')
        # for counter in range(100):
        while (True):
            log_likelihood=0

            # store emission soft counts
            soft_count={}
            for state in self.states:
                soft_count[state]={}
                for i in self.states:
                    soft_count[state][i]=0

            # store soft counts for transitions
            soft_count_trans={}
            soft_count_trans['#']={}
            for state in self.states:
                soft_count_trans[state]={}
                soft_count_trans['#'][state]=0
                for state2 in self.states:
                    soft_count_trans[state][state2]=0

            for observation in corpus:
                observation=observation.split()
                # print observation
                total_prob = self.total_probability(observation)
                # print "sanity check: total_prob=total_probability_bk", total_prob,self.total_probability_bk(observation),total_prob==self.total_probability_bk(observation)
                log_likelihood+=total_prob
                fwd_matrix = self.forward_algorithm(observation)
                bk_matrix = self.backward_algorithm(observation)

                # new_emissions stores the counts for observation
                new_emissions = {}
                # new_transitions={}

                for state in self.states:
                    new_emissions[state]=[]
                    for i in range(len(observation)):
                        new_emissions[state].append(0)

                # emission soft counts
                for i in range(len(observation)):
                    for state in self.states:

                        new_emissions[state][i] = fwd_matrix[state][i] * bk_matrix[state][i+1] 
                        new_emissions[state][i] = new_emissions[state][i]/(10**total_prob)
                        if soft_count[state].has_key(observation[i]):
                            soft_count[state][observation[i]]+=new_emissions[state][i]
                        else:
                            soft_count[state][observation[i]]=new_emissions[state][i]

                # transition soft counts
                for i in range(len(observation)-1):
                    for state in self.states:
                        for state2 in self.states:
                            soft_count_trans[state][state2]+=(fwd_matrix[state][i] * self.transitions[state][state2] * self.emissions[state2][observation[i+1]] * bk_matrix[state2][i+2])/(10**total_prob)

                # update transition probabilities from start
                for state in self.states:
                    soft_count_trans['#'][state]+= (self.transitions['#'][state] * self.emissions[state][observation[0]]* bk_matrix[state][1])/(10**total_prob) 
                    #self.transitions['#'][state]* self.emissions[state][observation[0]] # self.transitions['#'][state] #* self.emissions[state][observation[1]] * bk_matrix[state][2]
                # bss = self.best_state_sequence(observation)
                # for state in self.states:
                #     # if bss[0]==state:
                #     soft_count_trans['#'][state]+=total_prob * self.emissions[state][observation[0]]

            #normalize emission soft counts
            for state in self.states:
                running_sum=0
                for letter in soft_count[state]:
                    running_sum+=soft_count[state][letter]
                for letter in soft_count[state]:
                    soft_count[state][letter] =soft_count[state][letter]/running_sum

            #update emission probabilities
            for state in self.states:
                for letter in soft_count[state]:
                    if soft_count[state][letter]!=0:
                        self.emissions[state][letter] = soft_count[state][letter]

            #normalize transition soft counts
            for state in self.states:
                running_sum=0
                for state2 in self.states:
                    running_sum+= soft_count_trans[state][state2]
                for state2 in self.states:
                    soft_count_trans[state][state2] = soft_count_trans[state][state2]/running_sum

            running_sum=0
            for state in self.states:
                running_sum+=soft_count_trans['#'][state]
            for state in self.states:
                soft_count_trans['#'][state] = soft_count_trans['#'][state]/running_sum
            
            #update transition probabilities
            for state in self.states:
                for state2 in self.states:
                    self.transitions[state][state2] = soft_count_trans[state][state2]

            for state in self.states:
                self.transitions['#'][state] =soft_count_trans['#'][state]
            
            print '\t',log_likelihood-prev_log_likelihood

            if (log_likelihood - prev_log_likelihood) < convergence:
                return log_likelihood

            prev_log_likelihood=log_likelihood

        # -- TODO end
        return log_likelihood


    def expectation_maximization_scale(self, corpus, convergence, iterations):
        """given a corpus, which is a list of observations, and
        apply EM to learn the HMM parameters that maximize the corpus likelihood.
        stop when the log likelihood changes less than the convergence threhshold.
        update self.transitions and self.emissions, and return the log likelihood
        of the corpus under the final updated parameters."""
        # -- TODO -- fill in EM implementation
        prev_log_likelihood=-float('inf')
        # for counter in range(100):
        while (True):
            log_likelihood=0

            # store emission soft counts
            soft_count={}
            for state in self.states:
                soft_count[state]={}
                for i in self.states:
                    soft_count[state][i]=0

            # store soft counts for transitions
            soft_count_trans={}
            soft_count_trans['#']={}
            for state in self.states:
                soft_count_trans[state]={}
                soft_count_trans['#'][state]=0
                for state2 in self.states:
                    soft_count_trans[state][state2]=0

            for observation in corpus:
                observation=observation.split()
                # print observation
                total_prob = self.total_probability(observation)
                # print "\tsanity check: total_prob=total_probability_bk", total_prob,self.total_probability_bk(observation),total_prob==self.total_probability_bk(observation)
                log_likelihood+=total_prob
                fwd_matrix = self.forward_algorithm_scale(observation)
                bk_matrix = self.backward_algorithm_scale(observation)
                # new_emissions stores the counts for observation
                new_emissions = {}
                # new_transitions={}

                for state in self.states:
                    new_emissions[state]=[]
                    for i in range(len(observation)):
                        new_emissions[state].append(0)

                # emission soft counts
                for i in range(len(observation)):
                    for state in self.states:

                        new_emissions[state][i] = fwd_matrix[state][i] * bk_matrix[state][i+1] 
                        new_emissions[state][i] = new_emissions[state][i]/(10**total_prob)
                        if soft_count[state].has_key(observation[i]):
                            soft_count[state][observation[i]]+=new_emissions[state][i]
                        else:
                            soft_count[state][observation[i]]=new_emissions[state][i]

                # transition soft counts
                for i in range(len(observation)-1):
                    for state in self.states:
                        for state2 in self.states:
                            soft_count_trans[state][state2]+=(fwd_matrix[state][i] * self.transitions[state][state2] * self.emissions[state2][observation[i+1]] * fwd_matrix['scaling factor'][observation[i+1]] * bk_matrix[state2][i+2])/(10**total_prob)

                # update transition probabilities from start
                for state in self.states:
                    soft_count_trans['#'][state]+= (self.transitions['#'][state] * self.emissions[state][observation[0]]* bk_matrix[state][1])/(10**total_prob) 
                # bss = self.best_state_sequence(observation)
                # for state in self.states:
                #     # if bss[0]==state:
                #     soft_count_trans['#'][state]+=total_prob * self.emissions[state][observation[0]]

            #normalize emission soft counts
            for state in self.states:
                running_sum=0
                for letter in soft_count[state]:
                    running_sum+=soft_count[state][letter]
                for letter in soft_count[state]:
                    soft_count[state][letter] =soft_count[state][letter]/running_sum

            #update emission probabilities
            for state in self.states:
                for letter in soft_count[state]:
                    if soft_count[state][letter]!=0:
                        self.emissions[state][letter] = soft_count[state][letter]

            #normalize transition soft counts
            for state in self.states:
                running_sum=0
                for state2 in self.states:
                    running_sum+= soft_count_trans[state][state2]
                for state2 in self.states:
                    soft_count_trans[state][state2] = soft_count_trans[state][state2]/running_sum

            running_sum=0
            for state in self.states:
                running_sum+=soft_count_trans['#'][state]
            for state in self.states:
                soft_count_trans['#'][state] = soft_count_trans['#'][state]/running_sum
            
            #update transition probabilities
            for state in self.states:
                for state2 in self.states:
                    self.transitions[state][state2] = soft_count_trans[state][state2]

            for state in self.states:
                self.transitions['#'][state] =soft_count_trans['#'][state]
            
            print '\t',log_likelihood-prev_log_likelihood

            if (log_likelihood - prev_log_likelihood) < convergence:
                return log_likelihood

            prev_log_likelihood=log_likelihood

        # -- TODO end
        return log_likelihood


    def generate(self):
        """after an hmm has been trained, use it to generate songs
        REWRITE THIS"""
        print "beginning generation"

        # normalize just to be sure
        for t in self.transitions:
            hhmm.normalize(self.transitions[t])
        for e in self.emissions:
            hhmm.normalize(self.emissions[e])


        current='#'
        emission_notes=[]
        current = hhmm.probabilistic_choice(self.transitions[current])
        emission_notes.append(hhmm.probabilistic_choice(self.emissions[current]))

        length = random.randint(8,18)
        i=0
        while True:
            current = hhmm.probabilistic_choice(self.transitions[current])
            emission_notes.append(hhmm.probabilistic_choice(self.emissions[current]))
            i+=1
            if current==')' or i>length:
                break

        return emission_notes
        hhmm.write_midi(emission_notes)

if __name__=='__main__':
    import sys
    # flat = HMM('bach_chorales_cmajor_only.data')
    # alpha = flat.expectation_maximization(flat.observations[:25], convergence=0.001, iterations=300)
    # pdb.set_trace()
    # sys.quit()

    beginning = HMM('beginning.data')
    middle = HMM('middle.data')
    end = HMM('end.data')


    print "training beginning hmm..."
    alpha=beginning.expectation_maximization(beginning.observations[:50],convergence=0.01, iterations=200)

    print "training middle hmm..."
    alpha=middle.expectation_maximization(middle.observations[:50],convergence=0.01, iterations=200)

    print "training end hmm..."
    alpha=end.expectation_maximization(end.observations[:50],convergence=0.01, iterations=200)

    # alpha=hmm.expectation_maximization_scale(hmm.observations[:3],convergence=0.1, iterations=200)

    for i in xrange(5):
        emission_seq=[]
        emission_seq += beginning.generate()
        max_middle=0
        while True: 
            emission_seq+=middle.generate()
            if random.uniform(0,1) > 0.8390 or max_middle==10:
                break
            max_middle+=1
        emission_seq += end.generate()
        pdb.set_trace()
        hhmm.write_midi(emission_seq)

    # for i in xrange(4):
    #     beginning.generate()
    # for i in xrange(4):
    #     middle.generate()
    # for i in xrange(4):
    #     end.generate()


    # for obs in hmm.observations[:5]:
    #     obs=obs.split()
    #     print hmm.total_probability(obs), hmm.total_probability_bk(obs),hmm.total_probability(obs)==hmm.total_probability_bk(obs)
    # for i in xrange(4):
    #     hmm.generate()
