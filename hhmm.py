"""
hhmm.py 
Vikram Narayan
Implements a hierarchical hidden Markov model.
"""

import pdb
import random
import numpy 
import time
import Queue
import music21
import os

PRODUCTION_STATE = 0
INTERNAL_STATE = 1
EOF_STATE=2

# make a dictionary of all possible notes 
notes={}
f=open('bach_chorales_cmajor_only.data')
f2=f.read().split()
for item in f2:
	if item=='\n':
		continue
	else:
		notes[item]=''
f.close()
notes = list(notes)

# notes=["(", "c4","c#4","d4","e-4","e4","f4","f#4","g4","a-4","a4","b-4","b4", ")"]
# notes=['a','b']

def normalize(probability_dictionary):
	"""Assumes the only values in the dictionary are numbers."""
	total_sum = sum(probability_dictionary[key] for key in probability_dictionary) * 1.0
	if total_sum < numpy.finfo(float).eps:
		return probability_dictionary

	for key in probability_dictionary:
		probability_dictionary[key] = probability_dictionary[key]/total_sum 
	return probability_dictionary

def probabilistic_choice(probability_dictionary):
	"""Probabilistically choose from a dictionary of probabilities."""
	# mixed_keys = probability_dictionary.keys()
	# random.shuffle(mixed_keys)
	# n = numpy.random.uniform()
	# for key in mixed_keys:
	# 	weight = probability_dictionary[key]
	# 	if n < weight:
	# 		return key
	# 	n -= weight

	# return mixed_keys[0]
	keys = probability_dictionary.keys()
	r = numpy.random.uniform()
	cum_prob = 0
	for key in keys:
		cum_prob += probability_dictionary[key]
		if r <= cum_prob:
			return key
	return key

def write_midi(sequence, note_type='quarter'):
	"""Writes an emission sequence to a MIDI file using music21."""
	# the types of notes from which to select
	note_types = ['quarter']#, 'half', 'whole', '32nd']
	stream_notes = music21.stream.Stream()

	# currently we default to a 4/4 time sig, this may change in the future
	four_by_four = music21.meter.TimeSignature('4/4')
	stream_notes.append(four_by_four)
	for index,letter in enumerate(sequence):
		if letter==')' or letter=='fermata' or letter=='(':
			continue
			n = music21.note.Note(type='whole')
			n.pitch.midi = int(letter)
			stream_notes.append(n)
		else:
			n = music21.note.Note(type=random.choice(note_types))
			n.pitch.midi = int(letter)
			stream_notes.append(n)
	stream_notes.show()

def write_flattenedHHMM_tofile(hhmm, filename):
	"""write a flattened hhmm to a .hhmmt file (for transitions), and a
	.hhmme file (for emissions). Both are tab separated"""
	if hhmm.flattened:
		f=open(filename+'.hhmmt', 'w')
		# make a node list to ensure a consistent order when the transition 
		# matrix is written
		node_list = list(hhmm.root.vertical_transitions)
		for node1 in node_list:
			for node2 in node_list:
				if node2 in node1.horizontal_transitions:
					f.write(str(node1.horizontal_transitions[node2])+'\t')
				else:
					f.write('0\t')
			f.write('\n')
		f.close()

		# note emissions are writtened to files in the form
		# note,probability\tnote,probability ...etc.
		f2=open(filename+'.hhmme','w')
		for node in node_list:
			for note in node.emissions:
				f2.write(str(note)+','+str(node.emissions[note])+'\t')
			f2.write('\n')
		f2.close()
	else:
		print "Error: HHMM not flattened."

class InternalState:
	def __init__(self, parent, root = False):
		self.type = INTERNAL_STATE
		self.root = root
		self.parent = parent
		self.has_eof = False # does the state below the parent have an end state?
		self.horizontal_transitions = {}
		self.vertical_transitions = {} 
		self.depth=0
		self.name=None
		if root==False:
			self.depth = parent.depth+1

class ProductionState:
	def __init__(self, parent, note):
		self.type = PRODUCTION_STATE
		self.parent = parent
		self.note = note
		self.horizontal_transitions = {}
		self.emissions = {}
		self.emissions[note]=1
		self.depth = parent.depth+1
		self.name=None

class EOFState:
	def __init__(self, parent):
		self.type = EOF_STATE
		self.parent = parent
		self.horizontal_transitions = {}
		self.horizontal_transitions[parent] = 1
		self.depth = parent.depth+1

class HHMM:
	def __init__(self):
		self.root = InternalState(None, True) 
		self.nodes = 1
		self.flattened=False 
		# only certain nodes we need to access later will be available in the node dict
		self.node_dict = {} 

	def create_child(self, parent_node, internal=True, note=None, name=None):
		self.create_eof(parent_node)

		if internal:
			new_node = InternalState(parent_node)
		else:
			new_node = ProductionState(parent_node, note)

		if name:
			new_node.name=name
			self.node_dict[new_node.name]=new_node

		parent_node.vertical_transitions[new_node] = 1
		self.nodes += 1
		return new_node  

	def create_eof(self, parent_node):
		"""if there is not already an EOF state for this substate markov model, make one"""
		if parent_node.has_eof:
			return
		new_node = EOFState(parent_node)
		parent_node.has_eof = True
		parent_node.vertical_transitions[new_node]=0
		self.nodes += 1
		return new_node

	def initialize_horizontal_probs(self, parent_node):
		"""
		Given a parent node, initialize the horizontal_transitions 
		dictionary for each of its children.  
		""" 
		# print "\nSize of vertical_transitions: ", len(parent_node.vertical_transitions)
		# print "vertical_transitions:", parent_node.vertical_transitions

		for child in parent_node.vertical_transitions:
			if child.type == INTERNAL_STATE or child.type == PRODUCTION_STATE: # leave out the eof state for now
				for child2 in parent_node.vertical_transitions:
					child.horizontal_transitions[child2] = 1

				normalize(child.horizontal_transitions)
				# print "\tSize of horizontal_transitions: ", len(child.horizontal_transitions)
				# print "\tchild.horizontal_transitions: ", child.horizontal_transitions

		for child in parent_node.vertical_transitions:
			if child.type == INTERNAL_STATE: 
				self.initialize_horizontal_probs(child)

	def initialize_all_hprobs(self, parent_node):
		"""
		Invoke the initialize_horizontal_probs method to initialize 
		horizontal probabliities for the entire tree. 
		"""
		q = Queue.Queue()
		q.put(parent_node)

		while not q.empty():
			popped = q.get()
			self.initialize_horizontal_probs(popped)

			for child in popped.vertical_transitions:
				if child.type is INTERNAL_STATE:
					q.put(child)

	def traverse(self, node, max_iter=float('inf')):
		"""traverse an hhmm"""
		emission_string=[]
		started=0
		current_node=self.root
		types={0:"production state", 1:"internal state", 2: "eof state"}
		counter=0
		while True:
			# time.sleep(1)
			# print "Current node:", current_node
			# print "Node depth:", current_node.depth
			# print "Node type:", types[current_node.type]
			# print "Parent node:", current_node.parent
			# print "Horizontal Transitions Size:", len(current_node.horizontal_transitions), "\n"
			counter+=1

			if len(emission_string)>max_iter:
				print emission_string
				return emission_string
			if current_node.type==INTERNAL_STATE:
				current_node = probabilistic_choice(current_node.vertical_transitions)
			elif current_node.type==PRODUCTION_STATE:
				emission=probabilistic_choice(current_node.emissions)
				emission_string.append(emission)
				# print "EMISSION:", emission
				current_node = probabilistic_choice(current_node.horizontal_transitions)
			else: # current_node has type EOF_STATE 
				# print "==============EOF_STATE Reached===================="
				current_node = current_node.parent
				if current_node==self.root:
					print emission_string
					return emission_string
					
				else:
					current_node = probabilistic_choice(current_node.horizontal_transitions)

			started=1

	def is_SR(self, internal_node):
		"""
		checks whether an internal node is self referential (can transition to itself)
		"""
		if internal_node in internal_node.horizontal_transitions:
			if internal_node.horizontal_transitions[internal_node] > 0:
				return True
		return False

	def get_eof_state(self, node):
		"""
		Assumes the node has an EOF_STATE node in its horizontal_transitions dictionary 
		"""
		for brother in node.parent.vertical_transitions:
			if brother.type==EOF_STATE:
				return brother

	def convert_to_minSR(self, internal_node):
		"""
		Converts a maximally self referential (maxSR) HHMM to a minSR HHMM.
		Assumes: internal_node is an internal node, and can transition to itself
		"""
		# new horizontal transition probability is: existing transition probability PLUS 
		# prob of going from state i to state j via their parent state
		for child in internal_node.vertical_transitions:
			if child.type==PRODUCTION_STATE:
				for child2 in internal_node.vertical_transitions:
					if child2.type==PRODUCTION_STATE:
						child.horizontal_transitions[child2] = child.horizontal_transitions[child2] + (child.horizontal_transitions[self.get_eof_state(child)] * internal_node.horizontal_transitions[internal_node] * internal_node.vertical_transitions[child2])

		# old probability of an internal state loop has to propagate back through
		# the system
		for child in internal_node.vertical_transitions:
			if child.type==PRODUCTION_STATE:
				child.horizontal_transitions[self.get_eof_state(child)] = child.horizontal_transitions[self.get_eof_state(child)] * (1-internal_node.horizontal_transitions[internal_node])

		# set the probability of internal_node's self referential loop to 0
		internal_node.horizontal_transitions[internal_node]=0

		# normalize the new values
		normalize(internal_node.horizontal_transitions)
		for child in internal_node.vertical_transitions:
			normalize(child.horizontal_transitions)

	def get_pstates(self, node):
		"""return a list of all production states of an hhmm"""
		production_states=[]
		for child in node.vertical_transitions:
			if child.type==PRODUCTION_STATE:
				production_states+=[child]
			elif child.type==INTERNAL_STATE:
				production_states+=self.get_pstates(child)
		return production_states

	def ps_to_root(self, node,current_product):
		"""
		Find product of the vertical probabilities from the root to a production state.
		"""
		if node==self.root:
			return current_product
		else:
			while True:
				current_product = current_product * node.parent.vertical_transitions[node]
				if node.parent==self.root:
					return current_product
				else: 
					node=node.parent
			# return self.ps_to_root(node.parent, node.parent.vertical_transitions[node] * current_product)



	def flatten(self):
		"""
		flattens hhmm into an hmm 
		"""
		production_states=self.get_pstates(self.root)

		# probability of going from production state i to production state j =
		# (i -> EOF state) * (i's parent -> j's parent) * (j's parent -> j)
		for i in production_states:
			for j in production_states:
				if i.parent==j.parent:
					continue
				i_to_eof=i.horizontal_transitions[self.get_eof_state(i)]
				iparent_to_jparent=i.parent.horizontal_transitions[j.parent]
				jparent_to_j=j.parent.vertical_transitions[j]

				i.horizontal_transitions[j] = i_to_eof * iparent_to_jparent * jparent_to_j

		# vertical transition probabilities transformed into initial activation probabilities
		# by computing the product of vertical probs from root state to production state
		for i in production_states:
			self.root.vertical_transitions[i] = self.ps_to_root(i,1)

		# remove internal states to flatten the model
		delete_list=[]
		for state in self.root.vertical_transitions:
			if state.type!=PRODUCTION_STATE:
				delete_list.append(state)
		for item in delete_list:
			del self.root.vertical_transitions[item]

		self.flattened=True

	def traverse_flattened(self):
		"""
		traverses a flattened hhmm.
		"""
		emission_string=[]
		if self.flattened:
			current=probabilistic_choice(self.root.vertical_transitions)
			while True:
				if current.type==PRODUCTION_STATE:
					emission_string.append(current.note)
					current=probabilistic_choice(current.horizontal_transitions)
				elif current.type==EOF_STATE:
					break # this behavior may need to be changed


	def check_probs(self, nodedict):
		"""checks if a node's vertical/horizontal transition dictionaries sum to 1"""
		print sum(nodedict.values())

if __name__ == "__main__":
	hhmm = HHMM()

	# create sub-states for beginning, middle, and end,
	# create production states for each note
	parent = hhmm.root
	for i in ['beginning', 'middle', 'end']:
		new_child = hhmm.create_child(parent, name=i)
		for note in notes:
			hhmm.create_child(new_child, internal=False, note=note)
		hhmm.initialize_horizontal_probs(new_child)
		normalize(new_child.vertical_transitions)

	hhmm.initialize_horizontal_probs(parent)
	normalize(parent.vertical_transitions)

	beginning_node=hhmm.node_dict['beginning']
	middle_node=hhmm.node_dict['middle']
	end_node=hhmm.node_dict['end']

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
	eof_node=hhmm.get_eof_state(end_node)
	end_node.horizontal_transitions[eof_node]=1
	for node in end_node.horizontal_transitions:
		if node==eof_node:
			continue
		end_node.horizontal_transitions[node]=0

	# within each of [beginning, middle, end], p(internalstate->'(')=1
	# i.e., each phrase must begin with a start-of-phrase symbol
	# P(')'->EOF)=1
	for i in [beginning_node, middle_node, end_node]:
		for pstate in hhmm.get_pstates(i):
			eof_node=hhmm.get_eof_state(pstate)
			if pstate.note=='(':
				i.vertical_transitions[pstate]=1
				for node in i.vertical_transitions:
					if node!=pstate:
						i.vertical_transitions[node]=0
			elif pstate.note==')':
				pstate.horizontal_transitions[eof_node]=1
				for node in pstate.horizontal_transitions:
					if node!=eof_node:
						pstate.horizontal_transitions[node]=0
			else:
				pstate.horizontal_transitions[eof_node]=0

	# testing self referential loop stuff
	print "testing self referential loop..."
	for internal_node in parent.vertical_transitions:
		if internal_node.type==INTERNAL_STATE:
			sr=hhmm.is_SR(internal_node)
			if sr:
				# print internal_node.horizontal_transitions, "\n\n"
				hhmm.convert_to_minSR(internal_node)
			else: 
				print internal_node.type

	hhmm.flatten()
	# write_flattenedHHMM_tofile(hhmm,'hhmm_flattened')
	# print "STARTING TRAVERSAL"
	# emissions=hhmm.traverse(hhmm.root)
	# write_midi(emissions)
