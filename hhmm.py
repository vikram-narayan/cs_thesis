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

PRODUCTION_STATE = 0
INTERNAL_STATE = 1
EOF_STATE=2

# notes=["(", "c","c_#","d","e_b","e","f","f_#","g","a_b","a","b_b","b", ")"]
notes=["(", "c4","c#4","d4","e-4","e4","f4","f#4","g4","a-4","a4","b-4","b4", ")"]
# notes=['a','b']

def normalize(probability_dictionary):
	"""
	Assumes the only values in the dictionary are numbers.
	"""
	total_sum = sum(probability_dictionary[key] for key in probability_dictionary) * 1.0
	for key in probability_dictionary:
		probability_dictionary[key] = probability_dictionary[key]/total_sum 

def probabilistic_choice(probability_dictionary):
	mixed_keys = probability_dictionary.keys()
	random.shuffle(mixed_keys)
	n = numpy.random.uniform()
	for key in mixed_keys:
		weight = probability_dictionary[key]
		if n < weight:
			return key
		n -= weight

	return mixed_keys[0]

def write_midi(sequence, note_type='quarter'):
	"""
	Writes an emission sequence to a MIDI file using music21.
	"""
	note_types = ['quarter', 'half', 'whole', '32nd']
	stream_notes = music21.stream.Stream()
	four_by_four = music21.meter.TimeSignature('4/4')
	stream_notes.append(four_by_four)
	for letter in sequence:
		if letter=='(' or letter==')':
			continue
		n = music21.note.Note(letter, type=random.choice(note_types))
		stream_notes.append(n)
	stream_notes.show()

class InternalState:
	def __init__(self, parent, root = False):
		self.type = INTERNAL_STATE
		self.root = root
		self.parent = parent
		self.has_eof = False # does the state below the parent have an end state?
		self.horizontal_transitions = {}
		self.vertical_transitions = {} 
		self.depth=0
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

	def create_child(self, parent_node, internal=True, note=None):
		self.create_eof(parent_node)

		if internal:
			new_node = InternalState(parent_node)
		else:
			new_node = ProductionState(parent_node, note)

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


	def traverse(self, node):

		emission_string=[]
		started=0
		current_node=self.root
		while True:
			# time.sleep(1)
			print "Current node:", current_node
			print "Node depth:", current_node.depth
			types={0:"production state", 1:"internal state", 2: "eof state"}
			print "Node type:", types[current_node.type]
			print "Parent node:", current_node.parent
			print "Horizontal Transitions Size:", len(current_node.horizontal_transitions), "\n"
			if current_node.type==INTERNAL_STATE:
				current_node = probabilistic_choice(current_node.vertical_transitions)
			elif current_node.type==PRODUCTION_STATE:
				emission=probabilistic_choice(current_node.emissions)
				emission_string.append(emission)
				print "EMISSION:", emission
				current_node = probabilistic_choice(current_node.horizontal_transitions)
			else: # current_node has type EOF_STATE 
				print "==============EOF_STATE Reached===================="
				current_node = current_node.parent
				if current_node==self.root:
					print emission_string
					return emission_string
				else:
					current_node = probabilistic_choice(current_node.horizontal_transitions)

			started=1

	def flatten(self):
		"""
		Flatten the hhmm according to the following rules:
		"""
		


if __name__ == "__main__":
	hhmm = HHMM()

	# create sub-states for beginning, middle, and end,
	# create production states for each note
	parent = hhmm.root
	for i in xrange(3):
		new_child = hhmm.create_child(parent)
		for note in notes:
			hhmm.create_child(new_child, internal=False, note=note)
		normalize(new_child.vertical_transitions)

	normalize(parent.vertical_transitions)
	hhmm.initialize_horizontal_probs(parent)

	# for child in parent.vertical_transitions:
	# 	normalize(child.horizontal_transitions)
	# 	print child.horizontal_transitions, "\n\n"

	print "STARTING TRAVERSAL"
	emissions=hhmm.traverse(hhmm.root)
	write_midi(emissions)
