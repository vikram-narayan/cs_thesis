"""
hhmm.py 
Vikram Narayan
Creates a corpus of the bach chorale melodies.
"""
from music21 import *
import time
import pdb
import copy

def write_to_file(transposed_chorale, file_name):
	f=open(file_name, 'a')
	f.write('(')
	for item in transposed_chorale:
		if type(item)==note.Note:
			ns = str(item).split(' ')[1].split('>')[0]
			f.write(' '+ns)

	f.write(')\n')
	f.close()

for chorale in corpus.chorales.Iterator():
	transposed=chorale.parts[0].flat.transpose('a4',classFilterList=['Note', 'Chord', 'KeySignature'])
	print transposed
	write_to_file(transposed, 'bach_chorales_a4.data')
print "Done!"