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
			note_val = str(item).split(' ')[1].split('>')[0]
			print note_val
			print item
			if note_val.quarterLength==1:
				# if the note is a quarter note, just write it
				f.write(' '+note_val)
			elif note_val.quarterLength>1:
				# if the note is more than a quarter note, write in the
				# number of quarter notes that could fit in the note
				temp=note_val.quarterLength
				while temp >= 1:
					f.write(' '+note_val)
					temp-=1

			else: # if the note is less than a quarter note
				# if item.next():
				# 	# if the note is less than a quarter note and 
				# 	if item.next().quarterLength < 1:
				# 		f.write(' '+note_val)
				# 		continue
				# 	else:
				# 		f.write(' '+note_val)
				# else:
				# 	f.write(' '+note_val)
				f.write(' '+note_val)

	f.write(')\n')
	f.close()

for chorale in corpus.chorales.Iterator():
	transposed=chorale.parts[0].flat.transpose('a4',classFilterList=['Note', 'Chord', 'KeySignature'])
	print transposed
	write_to_file(transposed, 'bach_chorales_a4.data')
print "Done!"