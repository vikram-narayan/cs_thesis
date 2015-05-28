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
	write_list=[]
	write_list.append('(')
	for index,item in enumerate(transposed_chorale):
		if type(item)==note.Note:
			note_val = str(item).split(' ')[1].split('>')[0]

			# comment out to have the file consist of note names rather than MIDI values
			note_val = str(item.pitch.midi) 
			print note_val
			print item
			if item.quarterLength==1:
				# if the note is a quarter note, just write it
				write_list.append(note_val)
			elif item.quarterLength>1:
				# if the note is more than a quarter note, write in the
				# number of quarter notes that could fit in the note
				temp=item.quarterLength
				while temp >= 1:
					write_list.append(note_val)
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
				write_list.append(' '+note_val)
			if len(item.expressions)>0:
				for exp in item.expressions:
					if type(exp)==expressions.Fermata:
						write_list.append(')')
						write_list.append('(')
	write_list.append(' )')
	last_index = len(write_list)-1
	second_to_last_index=last_index-1
	if write_list[second_to_last_index]=='(' and write_list[last_index]==')':
		write_list = write_list[:second_to_last_index]

	for x in write_list:
		f.write(str(x)+' ')
	f.write('\n')
	f.close()

def make_note_dict(filename):
	"""make a dictionary of all notes """
	notes={}
	f=open(filename)
	f2=f.read().split()
	for item in f2:
		if item=='\n':
			continue
		else:
			notes[item]=''
	f.close()
	notes = list(notes)

if __name__=='__main__':
	# dicts specify the number of half steps to transpose a key into c major/a minor
	majors = dict([("A-", 4),("A", 3),("B-", 2),("B", 1),("C", 0),("D-", -1),("D", -2),("E-", -3),("E", -4),("F", -5),("G-", 6),("G", 5)])
	minors = dict([("A-", 1),("A", 0),("B-", -1),("B", -2),("C", -3),("D-", -4),("D", -5),("E-", 6),("E", 5),("F", 4),("F#",3),("G-", 3),("G", 2)])

	# data_filename = 'bach_chorales_cmajor_aminor.data'
	data_filename = 'bach_chorales_cmajor_only.data'

	# clear any previous contents of the file
	f=open(data_filename,'w')
	f.close()

	for chorale in corpus.chorales.Iterator():
		# transpose the soprano part (the melody) to c major or a minor 
		ksig = chorale.parts[0].analyze('key')
		print ksig.tonic.name, ksig.mode

		if ksig.mode=='major':
			transposed=chorale.parts[0].flat.transpose(majors[ksig.tonic.name])
		else:
			continue
			transposed=chorale.parts[0].flat.transpose(minors[ksig.tonic.name])

		# transposed=chorale.parts[0].flat.transpose('a4',classFilterList=['Note', 'Chord', 'KeySignature'])
		print "transposed to ", transposed.analyze('key')

		write_to_file(transposed, data_filename)
	print "Done!"