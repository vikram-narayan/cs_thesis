from music21 import *
import hhmm
"""
s=stream.Stream()
n=note.Note('a4', type='32nd')
s.append(meter.TimeSignature('4/4'))
s.append(n)
s.show()

paths=corpus.getComposer('bach')


c = converter.parse('tinynotation: 4/4 c4 d e f')
c.show('text')
fp = converter.freeze(c, fmt='pickle')
print fp
data1 = [('g4', 'quarter'), ('a4', 'quarter'), ('b4', 'quarter'), ('c#5', 'quarter')]
data2 = [('d5', 'whole')]
data = [data1, data2]
partUpper = stream.Part()
for mData in data:
    m = stream.Measure()
    for pitchName, durType in mData:
        n = note.Note(pitchName)
        n.duration.type = durType
        m.append(n)
    partUpper.append(m)

A = key.KeySignature(3)

for m in partUpper:
    for n in m:
        print "\t",type(n)
        # x=n.transpose('g4')
        # print "\t\t===========",x
        # n.transpose('g4',inPlace=True)
print type(partUpper)
partUpper.flat.transpose('a4').show()

partUpper.show()
"""
import pdb
seqs=[]
f = open('matlab25_chorales.data','r')

for line in f:
    temp=[]
    line=line.split()
    for n in line:
        temp.append(int(n))
    seqs.append(temp)

f.close()

smax=0
for s in seqs:
    if len(s)>smax:
        smax=len(s)

for s in seqs:
    if len(s)<smax:
        while len(s)!=smax:
            s.append(1)

print "making seq variable for matlab"



print "making note dictionary (used as states)"
ndict={}
for s in seqs:
    for i in s:
        ndict[i]=0


matlab_dict={}
matlab_reverse={}
counter=1
for n in ndict:
    matlab_dict[n]=counter
    matlab_reverse[counter]=n
    counter+=1


def back_to_midi(note_seq):
    new_seq=[]
    for n in note_seq:
        if matlab_reverse[n]==1:
            hhmm.write_midi(new_seq)
            return
        elif matlab_reverse[n]==0:
            continue
        new_seq.append(matlab_reverse[n])
    hhmm.write_midi(new_seq)

"""
for s in seqs:
    for x in s:
        print matlab_dict[x], ' ',
    print ';',''

print '\n\n\n\n'

for n in ndict:
    for n2 in ndict:
        print float(1)/len(ndict), ' ',
    print ';',''
"""

