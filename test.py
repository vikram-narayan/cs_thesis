from music21 import *
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
"""
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

