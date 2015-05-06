import pdb
f=open('bach_chorales_cmajor_aminor_midi.data','r')
l=[]
i=0
print "[ ",
maxlen=10
for line in f:

    line=line.split()
    if len(line) < maxlen:
        continue

    for j in xrange(maxlen):
        if line[j]=='(' or line[j]==')' or line[j]=='fermata':
            print '1', ' ',
            continue
        print line[j], " ",
    print ";",
    i+=1
    if i > 24:
        break
print "]"
f.close()

# x=numpy.array(l)
# mmat(x)
