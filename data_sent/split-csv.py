import csv

csv.register_dialect('tabs', delimiter='\t')
with open('simple.aligned', 'rb') as s:
	f = open('simple-aligned', 'w')
	csvreader = csv.reader(s, dialect='tabs')
	for row in csvreader:
		f.write(row[2]+'\n')
	f.close()

with open('normal.aligned', 'rb') as s:
        f = open('normal-aligned', 'w')
        csvreader = csv.reader(s, dialect='tabs')
        for row in csvreader:
                f.write(row[2]+'\n')
        f.close()

