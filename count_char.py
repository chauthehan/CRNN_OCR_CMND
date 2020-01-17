import os

maxi = 0
for link in os.listdir("/home/hansama/Documents/crnn/data"):
	if link.endswith(".txt"):
		with open("/home/hansama/Documents/crnn/data/" + link, 'r') as f:
			for line in f:
				if len(line) > maxi:
					print(maxi)
					maxi = len(line)
		f.close()

print(maxi)

