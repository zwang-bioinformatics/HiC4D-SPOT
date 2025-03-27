# Author: Bishal Shrestha
# Date: 03-24-2025  
# Description: Randomly sample k lines from a file.

import random
import sys


fileIn = sys.argv[1]
k = int(sys.argv[2])

with open(sys.argv[1], 'rb') as f:
  linecount = sum(1 for line in f)
  f.seek(0)
  #num_lines = int(linecount / k)
  num_lines = k
  random_linenos = sorted(random.sample(range(linecount), num_lines), reverse = True)
  lineno = random_linenos.pop()
  for n, line in enumerate(f):
    if n == lineno:
      print(line.rstrip().decode('utf-8'))
      #print(line)
      if len(random_linenos) > 0:
        lineno = random_linenos.pop()
      else:
        break




