"""
Home work for MSAN 501: Word Similarity and Relationships - Plotting the words
Si Chen <schen90@usfca.edu>
"""

"""
How to run:
- Make sure place this file under the same folder together with wordsim.py
- This script accepts one argument which is path to the 300-d data file.
- Recommend to use the python executable installed together with Anaconda, because
  Anaconda has most of the referenced libraries installed. But if your generic python
  has the following libraries installed, this script will run properly too.
"""

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import wordsim
import sys

"""
Get a list of vectors for n similar words for each w in words.
Flatten that into a single list of vectors, including the original
words' vectors.  Compute a word vector for each of those words and
put into a list. Use PCA to project the vectors onto two dimensions.
Extra separate X and Y coordinate lists and pass to matplotlib's scatter
function. Then, iterate through the expanded word list and plot the
string using text() with, say, fontsize=16. call show().
"""
def plot_words(gloves, words, n):
  if n <= 0:
    print "ERROR: Wrong input of expected number of returned words: ", n
    return -1

  full_word_list = []

  for word in words:
    # Get the list of the closest words of the given word
    list_of_words = wordsim.closest_words(gloves, word, n)
    # Add the word itself to the list
    list_of_words.append(word)
    # connect the list got in each iteration together
    full_word_list += list_of_words

  # Compute vectors
  vector_list = []
  for word in full_word_list:
    vector = gloves[word]
    vector_list.append(vector)

  # Convert the high dimentional vector to 2 dimentional
  pca = PCA(n_components=2)
  vecs2D = pca.fit_transform(vector_list)
  x = vecs2D[:,0]
  y = vecs2D[:,1]
  plt.scatter(x, y)
  for i, vector2d in enumerate(vecs2D):
    plt.text(vector2d[0], vector2d[1], full_word_list[i], fontsize=16)
  plt.show()

if __name__ == '__main__':
  if len(sys.argv) < 2:
    wordsim.print_help()
    exit(1)

  glove_filename = sys.argv[1]
  gloves = wordsim.load_glove(glove_filename)

  # Test case #1, per the tutorial
  words = ['king', 'queen', 'cat']
  plot_words(gloves, words, 4)

  # Test case #2, per the tutorial
  # The rendered result like the 180-degree rotation of the tutorial's,
  # which is kind of interesting. Because this test case uses the same
  # method as test case #2 which renders the same result as the tutorial.
  words = ['petals', 'software', 'car']
  plot_words(gloves, words, 4)
