import numpy as np
import gensim
import MeCab

model = gensim.models.Word2Vec.load("./word_vectors/ja.bin")
mecab = MeCab.Tagger()

def get_vector_average(vectors):
  return np.average(vectors, axis=0)

def get_vercors_from_japanese(text):
  # split words
  words = mecab.parse(text).split("\n")[:-2]
  vectors = []

  for word in words:
    word_body, description = word.split("\t")
    description_array = description.split(",")
    
    # exclude invalid method
    if description_array[0] in ["助詞", "記号", "助動詞"]:
      continue
    
    try:
      vector = model[word_body]
      vectors.append(vector)
    except:
      print("I don't know about:", word)
  
  return np.array(vectors)
