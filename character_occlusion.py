import numpy as np
import string
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import itertools
import functools
from functools import reduce
import csv
import random

su=0
freqs = {}
with open('words.csv', newline='') as csvfile:

    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in spamreader:
      if(row[1]!='count'):
        su+=int(row[1])
    print(su)
with open('words.csv', newline='') as csvfile:

    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    i=0
    for row in spamreader:
      i+=1
      if i>100000:
        break
      if(row[1]!='count'):
        freqs[row[0]] = int(row[1])/su
print(len(freqs))


# 14-segment digital display representations. 

# These are the representations according to the GIF above, 
# but keep in mind that there might be multiple possible representations
# for a given character. 

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# in alphabetical order
strokes = [
    {1, 2, 6, 7, 8, 9, 13}, #A
    {1,4,6,8,11,13,14},     #B
    {1,2,9,14},             #C
    {1,4,6,11,13,14},       #D
    {1,2,7,8,9,14},         #E
    {1,2,7,8,9},            #F
    {1,2,9,13,14},          #G
    {2,6,7,8,9,13},         #H
    {1,4,11,14},            #I
    {6,9,13,14},            #J
    {2,5,7,9,12},           #K
    {2,9,14},               #L
    {2,3,5,6,9,13},         #M
    {2,3,6,9,12,13},        #N
    {1,2,6,9,13,14},        #O
    {1,2,6,7,8,9},          #P
    {1,2,6,9,12,13,14},     #Q
    {1,2,5,7,9,12},         #R
    {1,3,8,13,14},          #S
    {1,4,11},               #T
    {2,6,9,13,14},          #U
    {2,5,9,10},             #V
    {2,6,9,10,12,13},       #W
    {3,5,10,12},            #X
    {3,5,11},               #Y
    {1,5,10,14}             #Z
]


# convert a set of segment indices (e.g. {3,5,11} to a 14-dim binary array format)
def display_to_rep(display):
    a = np.zeros(14)
    for i in display:
        a[i-1] = 1
    return a



# convert a letter to a 14-dim binary array format
def get_letter_rep(letter):
    if letter not in letters:
        raise ValueError("Not a valid letter.")
    idx = letters.index(letter)
    strokelist = strokes[idx]
    a = np.zeros(14)
    for i in strokelist:
        a[i-1] = 1
    return a

get_letter_rep("Z")



# convert letter to a set of segment indices, e.g. Y -> {3,5,11}
def get_letter_display(letter):
    if letter not in letters:
        raise ValueError("Not a valid letter.")
    idx = letters.index(letter)
    return strokes[idx]

# apply a mask to a representation of a letter 
def mask_char(rep, mask):
    #print(mask)
    mask_arr = np.ones(14)
    for i in mask:
        mask_arr[i-1] = 0
    return rep * mask_arr

# different types of occlusion masks
mask_bottom = {7,8,9,10,11,12,13,14}
mask_top = {1,2,3,4,5,6,7,8}
mask_right = {4,5,6,8,11,12,13}
mask_left = {2,3,4,7,9,10,11}

# masking the most used segment
mask_most_used = {9}

# masking just the top or bottom line segment (not the whole top or bottom half)
mask_bottom_line = {14}
mask_top_line ={1}

# can define addition masks below ... 

def get_letters_from_mask(rep, mask):
  mask_arr = np.ones(14)
  for i in mask:
      mask_arr[i-1] = 0
  l = []
  for letter in letters:
      if np.array_equal(get_letter_rep(letter) * mask_arr, rep):
          l.append(letter)
  return l

print(get_letters_from_mask(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1]), {1,2,3,4,5,6,7,8,9,10,11,12,13}))

def get_rep_from_word(word):
  return [get_letter_rep(l) for l in word.upper()]

def get_rep_from_mask(word, mask):
  return [mask_char(get_letter_rep(l),mask) for l in word.upper()]

print(get_rep_from_mask("tree", {1,2,3,4,5,6,7}))

def get_words_from_mask(rep, mask):
  mask_arr = np.ones(14)
  for i in mask:
      mask_arr[i-1] = 0
  words = []

  for word in freqs:
      #if(len(word)==1):
      #  print(word)
      l = []
      if(len(word)!=len(rep)):
        continue
      if all(np.array_equal(get_letter_rep((word.upper())[i]) * mask_arr, rep[i]) for i  in range(len(word.upper()))):
          words.append(word)

  return words

def get_probabilities(rep, mask):
    mask_arr = np.ones(14)
    for i in mask:
        mask_arr[i-1] = 0
    words = {}
    prob_sum = 0
    for word in freqs:
        #if(len(word)==1):
        #  print(word)
        l = []
        if(len(word)!=len(rep)):
            continue
        if all(np.array_equal(get_letter_rep((word.upper())[i]) * mask_arr, rep[i]) for i  in range(len(word.upper()))):
            words[word] = freqs[word]
            prob_sum += freqs[word]
    for word in words:
        words[word] /= prob_sum
    return words

def convert_masks(masks):
    new_masks = []
    for mask in masks:
        mask_arr = np.ones(14)
        for i in mask:
            
            mask_arr[i-1] = 0
        new_masks.append(mask_arr)
    #print(new_masks)
    return new_masks

def lines_in_letter(letter):
    lines = 0
    if letter[0] == 1:
        lines+=1
    if letter[1] == 1 or letter[8] == 1:
        lines+=1
    if letter[2] == 1 or letter[11] == 1:
        lines+=1
    if letter[3] == 1 or letter[10] == 1:
        lines+=1
    if letter[4] == 1 or letter[9] == 1:
        lines+=1
    if letter[5] == 1 or letter[12] == 1:
        lines+=1
    if letter[6] == 1 or letter[7] == 1:
        lines+=1
    if letter[13] == 1:
        lines+=1
    return lines

def probs_from_masks_freq(w, masks):
    rep = [get_letter_rep(l)*mask for l,mask in zip(w.upper(), masks)]
    words = {}
    prob_sum = 0
    for word in freqs:
        #if(len(word)==1):
        #  print(word)
        l = []
        if(len(word)!=len(rep)):
            continue
        if all(np.array_equal(get_letter_rep((word.upper())[i]) * mask, rep[i]) for i, mask  in zip(range(len(word.upper())), masks)):
            numlines = sum(lines_in_letter(get_letter_rep((word.upper())[i])) for i in range(len(word.upper())) if not np.array_equal(np.zeros(14), rep[i]))
            adjusted = 1 / (5 ** numlines)
            words[word] = freqs[word]
            prob_sum += freqs[word]
    for word in words:
        words[word] /= prob_sum
    return words

def probs_from_masks_lines(w, masks):
    rep = [get_letter_rep(l)*mask for l,mask in zip(w.upper(), masks)]
    words = {}
    prob_sum = 0
    for word in freqs:
        #if(len(word)==1):
        #  print(word)
        l = []
        if(len(word)!=len(rep)):
            continue
        if all(np.array_equal(get_letter_rep((word.upper())[i]) * mask, rep[i]) for i, mask  in zip(range(len(word.upper())), masks)):
            numlines = sum(lines_in_letter(get_letter_rep((word.upper())[i])) for i in range(len(word.upper())) if not np.array_equal(np.zeros(14), rep[i]))
            #print(f"word: {word}, numlines: {numlines}")
            adjusted = 1 / (5 ** numlines)
            words[word] = adjusted
            prob_sum += adjusted
    for word in words:
        words[word] /= prob_sum
    return words

def probs_from_masks_segs(w, masks):
    rep = [get_letter_rep(l)*mask for l,mask in zip(w.upper(), masks)]
    words = {}
    prob_sum = 0
    for word in freqs:
        #if(len(word)==1):
        #  print(word)
        l = []
        if(len(word)!=len(rep)):
            continue
        if all(np.array_equal(get_letter_rep((word.upper())[i]) * mask, rep[i]) for i, mask  in zip(range(len(word.upper())), masks)):
            numlines = sum(np.sum(get_letter_rep((word.upper())[i])) for i in range(len(word.upper())) if not np.array_equal(np.zeros(14), rep[i]))
            #print(f"word: {word}, numlines: {numlines}")
            adjusted = 1 / (5 ** numlines)
            words[word] = adjusted
            prob_sum += adjusted
    for word in words:
        words[word] /= prob_sum
    return words


all_mask = {1,2,3,4,5,6,7,8,9,10,11,12,13,14}

masks = convert_masks([{},{},{}, all_mask])
#print(probs_from_masks("pour", masks))
#print(get_words_from_mask([np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,1])], {1,2,3,4,5,7,8,9,10,11,12,13}))

#print(get_probabilities(get_rep_from_mask("lice", mask_top), mask_top))



#for letter in letters:
#    print(get_words_from_mask(get_rep_from_mask(letter, mask_bottom), mask_bottom))

# function to plot a given word
def plot_word(word):
    fig, ax = plt.subplots(1, len(word), figsize=(10, 5))
    
    imageObject = Image.open("digital_chars.gif")
    print(imageObject)
    for i, char in enumerate(word.upper()):
        imageObject.seek(letters.index(char))
        ax[i].imshow(imageObject)
        ax[i].axis('off')

    plt.show()

    
#plot_word('pace')


data = [{
    'right': 10,
    #'miami': 1
},
{
    'born': 9,
    'dokn': 1,
    #'dorm': 1
},
{
    'born': 8,
    'burn': 2,
    #'durn': 1
},
{
    'best': 2,
    'fest': 1,
    'nest': 5,
    'rest': 2,
},
{
    'bear': 6,
    'derp': 2,
    'dcoo': 1,
    #'pedo': 1
},
{
    'look': 8,
    'loop': 3
},
{
    'pour': 5,
    'pout': 5,
    'pouf': 1
},
{
    'harp': 9,
    'warp': 2
},
{
    'fight': 7,
    #'sight': 3,
    'eight': 1
},
{
    'east': 10,
    'easy': 1
},
{
    'wash': 3,
    'wasp': 8
}
]

q_masks = [
    [mask_top] * 5,
    [mask_top]*4,
    [{1,2,3,4,5,6}]*4,
    [{1,3,4,5,6,7,8,10,11,12,13,14}, {}, {}, {}],
    [mask_bottom]*5,
    [{}, {9,10,11,12,13,14},{}, {1,3,4,5,6,8,12,13}],
    [{},{},{}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14}],
    [mask_bottom, {}, mask_top, {}],
    [mask_bottom, mask_bottom, {}, {}, {1,2,3,4,5,6}],
    [{1,2,3,4,5,6}]*4,
    [mask_top, {}, mask_bottom, {1,2,3,4,5,6,7,8,9,10,11,12,13,14}]
]

q_words = [
    'right',
    'burn',
    'burn',
    'nest',
    'bear',
    'look',
    'pour',
    'harp',
    'fight',
    'easy',
    'wasp'
]

q_common = [
    {
        'right'
    },
    {
        'burn', 'born'
    },
    {
        'burn', 'born'
    },
    {
        'nest', 'pest', 'rest', 'lest', 'vest', 'west'
    },
    {
        'bear', 'dear'
    },
    {
        'look', 'loop'
    },
    {
        'pour', 'pout'
    },
    {
        'harp', 'warp'
    },
    {
        'fight', 'eight'
    },
    {
        'easy', 'east'
    },
    {
        'wash', 'wasp'
    }
]
freq_err = 0
rand_err = 0
line_err = 0
seg_err = 0
for i in range(11):
    probs = probs_from_masks_freq(q_words[i], convert_masks(q_masks[i]))
    print(probs)
    err = 0
    for w in q_common[i]:
        data_prob = 0
        if w in data[i]:
            data_prob = data[i][w] / sum(data[i].values())
        err += abs(data_prob - probs[w]/sum(probs[wo] for wo in q_common[i]))
    err /= len(q_common[i])
    print(err)
    freq_err+=err
    err = 0
    for w in q_common[i]:
        data_prob = 0
        if w in data[i]:
            data_prob = data[i][w] / sum(data[i].values())
        err += abs(data_prob - 1/len(q_common[i]))
    err /= len(q_common[i])
    print(err)
    rand_err += err
    err = 0
    probs = probs_from_masks_lines(q_words[i], convert_masks(q_masks[i]))
    for w in q_common[i]:
        data_prob = 0
        if w in data[i]:
            data_prob = data[i][w] / sum(data[i].values())
        err += abs(data_prob - probs[w]/sum(probs[wo] for wo in q_common[i]))
    err /= len(q_common[i])
    print(err)
    line_err += err
    err = 0
    probs = probs_from_masks_segs(q_words[i], convert_masks(q_masks[i]))
    for w in q_common[i]:
        data_prob = 0
        if w in data[i]:
            data_prob = data[i][w] / sum(data[i].values())
        err += abs(data_prob - probs[w]/sum(probs[wo] for wo in q_common[i]))
    err /= len(q_common[i])
    print(err)
    seg_err += err

print(f"freq: {freq_err / 11}")
print(f"rand: {rand_err / 11}")
print(f"lines: {line_err / 11}")
print(f"segs: {seg_err / 11}")


        