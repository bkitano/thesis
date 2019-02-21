import codecs
from os import listdir
from os.path import isfile, join

folder = '/Users/bkitano/Desktop/Classes/Spring_2019/thesis/corpus/cleaned_txt/'
onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

# for filename in onlyfiles:
#     with codecs.open(folder + filename) as f:
#         text = f.read()
#         p = text.split(' ')
#         print(p)

weird = [
    u'\xc5', # capital A with hat
    u'\xbf' # questionmark
]

for w in weird:
    print(w)