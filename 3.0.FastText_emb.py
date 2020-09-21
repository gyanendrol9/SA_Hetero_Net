from gensim.models.fasttext import FastText

stopwords = ['rt','amp','url','sir','day','title','shri','crore','time',"a", "about","above", "across", "after", "afterwards", "again", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
tstop = ['rt','amp','url']
import itertools   

import re

def text_cleaner(text):
    newString = text.lower() 
    newString = re.sub(r"'s\b"," ",newString)
    newString = re.sub("[^a-zA-Z_@#]", " ", newString) 
    long_words=[]
    for i in newString.split():
        if i not in tstop:                  
            long_words.append(i)
    return long_words

import emoji

def remove_link(text):
    regex = r'https?://[^\s<>)"‘’]+'
    match = re.sub(regex,' ', text)
    regex = r'urls?'
    match = re.sub(regex,' ', match)
    tweet=emoji.demojize(match) 
    tweet = re.sub("[0-9:;,.()!?…]", " ", tweet)
    tweet = re.sub("[ ]+", " ", tweet) 
    return tweet.strip()


import pickle

f_tweet = open('data/semeval13_multiplex_network.pkl', 'rb')
tweet_net=pickle.load(f_tweet)       #((hashtags,mentions,edges,h_k,m_k),res,label)
f_tweet.close()

sentences=[]
for tid in tweet_net:
    line=' '.join(tweet_net[tid][1]).lower().replace('@',' M_').replace('#',' H_')
    line=remove_link(line.replace(' M_ ',' ').replace(' H_ ',' '))
    sentences.append(line)
    print(line)


f_tweet = open('Tweets_biased_multiplex_walk', 'r')
tweet_net=f_tweet.readlines()       #((hashtags,mentions,edges,h_k,m_k),res,label)
f_tweet.close()

for tid in tweet_net:
    line=tid.strip().replace('\t',' ')
    sentences.append(line)
    print(line)



from nltk.stem import WordNetLemmatizer
import nltk
stemmer = WordNetLemmatizer()

import random
random.shuffle(sentences)

def preprocess_text(document):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if len(word) > 3]
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text

en_stop = set(nltk.corpus.stopwords.words('english'))
#final_corpus = [preprocess_text(sentence) for sentence in sentences if sentence.strip() !='']

word_punctuation_tokenizer = nltk.WordPunctTokenizer()
word_tokenized_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in sentences]

embedding_size = 60
window_size = 10
min_word = 5
down_sampling = 1e-2

ft_model = FastText(word_tokenized_corpus,
                      size=embedding_size,
                      window=window_size,
                      min_count=min_word,
                      sample=down_sampling,
                      sg=1,
                      iter=100)

ft_model.save('model/semeval13_multiplex_walk_fastext.model')    

ft_model.wv.most_similar(['pmoindia'], topn=5)
print(ft_model.wv.similarity(w1='pmoindia', w2='pmmodi'))

#model = FastText.load('gensim_out/42K_combo_text_fasttext_sg.model')

