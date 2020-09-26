# Sentiment Analysis of Tweets using Heterogeneous Multi-layer Network Representation and Embedding

<h4>Loitongbam Gyanendro Singh, Anasua Mitra, Sanasam Ranbir Singh</h4>

<p>
Sentiment classification on Twitter text often needs to deal with the problems of under-specificity, noise, and multilingual content.
In this study, we propose a heterogeneous multi-layer network-based representation of tweets to generate multiple representations of a tweet and address the above issues. The generated representations are further ensembled and classified using a neural-based early fusion approach. Further, we propose a centrality aware random-walk for node embedding and tweet representations suitable for the multi-layer network.
From various experimental analyses, it is evident that the proposed method can address the problem of under-specificity, noisy text, and multilingual content present in a tweet and provides better classification performance than the text-based counterpart.
Further, the proposed centrality aware based random walk provides better representations than unbiased and other biased counterparts. 
</p>
<h4>Proposed framework</h4>
<img src="https://github.com/gloitongbam/SA_Hetero_Net/blob/master/ensemble_new.png" alt="Framework">
<br>

## Requirement
The software can run on CPU or GPU, dependency requirements are following:

+ python
+ tensorflow

'''
pip install numpy
pip install keras
pip install pandas
pip install networkx
pip install gensim
pip install nltk
pip install editdistance
pip install emoji
'''

<h3>This work is accepted in EMNLP 2020</h3>
