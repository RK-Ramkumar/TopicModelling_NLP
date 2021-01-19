#!/usr/bin/env python
# coding: utf-8

# # NLP - Topic Analysis 

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import re
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models import ldamodel


# A popular mobile phone brand has launched their smartphone in market. 
# The client wants to understand the VOC (voice of the customer) on the product. 
# This will be useful to not just evaluate the current product, but to also get some direction for developing the product pipeline. 
# The client is particularly interested in the different aspects that customers care about. 
# Product reviews by customers on a leading e-commerce site should provide a good view.
# 
#     Sentiment: The sentiment against the review (4,5 star reviews are positive, 1,2 are negative)
#     Reviews: The main text of the review
# 
# 1.Read the review data.

# In[3]:


topicData = pd.read_csv("reviews.csv")
topicData.head()


# 2.Normalize casings for the review text and extract the text into a list for easier manipulation.

# In[4]:


#Casing Normalize 
reviewsExtract =  [review.lower() for review in topicData.review.values]
print(reviewsExtract[0:5])


# 3.Tokenize the reviews using NLTKs word_tokenize function.
#     
#     Tokenization is essentially splitting a phrase, sentence, paragraph, or an entire text document into smaller units, such as individual words or terms

# In[5]:


#Tokenize using NLTK's
reviewTokens =  [word_tokenize( review ) for review in reviewsExtract]
print(reviewTokens[0])


# 4.Perform parts-of-speech tagging on each sentence using the NLTK POS tagger.
# 
#     POS-tagging simply implies labelling words with their appropriate Part-Of-Speech (Noun, Verb, Adjective, Adverb, Pronoun, …). POS tagging can be really useful, particularly if you have words or tokens that can have multiple POS tags

# In[6]:


#Parts-Of-Speech Tagging
reviewTags =  [nltk.pos_tag(review) for review in reviewTokens]
print(reviewTags[0])


# 5.For the topic model, include all the POS tags that correspond to nouns and Limit the data to only terms with these tags.
# 
#     As corpus tends have a broad and varied vocab-ulary, that can be time consuming to topic model, limiting articles to only the nouns also offers the advantage of reducing the size of the vocabulary to be modelled.Topic Modelling is more efficient in noun only approach 

# In[54]:


#extraction of tags starts with NN
reviewsNoun = []
for nountag in reviewTags :  reviewsNoun.append([token for token in nountag if re.search("NN.*",token[1])])
print(reviewsNoun[0])


# 6.Lemmatize : 
#    
#        lemmatizing means to extract the ‘lemma’ from a given word after its morphological analysis. For example: If we lemmatize ‘studies’ and ‘studying’, we will end up with ‘study’ as its lemma.

# In[13]:


#Lemmatize
reviewsLemm = []
lemm = WordNetLemmatizer()
for data in reviewsNoun :  reviewsLemm.append([lemm.lemmatize(word[0]) for word in data]) 
print(reviewsLemm[0])


# 7.Remove stopwords and punctuation (if there are any). 

# In[19]:


#Removing Stop Words and Punctuations
stoplist =stopwords.words('english')
stopupdated = stoplist + list(punctuation) + ["..."] + [".."]
reviewsStopRemoved = []
for data in reviewsLemm : reviewsStopRemoved.append([word for word in data if word not in stopupdated])
print(reviewsStopRemoved[0:3]) #displaying the results


# 8.Creating a topic model using LDA on the cleaned-up data - with 12 topics.
# 
#     Topic Modeling is a technique to extract the hidden topics from large volumes of text.
#     Latent Dirichlet Allocation(LDA) is a popular algorithm for topic modeling. 
# 
#     LDA’s considers each document as a collection of topics in a certain proportion. And each topic as a collection of keywords, again, in a certain proportion.
# 
#     Once you provide the algorithm with the number of topics, all it does it to rearrange the topics distribution within the documents and keywords distribution within the topics to obtain a good composition of topic-keywords distribution.
# 

# In[50]:


#index Mapping
wordId = corpora.Dictionary(reviewsStopRemoved)
corpus = [wordId.doc2bow(review) for review in reviewsStopRemoved]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=wordId,num_topics=12,random_state=42,passes=10,per_word_topics=True)
list = lda_model.print_topics()
for topic in list: 
    print(topic)

#coherence calculations
co_lda_model = CoherenceModel(model=lda_model, texts=reviewsStopRemoved,dictionary=wordId,coherence='c_v')
co_lda = co_lda_model.get_coherence()
print('\nResult : ',co_lda)


# 9.From the business lens, the topics can combine in below ways,
1. Topic 2,5,7 possibly talks about - Pricing
2. Topic 4, 6 and 10 talks about - battery quality Issues
3. Topic 3 and 11 are talks about - Performances
# 10.Creating the topic model using LDA with the optimal number of topics (Here, I choose 8)
# 
#     Coherence provides a convenient measure to judge how good a given topic model is. 
#     
#     For finding the optimal number of topics, build many LDA models with different values of number of topics(k) & pick the one  that gives the highest coherence value.
#     Choosing a ‘k’ that marks the end of a rapid growth of topic coherence usually offers meaningful & interpretable topics. Picking an even higher value can sometimes provide more granular sub-topics.
# 

# In[34]:


#Creating model with 8 topics
lda8model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=wordId,num_topics=8,random_state=42,passes=10,per_word_topics=True)
co_lda8model = CoherenceModel(model=lda8model, texts=reviewsStopRemoved,dictionary=wordId,coherence='c_v')
co8lda = co_lda8model.get_coherence()
print('\nScore Result : ',co8lda)


# The coherence is now 0.53 which is a significant increase from previous value 0.47 

# 11. Creating a table with the topic name and the top 10 terms in each to present to the business.

# In[46]:


results = lda8model.show_topics(formatted=False)
t_words = [(topics[0],[name[0] for name in topics[1]]) for topics in results]
for t,w in t_words:
    print(str(t) + " : "+str(w))




