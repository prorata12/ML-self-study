from ch8_1 import *

"""
8.2. BoW 모델 소개
text --> feature vector
"""

"""
8.2.1 Text -> feature vector + n-gram explanation
"""
# Bag of Word model
# 각 단어가 몇 번씩 등장했는지 확인
print('\n 8.2.1 \n ')
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array(['The sun is shining', 'The weather is sweet', 'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)
print(bag.toarray()) # 각 열이 각 단어의 등장 횟수, 각 행이 한 문장 : term frequency (문서 d에서 등장한 단어 t의 횟수 : tf(t,d)) #text(word), document
print(count.vocabulary_) #각 단어가 mapping됨

# 1-gram/unigram model : item sequenece {The,sun,is}의 각 단어단어가 하나의 feature인 경우(즉, 각 token/item이 하나의 단어)
# n-gram : NLP에서 연속된 아이템의 sequence : Kanaris - 3~4-gram in spam email

# 1 gram : The, sun, is, shining
# 2 gram : The sun, sun is, is shining

count2 = CountVectorizer(ngram_range=(2,2)) # 2-gram expression
bag2 =count2.fit_transform(docs)
print(bag2.toarray())
#(document, word index) wordfrequency in doucment
print(count2.vocabulary_)



"""
8.2.2 tf-idf를 사용한 단어 적합성 평가
tf-idf : term frequency - inverse document frequency (단어 빈도 * 역문서 빈도)
: defined as (term frequency) * (inverse document frequency)
"""
print('\n 8.2.2 \n')
# tf-idf(t,d) = tf(t,d) * idf(t,d)

np.set_printoptions(precision=2) # 소수자리 표현


#########################################################################################################

from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True, 
                         norm='l2', 
                         smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

tf_is = 3
n_docs = 3
idf_is = np.log((n_docs+1) / (3+1))
tfidf_is = tf_is * (idf_is + 1)
print('tf-idf of term "is" = %.2f' % tfidf_is)

tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
print(raw_tfidf)

l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
print(l2_tfidf)

"""
8.2.3
"""
print('\n 8.2.3 \n')
df.loc[0, 'review'][-50:]

import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

print(preprocessor(df.loc[0, 'review'][-50:]))
print(preprocessor("</a>This :) is :( a test :-)!"))

df['review'] = df['review'].apply(preprocessor)
print(df['review'].map(preprocessor))

"""
8.2.4
"""

def tokenizer(text):
    return text.split()

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

tokenizer('runners like running and thus they run')
tokenizer_porter('runners like running and thus they run') # 어간 추출기, 제공 알고리즘

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

stop = stopwords.words('english')

print([w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop])
print(stop)