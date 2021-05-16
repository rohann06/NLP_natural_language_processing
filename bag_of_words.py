import nltk

para = """Ok, so he might show signs of nervousness at times. But I see it more as the result of maybe some shyness, and humility with a lot of self awareness, almost bashfulness. I bit like a real life Hugh Grant. But with double the IQ.
Intelligent people are also very sensitive to things around them which in Elon's case includes the energy of people he was interacting with in those interviews.
Furthermore he is probably forced to spend 99% of his time working on incredibly complex technical problems where human relation skills, let alone public speaking skills are irrelevant. So on the odd occasion when he does have to step out into the public eye he might feel the nerves, and his lack of exposure to talking to people might show.
I bet you if he spent half of the time in the public speaking realm as most other people he would be as good as the best at this sort of delivery.
What is most important here to bear in mind though is that this is a genuine guy, not trying to project any image or impress with artificial charm.
Just a real person showing a bit of the nerves. Just like you and I probably would. So give the guy a break."""


import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()

sentence = nltk.sent_tokenize(para)

corpus=[]

#CLEANING
for i in range(len(sentence)):
    review = re.sub('[^a-zA-Z]',' ',sentence[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#BAGOFWORDS
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()