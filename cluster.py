from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import matplotlib.pyplot as plt
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re, math
import pickle

dataset = pd.read_csv('Dataset.csv', nrows=300)

dataset.pop("title")
dataset.pop("label")
dataset.pop("author")

#documents= dataset["text"]

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
def CleanData(documents):
    
    Totalrows =  dataset.shape[0]

    for i in range(0, Totalrows):
            text = dataset["text"][i]
            if not text:
                    continue;
            text= str(text)
            review = re.sub('[^a-zA-Z]', ' ',text)
            review = review.lower()
            review = review.split()
            review = [ps.stem(word) for word in review if not word in stop_words]
            dataset["text"][i] = ' '.join(review)
            

CleanData(dataset)


#print(documents.shape)
from sklearn.model_selection import train_test_split
X_train, X_test= train_test_split(dataset,test_size=0.33, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X_train["text"])

'''
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',max_iter=300,n_init=10, random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    print(i)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()'''



true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,random_state = 42)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

ClusterMapping={}
counterLength = X_test.shape[0]
counter=0
while(counter<counterLength):
    _news =dataset.iloc[counter]
    Y = vectorizer.transform([_news.text])
    prediction = model.predict(Y)
    ClusterID = prediction[0]
    if(ClusterID in ClusterMapping):
            KeyValue =  ClusterMapping[ClusterID]
            ClusterMapping[ClusterID] = str(KeyValue) + ',' +str(_news.id)
                                        
    else:
        ClusterMapping[ClusterID] = str(_news.id)
    counter=counter+1

for c, n in ClusterMapping.items():
    print("\n Cluster ID ==> {0} News IDs ==> {1}".format(c, n))

Y = vectorizer.transform(["Clinton Campaign Demands FBI Affirm Trump's Russia Ties  With the 2016 election campaign winding down, the Clinton campaign is ratcheting up demands for the FBI to publicly confirm the campaignâ€™s allegations that Republican nominee Donald Trump is secretly in league with Russia. Sen. Harry Reid (D â€“ NV) went so far as to claim the FBI has secret â€œexplosiveâ€ evidence of coordination between the Trump campaign and the Russian government that it is withholding."]) 
prediction = model.predict(Y)
print(prediction)

