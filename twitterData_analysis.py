import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

twitter_data = pd.read_csv('result.csv')

plt.figure()
hist1,edges1 = np.histogram(twitter_data.followers)
plt.bar(edges1[:-1],hist1,width=edges1[1:]-edges1[:-1])

print(twitter_data.corr())

plt.scatter(twitter_data.friends,twitter_data.followers)
#plt.scatter(youtube_data.viewCount,youtube_data.likeCount)

y = twitter_data.followers
X = twitter_data.friends
X = sm.add_constant(X)

lr_model = sm.OLS(y,X).fit()

print(lr_model.summary())

X_prime = np.linspace(X.friends.min(),X.friends.max(),100)
X_prime = sm.add_constant(X_prime)

y_hat = lr_model.predict(X_prime)
plt.scatter(X.friends,y)
plt.xlabel("friends Count")
plt.ylabel("followers Count")
plt.plot(X_prime[:,1],y_hat) 