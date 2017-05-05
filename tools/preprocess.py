import gzip
import sys, operator
import multiprocessing

# Path of resources
reviewPath = '../dataset/Ama/reviews_Clothing_Shoes_and_Jewelry.json.gz'
reviews = []

# Read data from reviews
cnt = 0
with gzip.open(reviewPath, 'r') as reviewFile:
  for review in reviewFile:
    reviews.append([eval(review)['reviewerID'], eval(review)['asin'], eval(review)['unixReviewTime']])
    if cnt % 10000 == 0: sys.stdout.write('Done %d.\r' % cnt)
    cnt += 1

# Filter spase user
reviews = sorted(reviews, key = lambda x: x[0], reverse=True)

start, cleanReviews = 0, []

for i, review in enumerate(reviews):
  if review[0] == reviews[start][0]: continue
  if i - start > 5: cleanReviews.extend(reviews[start:i])
  start = i

# Output Reviews
userIndex, itemIndex = {}, {}

for user in set([review[0] for review in cleanReviews]):
  userIndex[user] = len(userIndex)

for item in set([review[1] for review in cleanReviews]):
  itemIndex[item] = len(itemIndex)
    
with open('../dataset/Ama/reviews.txt', 'w') as reviewFile:
  for review in cleanReviews:
    reviewFile.write(str(userIndex[review[0]]))
    reviewFile.write(' ')
    reviewFile.write(str(itemIndex[review[1]]))
    reviewFile.write(' ')
    reviewFile.write(str(review[2]))
    reviewFile.write('\n')

with open('../dataset/Ama/userMap.txt', 'w') as userMap:
  for user, index in userIndex.items():
    userMap.write(user)
    userMap.write('\t')
    userMap.write(str(index))
    userMap.write('\n')

with open('../dataset/Ama/itemMap.txt', 'w') as itemMap:
  for item, index in itemIndex.items():
    itemMap.write(item)
    itemMap.write('\t')
    itemMap.write(str(index))
    itemMap.write('\n')

print('Finish ... ')
