import gzip
import sys, operator
import multiprocessing

def filterSpase(reviews, index, min):
  # Filter spase user
  reviews = sorted(reviews, key = lambda x: x[index], reverse=True)

  start, cleanReviews = 0, []

  for i, review in enumerate(reviews):
    if review[index] == reviews[start][index]: continue
    if i - start > min: cleanReviews.extend(reviews[start:i])
    start = i
    
  return cleanReviews

# Path of resources
reviewPath = '../dataset/Ama/reviews_Clothing_Shoes_and_Jewelry.json.gz'
reviews = []

# Read data from reviews
cnt = 0
with gzip.open(reviewPath, 'r') as reviewFile:
  for review in reviewFile:
    reviews.append([eval(review)['reviewerID'], eval(review)['asin'], eval(review)['unixReviewTime'], eval(review)['overall']])
    if cnt % 10000 == 0: sys.stderr.write('Done %d.\r' % cnt)
    cnt += 1
sys.stderr.write('\n')

# Filter spase user
histCount = 0
while(histCount != len(reviews)):
  histCount = len(reviews)
  print histCount
  reviews = filterSpase(reviews, 0, 1)
  reviews = filterSpase(reviews, 1, 1)

cleanReviews = filterSpase(reviews, 0, 10)

# Output Reviews
userIndex, itemIndex = {}, {}

for user in set([review[0] for review in cleanReviews]):
  userIndex[user] = len(userIndex)

for item in set([review[1] for review in cleanReviews]):
  itemIndex[item] = len(itemIndex)
    
with open('../dataset/Ama/reviews_1.txt', 'w') as reviewFile:
  for review in cleanReviews:
    reviewFile.write(str(userIndex[review[0]]))
    reviewFile.write(' ')
    reviewFile.write(str(itemIndex[review[1]]))
    reviewFile.write(' ')
    reviewFile.write(str(review[3]))
    reviewFile.write(' ')
    reviewFile.write(str(review[2]))
    reviewFile.write('\n')

with open('../dataset/Ama/userMap_1.txt', 'w') as userMap:
  for user, index in userIndex.items():
    userMap.write(user)
    userMap.write('\t')
    userMap.write(str(index))
    userMap.write('\n')

with open('../dataset/Ama/itemMap_1.txt', 'w') as itemMap:
  for item, index in itemIndex.items():
    itemMap.write(item)
    itemMap.write('\t')
    itemMap.write(str(index))
    itemMap.write('\n')

print('Finish ... ')