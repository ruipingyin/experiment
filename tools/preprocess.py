import gzip

def readData():
    # Delte the duplicated data

    with gzip.open(reviewPath, 'r') as reviewFile:
        for review in reviewFile:
            reviews[(eval(review)['reviewerID'], eval(review)['asin'])] = (eval(review)['overall'], eval(review)['unixReviewTime'])
    print('Review Count =', len(reviews))

def generateMap():
    # Create user map and item map
    global users
    global items
    
    users, items = {}, {}

    for review, _ in reviews.items():
        users.setdefault(review[0], set())
        users[review[0]].add(review[1])
        items.setdefault(review[1], set())
        items[review[1]].add(review[0])
    
    print('nUsers =', len(users), ', nItems =', len(items))
    
def cleanData(userSparse, itemSparse):
    # Clean data with seldom infomations with:
    # the users who bought more than 10 items & the items which was bought more than 5 users
    global cleanIndex
    
    sparsePairs = []
    for review, _ in reviews.items():
        if len(users[review[0]]) < userSparse or len(items[review[1]]) < itemSparse:
            sparsePairs.append(review)
            cleanIndex = True
    
    for review in sparsePairs:
        del reviews[review]

    print('Review Count =', len(reviews))
    

reviewPath = '../dataset/reviews_Clothing_Shoes_and_Jewelry.json.gz'
metaPath = '../dataset/meta_Clothing_Shoes_and_Jewelry.json.gz'
reviews = {}
users = {}
items = {}
itemPerUser = 10
userPerItem = 5

cleanIndex = True

readData()


generateMap()
while(len(users) > 80000 or len(items) > 100000):
    cleanData(6, 1)
    generateMap()
    cleanData(1, 3)
    generateMap()


print('Clean succ!')

# Output Reviews
userIndex = {}
itemIndex = {}

for user, _ in users.items():
    userIndex[user] = len(userIndex)

for item, _ in items.items():
    itemIndex[item] = len(itemIndex)
    
with open('../dataset/reviews.txt', 'w') as reviewFile:
    for review, info in reviews.items():
        reviewFile.write(str(userIndex[review[0]]))
        reviewFile.write('::')
        reviewFile.write(str(itemIndex[review[1]]))
        reviewFile.write('::')
        reviewFile.write(str(int(info[0])))
        reviewFile.write('::')
        reviewFile.write(str(info[1]))
        reviewFile.write('\n')

with open('../dataset/userMap.txt', 'w') as userMap:
    for user, index in userIndex.items():
        userMap.write(user)
        userMap.write('\t')
        userMap.write(str(index))
        userMap.write('\n')

with open('../dataset/itemMap.txt', 'w') as itemMap:
    for item, index in itemIndex.items():
        itemMap.write(item)
        itemMap.write('\t')
        itemMap.write(str(index))
        itemMap.write('\n')
        
print('Finish ... ')