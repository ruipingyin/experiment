import gzip
import sys, operator
import threading
from utils import ColorPrint
import urllib2

itemPath = '../dataset/Ama/itemMap_1.txt'
metaPath = '../dataset/Ama/meta_Clothing_Shoes_and_Jewelry.json.gz'
imPath = '../dataset/Ama/imPath_1.txt'
imPath1 = '../dataset/Ama/imPathEmpty_1.txt'

''' Get Item List for Download Image 
# load item list
itemList = []
with open(itemPath, 'r') as itemListFile:
  for itemInfo in itemListFile:
    itemList.append(itemInfo.strip('\r\n').split())

# load image url from original dataset
itemMap = {}

cnt = 0
with gzip.open(metaPath, 'r') as metaFile:
  for meta in metaFile:
    if eval(meta).has_key('imUrl'):
      itemMap[eval(meta)['asin']] = eval(meta)['imUrl']
    else:
      itemMap[eval(meta)['asin']] = ''
    if cnt % 10000 == 0: sys.stderr.write('Done %d.\r' % cnt)
    cnt += 1

EmptyList = []
with open(imPath, 'w') as imPathFile:
  for imName in itemList:
    if itemMap[imName[0]] == '':
      EmptyList.append(imName)
    else:
      imPathFile.write(imName[1])
      imPathFile.write('\t')
      imPathFile.write(itemMap[imName])
      imPathFile.write('\n')
      
with open(imPath1, 'w') as imPathFile1:
  for imName in EmptyList:
    imPathFile1.write(imName[0])
    imPathFile1.write('\t')
    imPathFile1.write(imName[1])
    imPathFile1.write('\n')
    
print '\nFinish'
'''
rootDir = '/data/ruyin/Graphs/images/'

class ReadThread(threading.Thread):
  def __init__(self, threadID):
    threading.Thread.__init__(self)
    self.threadID = threadID
    
  def run(self):
    itemCount = len(itemList) / 20
    downloadCount = 0
    for item in itemList[itemCount * self.threadID : itemCount * self.threadID + itemCount]:
      self.saveIm(item[1], item[0])
      downloadCount += 1
      if downloadCount % 100 == 0: ColorPrint('%d has processed %d' % (self.threadID, downloadCount * 100 / itemCount), 1)
    
  def saveIm(self, url, name):
    try:
      data = urllib2.urlopen(url).read()
      with open(rootDir + str(name) + '.jpg', 'wb') as f:
        f.write(data)
    except:
      threadLock.acquire()
      with open('err.txt', 'a') as errf:
        errf.write(url)
        errf.write('\t')
        errf.write(str(name))
        errf.write('\n')
      threadLock.release()
        
itemList = []   # name \t url
with open(imPath, 'r') as imListFile:
  for line in imListFile:
    itemList.append(line.strip('\r\n').split())

ColorPrint('nItems = %d' % len(itemList), 1)

threadLock = threading.Lock()
threads = []

for i in range(0, 20):
  thread = ReadThread(i)
  thread.start()
  threads.append(thread)

for t in threads:
  t.join()

print("Exiting Main Thread")