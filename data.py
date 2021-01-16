from os import walk, path
from collections import Counter


class LoadData:

    def __init__(self, mode=''):
        self.mode = mode
        self.url = self.link()
        self.mostCommonWords = 15000 #input('Enter m (Most common words to keep): ')
        self.discardFirstWords = 50 #input('Enter n (Most common words to discard [n < m]): ')

    def link(self):
        url = 'C:\\Users\\Apostolis\\Desktop\\aiHW\\aclImdb' #input("Enter 'aclImdb' folder path: ")
        return url

    def read_train(self):
        return self.url + '\\train'

    def read_test(self):
        return self.url + '\\test'

    def getVector(self, data, type):   #Returns vector of vectors
        if data == 'test':
            url = self.read_test()+'\\'
        elif data == 'train':
            url = self.read_train()+'\\'
        else:
            return
        if type != 'neg' and type != 'pos':
            return
        ls = []
        vectors = []
        vector = []
        f = []
        for (dirpath, folders, files) in walk(url+type):
            f.extend(files)
        for j in f:
            ls.extend(self.read_file(url + type + '\\' + j))
            dc = open(self.url+'\\'+'dictionary.txt', encoding="utf-8")
            line = dc.readline().strip()
            while line:
                if line in ls:
                    vector.append(1)
                else:
                    vector.append(0)
                line = dc.readline().strip()
            vectors.append(vector)
        return vectors

    def createDictionary(self):
        if not path.exists(self.url+'\\'+'dictionary.txt'):
            w = ['neg', 'pos']
            words = []
            for i in w:
                for(dirpath, folders, files) in walk(self.read_train()+'\\'+i):
                    files.extend(files)
                for j in files:
                    words.extend(self.read_file(self.read_train()+'\\'+i+'\\'+j))
            wordsCount = (w for w in words)
            cw = Counter(wordsCount)
            cw = cw.most_common(self.mostCommonWords)[self.discardFirstWords:]
            ww = [w[0] for w in cw]
            with open(self.url+'\\'+'dictionary.txt', "w", encoding="utf-8") as file:
                for i in ww:
                    file.write(i+"\n")

    def read_file(self, url):
        ls = []
        file = open(url, "r", encoding="utf-8")
        line = file.readline()
        while(line):
            ls.extend(line.split())
            line = file.readline()
        return ls
