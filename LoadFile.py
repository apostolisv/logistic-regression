from os import walk, path
from collections import Counter
import numpy as np


class LoadFile:
    def __init__(self, mode=''):
        self.mode = mode
        self.url = self.link()
        self.mostCommonWords = 15000 #input('Enter m (Most common words to keep): ')
        self.discardFirstWords = 50 #input('Enter n (Most common words to discard [n < m]): ')

    def link(self):
        url = 'C:\\Users\\Apostolis\\Desktop\\aiHW\\aclImdb' #input("Enter 'aclImdb' folder path: ")
        return url

    def read_train(self):
        train_url = self.url + '\\train'
        return train_url

    def read_test(self):
        test_url = self.url + '\\test'
        return test_url

    def getVector(self, data):
        if data == 'test':
            url = self.read_test()+'\\'
            w = ['neg', 'pos']
        elif data == 'train':
            url = self.read_train()+'\\'
            w = ['neg', 'pos', 'unsup']
        else:
            return
        f = []
        vectors = np.array(np.zeros(self.mostCommonWords-self.discardFirstWords))
        for i in w:
            for (dirpath, folders, files) in walk(url+i):
                f.extend(files)

    def load_test(self):
        pass

    def load_train(self):
        pass

    def createDictionary(self):
        if not path.exists(self.url+'\\'+'dictionary.txt'):
            w = ['neg', 'pos', 'unsup']
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
        txtfile = open(url, "r", encoding="utf-8")
        line = txtfile.readline()
        ls.extend(line.split())
        while(line):
            line = txtfile.readline()
            ls.extend(line.split())
        return ls




    def exists(self, ls, word):
        if(word in ls):
            return True
        return False