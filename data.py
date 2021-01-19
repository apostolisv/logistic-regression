from os import walk, path
from collections import Counter


class LoadData:

    def __init__(self):
        self.path = self.link()
        self.mostCommonWords = 175  # input('Enter m (Most common words to keep): ')
        self.discardFirstWords = 15  # input('Enter n (Most common words to discard [n < m]): ')

    def link(self):
        path = 'C:\\Users\\Apostolis\\Desktop\\aiHW\\aclImdb'  # input("Enter 'aclImdb' folder path: ")
        return path

    def read_train(self):
        return self.path + '\\train'

    def read_test(self):
        return self.path + '\\test'

    def getVector(self, data, ctype):   # Returns vector of vectors
        if data == 'test':
            path = self.read_test()+'\\'
        elif data == 'train':
            path = self.read_train()+'\\'
        else:
            return
        vectors = []
        f = []
        for (dirpath, folders, files) in walk(path+ctype):
            f.extend(files)
        for j in f:
            ls = []
            vector = []
            dc = open(self.path + '\\' + 'dictionary.txt', encoding="utf-8")
            ls.extend(self.read_file(path + ctype + '\\' + j))
            line = dc.readline().strip()
            while line:
                if line in ls:
                    vector.append(1)
                else:
                    vector.append(0)
                line = dc.readline().strip()
            if ctype == 'neg':
                vector.append(0)
            else:
                vector.append(1)
            vectors.append(vector)
        return vectors

    def createDictionary(self):
        if not path.exists(self.path + '\\' + 'dictionary.txt'):
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
            with open(self.path + '\\' + 'dictionary.txt', "w", encoding="utf-8") as file:
                for i in ww:
                    file.write(i+"\n")

    def read_file(self, path):
        ls = []
        file = open(path, "r", encoding="utf-8")
        line = file.readline().strip()
        while line:
            ls.extend(line.split())
            line = file.readline().strip()
        file.close()
        return ls

    def get_external(self, path):
        vector = []
        dc = open(self.path + '\\' + 'dictionary.txt', encoding="utf-8")
        ls = self.read_file(path)
        line = dc.readline().strip()
        while line:
            if line in ls:
                vector.append(1)
            else:
                vector.append(0)
            line = dc.readline().strip()
        return vector
