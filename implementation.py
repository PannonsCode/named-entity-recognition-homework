import numpy as np
from typing import List, Tuple

from model import Model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torchcrf import CRF
from typing import *
import string
import json



def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    #return RandomBaseline()
    return StudentModel(device)


class RandomBaseline(Model):
    options = [
        (3111, "B-CORP"),
        (3752, "B-CW"),
        (3571, "B-GRP"),
        (4799, "B-LOC"),
        (5397, "B-PER"),
        (2923, "B-PROD"),
        (3111, "I-CORP"),
        (6030, "I-CW"),
        (6467, "I-GRP"),
        (2751, "I-LOC"),
        (6141, "I-PER"),
        (1800, "I-PROD"),
        (203394, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self, dev: str):
		
		#load pre-trained classifier model
        self.model = torch.load('model/model.pt', map_location=dev)
        self.model.eval()
        
		#glove ready for use
        self.voc = torch.load('model/vocabulary.txt')
        
		#variables used for elaboration
        self.PAD = 'pad'
        self.UNK = 'unk'
        self.max_len = 0
        
        #load the dictionary of classes
        my_dict = open('model/dict.json', 'r')
        self.classes_dict = json.loads(my_dict.read())

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
		
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        
        lenghts = []
        
        #Compute a list of lengths of each list and tha max length
        new_tokens = []
        lenghts, self.max_len = self.compute_max_len(tokens)
        
        #Add padding
        for i in range(len(tokens)):
            while len(tokens[i])<self.max_len:
                tokens[i].append(self.PAD)
				
		#Covert werds in the correpondent tensors
        new_tokens = self.VecFromWord(tokens)

		#Loop for predict tags
        final_labels=[]
        for token in new_tokens:
            labels = []
            token = token.view(1,token.shape[0], token.shape[1])
            pred = self.model(token.type(torch.float)) #prediction
            for pi in pred:
                for i in pi:
                    labels.append(self.classes_dict.get(str(i)))
                final_labels.append(labels)

        #Remove pad
        new_final = []   
        for (l,i) in zip(final_labels,lenghts):
            new_l = []
            new_l = l[:i]
            new_final.append(new_l)

        #Return List[List[str]]
        return new_final
        
    def compute_max_len(self, data: List[List[str]]):

        lenghts = []
        for x in data:
            lenghts.append(len(x))
        return lenghts, max(lenghts)
		
    def VecFromWord(self, text: List[List[str]]):

        new_vecs = []
        
        for t in text:
			#associate each word in every text to a tensor, using the vocabulary(pre-trained word-embedding)
            vecs = self.transform(t)
			# form: [[tensor1_1,....,tensor1_N],...,[tensorM_1,.....,tensorM_N]]
            new_vecs.append(vecs)

        return torch.stack(new_vecs)

	#function for effective transormation
    def transform(self, l: list):

        vec = []
        
        for word in l:
            word = word.lower()
            if word in self.voc.keys():
                vec.append(self.voc[word].squeeze(0))
            elif word == self.PAD:
                vec.append(self.voc[self.PAD].squeeze(0))
            else:
                vec.append(self.voc[self.UNK].squeeze(0)) 

        return torch.stack(vec)
    

class NERClassifierCRF(torch.nn.Module):

    def __init__(self, embedding_dim, n_hidden, tagset_size):
        super(NERClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional = True, num_layers = 2, droput=0.3,  batch_first=True)
        self.hidden1 = torch.nn.Linear(2*n_hidden, 100)
        self.hidden2 = torch.nn.Linear(100,80)
        self.hidden3 = torch.nn.Linear(80,50)
        self.hidden4 = torch.nn.Linear(50,14)
        self.crf = CRF(14, batch_first = True)

    def forward(self, sentence):
        out, _ = self.lstm(sentence)
        out = self.hidden1(out)
        out = torch.relu(out)
        out = nn.Dropout(0.2)(out)
        out = self.hidden2(out)
        out = torch.relu(out)
        out = nn.Dropout(0.3)(out)
        out = self.hidden3(out)
        out = torch.relu(out)
        out = nn.Dropout(0.3)(out)
        out = self.hidden4(out)
        out = F.log_softmax(out, dim=2)
        out = self.crf.decode(out)
        return out
