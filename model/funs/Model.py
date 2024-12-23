from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self, X_train):
        self.X_train = X_train
        self.model = None

    def ANN(self):
        self.model = Sequential([
            Dense(256, activation='relu', input_shape=(self.X_train.shape[1],)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(4, activation='softmax')
        ])

        return self.model, 'ANN'
    
    def DecisionTree(self):
        self.model = DecisionTreeClassifier(max_depth=5)
        return self.model, 'DecisionTree'
    
    def RandomForest(self):
        self.model = RandomForestClassifier()
        return self.model, 'RandomForest'
    
    
