import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
import random

import math

class Factory:
    pass

class Densefactory(Factory):
    def __init__(self, Dense_list, optimizer, loss,lr = 0.001, metrics = ['mae', 'mse']) -> None:
        self.__optimizer_dict = {
            'sgd': tf.keras.optimizers.SGD,
            'adam': tf.keras.optimizers.Adam
        } 
        self.__loss_dict = {
            'mae': tf.keras.losses.MAE,
            'mse': tf.keras.losses.MSE
        }  
        layers = [tf.keras.layers.Dense(i) for i in Dense_list]
        model = tf.keras.Sequential(layers)
        model.compile(optimizer = self.__optimizer_dict[optimizer.lower()](lr = lr), loss= self.__loss_dict[loss.lower()], metrics=metrics)
        self.model = model
    def __call__(self):
        return self.model
    def fit(self,features,labels, epochs):
        self.model.fit(features, labels, epochs=epochs)


class Utilities:
    def __init__(self) -> None:
        pass
    @staticmethod
    def shuffle_dataset(features, labels, global_seed, local_seed=None):
        assert features.shape == labels.shape
        shape = features.shape
        tf.random.set_seed(seed = global_seed)
        indicies = np.arange(len(features))
        if local_seed == None:
            local_seed = global_seed
        shuffled_indicies = tf.random.shuffle(indicies, seed = local_seed)
        return tf.gather(features, shuffled_indicies),tf.gather(labels, shuffled_indicies)
    
    @staticmethod
    def train_test(features,labels, train_percent = 0.2, validation_percent=None):
        assert len(features) == len(labels)
        data_len = len(labels)
        train_lim = math.floor(data_len*train_percent)
        if validation_percent == None:
            return [features[:train_lim], labels[:train_lim], features[train_lim:], labels[train_lim:]]
        else:
            validation_lim = math.floor(data_len*(validation_percent+train_percent))
            return [features[:train_lim], labels[:train_lim], features[train_lim:validation_lim], labels[train_lim:validation_lim], features[validation_lim:], labels[validation_lim:]]
    
    @staticmethod
    def plot_desicion_boundry(model,x,y,percision=0.05):
        import matplotlib.pyplot as plt
        min_xx,min_xy = np.amin(x[:,0])-0.1, np.amin(x[:,1])-0.1
        max_xx,max_xy = np.amax(x[:,0])+0.1, np.amax(x[:,1])+0.1
        percision = math.floor(1/percision)
        meshx = np.linspace(min_xx,max_xx, percision)
        meshy = np.linspace(min_xy,max_xy, percision)
        xs,ys = np.meshgrid(meshx, meshy)
        x_in = np.c_[xs.ravel(),ys.ravel()]
        z = model.predict(x_in)
        plt.contourf(xs,ys,np.reshape(z,newshape=xs.shape),cmap=plt.cm.YlGnBu, alpha=0.8)
        plt.scatter(x[:,0], x[:,1],  c=y, s=10, cmap = plt.cm.RdBu)

    @staticmethod
    def k_fold(x,y,k,number):
        assert len(x) == len(y)
        length = len(x)
        border_idx1 = int(math.floor((k-1)/number*length))
        border_idx2 = int(math.floor((k)/number*length))
        x_valid = x[border_idx1:border_idx2]
        y_valid = y[border_idx1:border_idx2]
        xt1 = x[0:border_idx1]
        xt2 = x[border_idx2:-1]
        yt1 = y[0:border_idx1]
        yt2 = y[border_idx2:-1]
        if (type(x) == pd.DataFrame): 
            return pd.concat([xt1, xt2]), pd.concat([yt1, yt2]), x_valid, y_valid
        else:
            return np.append(xt1, xt2), np.append(yt1,yt2), x_valid, y_valid
    
    @staticmethod
    def set_seed(seed: int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.experimental.numpy.random.seed(seed)
        tf.random.set_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")



class Preprocessing: 
    @staticmethod
    def normalize_dummify(table):
        numerical = table.select_dtypes(include = 'number')
        n_columns = numerical.columns
        catergorical = table.select_dtypes(exclude = 'number')
        dummy_catergorical = pd.get_dummies(catergorical)
        scalar = preprocessing.MinMaxScaler()
        transformed_numerical = pd.DataFrame(scalar.fit_transform(numerical), columns = n_columns)

        return pd.concat([transformed_numerical,dummy_catergorical], axis=1)
            
    def standardize_dummify():
        pass






def main():
    pass


if __name__ == '_-main__':
    main()