import os 
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from utils.logger import check_and_mkdir_if_necessary, ROOT_DIR, today
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])



class AdultDataset(Dataset):
    def __init__(self):
        super(AdultDataset, self).__init__()

        self.name = 'adult'
        path = os.path.join(ROOT_DIR, 'data', 'input', self.name, 'adult.csv')

        
        # Load dataset
        self.data = pd.read_csv(path)
        
        self.__preprocess_data()        
        self.__normalize_data()
        self.__save_clean()

        self.num_features = self.data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        sample = torch.tensor(sample.values, dtype=torch.float32)
        return sample

    def __normalize_data(self):
        minmax_scaler = MinMaxScaler()
        numerical_fields = ['age', 'educational-num', 'capital-gain', 'hours-per-week']
        self.data[numerical_fields] = minmax_scaler.fit_transform(self.data[numerical_fields])

        
        label_encoder = LabelEncoder()
        categorical_fields = ['employment-type', 'race', 'relationship']
      
        for field in categorical_fields:
            self.data[field] = label_encoder.fit_transform(self.data[field])

    def __preprocess_data(self):
        self.data = pd.read_csv(os.path.join(config.get('datasets_dir'), 'adult', 'adult.csv'))
        
        #1 Mapping 'income' column values
        self.data['income'] = self.data['income'].map({'<=50K': 1, '>50K': 0}).astype(int)

        #2 Mapping 'gender' column values
        self.data['gender'] = self.data['gender'].map({'Male': 1, 'Female': 0}).astype(int)

        #3 Handling missing values in 'native_country', 'workclass', and 'occupation' columns
        self.data['native-country'] = self.data['native-country'].replace('?', np.nan)
        self.data['workclass'] = self.data['workclass'].replace('?', np.nan)
        self.data['occupation'] = self.data['occupation'].replace('?', np.nan)

        self.data.dropna(how='any', inplace=True) # Drop any rows that has a field with NaN value

        #4 Mapping 'native-country' values to binary labels
        self.data.loc[self.data['native-country'] != 'United-states', 'native-country'] = 'Non-US'
        self.data.loc[self.data['native-country'] == 'United-states', 'native-country'] = 'US'
        self.data['native-country'] = self.data['native-country'].map({'US': 1, 'Non-US': 0}).astype(int)

        #5 Mapping 'marital-status' values to to binary labels
        marital_status_labels = ['Divorced', 'Married-spouse-absent', 'Never-married', 'Separated','Widowed']
        self.data['marital-status'] = self.data['marital-status'].replace(marital_status_labels, 'Single')

        marital_status_labels = ['Married-AF-spouse', 'Married-civ-spouse']
        self.data['marital-status'] = self.data['marital-status'].replace(marital_status_labels, 'Couple')

        self.data['marital-status'] = self.data['marital-status'].map({'Couple': 0, 'Single': 1})

        #6 Mapping 'relationship' values to numeric labels
        relationships = {'Unmarried': 0, 'Wife': 1, 'Husband': 2, 'Not-in-family': 3, 'Own-child': 4, 'Other-relative': 5}
        self.data['relationship'] = self.data['relationship'].map(relationships)

        #7 Mapping 'race' values to numeric labels
        races = {'White': 0, 'Amer-Indian-Eskimo': 1, 'Asian-Pac-Islander': 2, 'Black': 3, 'Other': 4}
        self.data['race'] = self.data['race'].map(races)

        #8 Adding a new column 'employment-type' based on 'workclass' values
        def create_employment_type_field(x):
            emp_type = x['workclass']
            if emp_type in ['Federal-gov', 'Local-gov', 'State-gov']:
                return 'govt'
            elif emp_type == 'Private':
                return 'private'
            elif emp_type in ['Self-emp-inc', 'Self-emp-not-inc']:
                return 'self-employed'
            else:
                return 'without-pay'
            
        self.data['employment-type'] = self.data.apply(create_employment_type_field, axis=1)
        employment_types = {'govt': 0, 'private': 1, 'self-employed': 2, 'without-pay': 3, }

        self.data['employment-type'] = self.data['employment-type'].map(employment_types)
        self.data.drop(labels=['workclass', 'occupation', 'education'], axis=1, inplace=True)

        #9 Mapping 'capital-gain' and 'capital-loss' values to binary labels
        self.data.loc[(self.data['capital-gain'] > 0), 'capital-gain'] = 1
        self.data.loc[(self.data['capital-gain'] == 0), 'capital-gain'] = 0

        self.data.loc[(self.data['capital-loss'] > 0), 'capital-loss'] = 1
        self.data.loc[(self.data['capital-loss'] == 0), 'capital-loss'] = 0


        #10 Removing 'fnlwgt' column as it's not that important to the case in hand
        self.data.drop(['fnlwgt'], axis=1, inplace=True)
        
    def __save_clean(self):
        working_path = os.path.join(config.get('working_dir'), self.name, self.name+'_cleaned.csv')
        self.data.to_csv(working_path, index=False)


# class MyDataset(Dataset):
#     def __init__(self):
#         super(MyDataset, self).__init__()
#         self.name = 'mydataset'
#         self.transform = img_transform
        

#         # Create required directories    
#         working_path = os.path.join(config.get('working_dir'), self.name)
#         if not os.path.exists(working_path): os.makedirs(working_path,  exist_ok=True)
        
#         models_path = os.path.join(config.get('models_dir'), self.name)
#         if not os.path.exists(models_path): os.makedirs(models_path, exist_ok=True)

#         # Load Data
#         path = os.path.join(config.get('datasets_dir'), self.name, 'mydataset.csv')
        
#         self.data = pd.read_csv(path)
#         self.__preprocess_data()       
#         self.__normalize_data()
    
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data.iloc[idx]
#         sample = torch.tensor(sample.values, dtype=torch.float32)
#         return sample

#     def __normalize_data(self):
#         scaler = MinMaxScaler()
#         self.data[['age', 'weight', 'height']] = scaler.fit_transform(self.data[['age', 'weight', 'height']])

#     def __preprocess_data(self):
#         # self.data.drop(columns=['age'])
#         pass



class CustomMNIST(Dataset):
    def __init__(self):
        super(CustomMNIST, self).__init__()
        self.name = 'mnist'
        self.transform = img_transform
        

        # Create required directories 
        input_path = os.path.join(ROOT_DIR, 'data', 'input', self.name)
        check_and_mkdir_if_necessary(input_path)

        self.data = datasets.MNIST(root=input_path, download=True, train=True, transform=img_transform)
        
        self.__preprocess_data()       
        self.__normalize_data()

        self.num_features = 784

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        # plt.imshow(image.view(28, 28))
        # plt.show()
        image = image.view(-1) # Convert the image to 2D array
        return image

    def __normalize_data(self):
        pass

    def __preprocess_data(self):
        # MNIST does not require preprocessing other than appaying `transform`
        pass