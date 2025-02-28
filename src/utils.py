from typing import List , Dict

from pandas import DataFrame

from src.config import  DATA_DIR ,MAX_SEQ_LENGTH
from glob import glob
import os
from sklearn.model_selection import train_test_split
import pandas as pd

############
def mapping(value:List , table:dict)->[str]:
    """This function uses the lookup table for encoding
    This function used for data encoding , by changing the character into numbers and vice versa depending
    on the lookup table
    """
    result = []
    for i in value:
        result.append(table.get(i,-1 ))
    return result


def label_from_path(single_image_path:str)->List[str]:
    """
    This Function extract the label from the image path then store the labels in form of list of character
    input : image_directory
    return: list of characters
    """
    single_image_path = os.path.split(single_image_path)[-1]
    single_image_path = single_image_path.split('.')[0]
    y = [i.lower() for i in single_image_path]
    padding = ["_"]*(MAX_SEQ_LENGTH-len(y) )
    y.extend(padding)
    return y
################################3
def find_label_set()->Dict[str , List[List[str]]]:
    """
    extract all the label of data in form of list of lists of characters
    :return: list of lists of characters
    """
    image_dir = glob(DATA_DIR)
    label = list(map(label_from_path ,image_dir))
    return {'X':image_dir ,'Y':label }





def  create_lookup_table(labels:[[str]]):
    """
    Create a lookup table for decoding and encoding each of the labels
    input : labels
    return:lookup table
    """
    chars = [char for word in labels for char in word]
    unique_char = sorted(list(set(chars)))
    encoding_table  = {char : num+1 for num,char in enumerate(unique_char) if char !='_'  }
    decoding_table = {num+1 : char for num, char in enumerate(unique_char) if char !='_'}
    encoding_table['_'] = 0
    decoding_table[0] = '_'
    df: DataFrame = pd.DataFrame(encoding_table.items() , columns = ['Char' , "Code"] )

    base_dir = os.path.dirname(os.path.dirname(__file__))
    assets_dir = os.path.join(base_dir ,"assets")
    os.makedirs(assets_dir , exist_ok=True)
    save_dir = os.path.join(assets_dir , 'label.csv')
    df.to_csv(save_dir, index=False)
    return encoding_table , decoding_table

def create_data()->Dict[str, List[List[str]]]:
    """
    This function used for returning the encoded and decoded version of labels

    :return: Dictionary of list of string for the encoded and decoded version of data
    """
    data = find_label_set()
    X, Y = data['X'], data['Y']
    encoding_table , decoding_table = create_lookup_table(Y)
    encoded_label = map(lambda x : mapping(x, encoding_table) , Y )
    return { "img_dir": X ,"encoded_label":list(encoded_label) ,"decoded_label" :Y }


def Split_the_data():
    Data = create_data()
    img_dir, encoded_label, decoded_label = Data['img_dir'], Data['encoded_label'], Data['decoded_label']
    x_train, x_test, y_train, y_test = train_test_split(img_dir, encoded_label, random_state=1234)
    return  {"x_train":x_train, "y_train":y_train ,
             "x_test":x_test , "y_test":y_test}

