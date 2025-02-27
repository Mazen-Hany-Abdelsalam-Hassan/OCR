
from src.utils import Split_the_data
import matplotlib.pyplot as plt
from src.dataset import Dataset_OCR
Data = Split_the_data()
x_train , y_train = Data['x_train'] ,Data['y_train']
data = Dataset_OCR(x_train , y_train)
plt.imshow(data[0][0][0] , cmap = 'gray')
plt.show()