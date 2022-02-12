from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

# recommended color for different digits
color_mapping = {0:'red',1:'green',2:'blue',3:'yellow',4:'magenta',5:'orangered',
                6:'cyan',7:'purple',8:'gold',9:'pink'}

def plot2d(data,label,split='train'):
    # 2d scatter plot of the hidden features
    for i in range(len(data)):
        plt.scatter(data[i][0],data[i][1],c=color_mapping[label[i]],s=20)
    plt.show()

def plot3d(data,label,split='train'):
    # 3d scatter plot of the hidden features
    project = plt.axes(projection="3d")
    for i in range(len(data)):
        project.scatter(data[i][0],data[i][1],data[i][2],c=color_mapping[label[i]],s=22)
    plt.show()
    
