from IteratedFunctionSystems.Transformations.Transforms import LinearTransform
#from configparser import ConfigParser
from configobj import ConfigObj
import os


class Config():
    
    def __init__(self, configPath):
        self.configPath = configPath

        if os.path.isfile(configPath) is False:
            print("Config file doesnt exist")
            exit()

        self.parser = ConfigObj(configPath)
        

    def getSectionValue(self, section, key):
        return self.parser.get(section)[key]




def getStringtoBool(string):
    if string.upper() == "FALSE":
        return False
    if string.upper() == "TRUE":
        return True


def getStringtoList_Float(string, sep = ","):
    myList = []
    items = string.split(sep = sep)
    for _item in items:
        myList.append(float(_item))

    return myList

def convertListItemstoFloat(mylist):
    return [float(_item) for _item in mylist] 

