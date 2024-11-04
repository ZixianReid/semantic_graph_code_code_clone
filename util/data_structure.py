
class Queue():
    """Construct the queue structure using list
    """
    def __init__(self) -> None:
        self.__list = list()
    
    def is_empty(self):
        return self.__list == []
    
    def push(self, data):
        self.__list.append(data)
    
    def pop(self):
        if self.is_empty():
            return False
        
        return self.__list.pop(0)


class Stack():
    """Construct the stack structure using list
    """
    def __init__(self) -> None:
        self.__list = list()
    
    def is_empty(self):
        return self.__list == []
    
    def push(self, data):
        self.__list.append(data)
    
    def pop(self):
        if self.is_empty():
            return False
        return self.__list.pop()