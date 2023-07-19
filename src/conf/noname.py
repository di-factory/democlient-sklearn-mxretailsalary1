
from typing import List
from pydantic import BaseModel
class A:
    class M(BaseModel):
        pass
    
    def __init__(self):
        val : A.M = A.M()


class AA(A):
    def __init__(self):
        super().__init__()
            


class AAA(AA):
    class M(BaseModel):
      m : List[float]
      
    def __init__(self):
        super().__init__()
        

aaa = AAA()