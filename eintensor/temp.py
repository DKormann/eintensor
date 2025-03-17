
class C:
  def __new__(cls, *args, **kwargs):
    print('C.__new__')
    
    return object.__new__(cls)
  
  def __init__(self,name):
    print('C.__init__')
    self.name = name


c = C('c')

print(c)
print(c.name)

