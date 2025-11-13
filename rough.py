# a={
#     "name":"Ram",
#     "age":20
# }
# print(a["name"])

from dataclasses import dataclass

@dataclass
class Hello:
    name:str="Ram"
    age:int="21"
    
h1=Hello()
h2=Hello("Sita",18)

print(type(h1))
print(h2.name)
