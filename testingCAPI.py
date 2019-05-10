# Though it looks like an ordinary python import, the addList module
# is implemented in C
import addList

li = [1, 2, 3, 4, 5]
print("Sum of List - {} = {}".format(li, addList.add(li)))
