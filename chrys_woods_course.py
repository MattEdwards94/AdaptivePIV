import sys

# simple print statement showing how to print hello world
a = "Hello"
b = "from"
c = "Python!"

print("%s %s %s" % (a, b, c))

# basic loop to show how to create a range of values to loop through
for i in range(1, 11):
    five_times_i = 5 * i
    print("5 times %s equals %s" % (i, five_times_i))


# loop showing how to prevent a newline from being formed, such that we can
# manipulate how printing is performed.
for i in range(1, 4):
    for j in range(1, 4):
        i_times_j = i * j
        print(i_times_j, end="")

    print("\n", end="")


# lists

my_list = ["cat", "dog", 261]
# demonstrates behaviour for printing lists as a whole and in sections
print(my_list)
print(my_list[0])
print(my_list[1])
print(my_list[2])
# shows how to get the length of a list
print("my_list contains %s items" % (len(my_list)))


# different ways to enumerate over a list
my_list = [3, 5, "green", 5.3, "house", 100, 1]

for elem in my_list:
    print(elem)

# enumerate provides a pair of values corresponding to the index and the value
for i, elem in enumerate(my_list):
    print("Element %s is %s" % (i, elem))


# ways to slice a list
my_list = [3, 5, "green", 5.3, "house", 100, 1]
print(my_list[-1])  # the last element of the list

print(my_list[2:5])  # elements from index 2 to (but not including) index 5

print(my_list[3:])  # elements from index 3 until the end of the list

print(my_list[:4])  # elements from the beginning to (not including) index 4

print(my_list[::2])  # every other element from the list

print(my_list[::-1])  # all the elements in reverse order


# import sys needed for this, at top of file
for i, argument in enumerate(sys.argv):
    print("Argument %s equals %s" % (i, argument))

# conditions
for i in range(1, 11):
    if i < 5:
        print("%s is less than 5." % i)

    elif i > 5:
        print("%s is greater than 5." % i)

    else:
        print("%s is equal to 5." % i)


with open(sys.argv[0]) as f:
    for linenumber, line in enumerate(f, start=1):
        # uncomment to print out all the lines of code from this file
        # print("%4s: %s" % (linenumber, line), end="")
