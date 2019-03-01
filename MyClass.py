class MyClass:
    """A simple example class"""
    

    def f(self):
        return 'hello world'


if __name__ == "__main__":
    print(MyClass.i)
    print(MyClass.f)
    print(MyClass.__doc__)
    MyClass.i = 25
    print(MyClass.i)

    x = MyClass()
    print(x.f)