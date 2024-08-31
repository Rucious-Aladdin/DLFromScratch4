class MyHelloWorld:
    def __init__(self):
        self.message = "Hello, World!"

    def say(self):
        print(self.message)
    
    def say_name(self, name):
        print(f"Hello, {name}!")

if __name__ == "__main__":
    mhw = MyHelloWorld()
    mhw.say()
    mhw.say_name("Alice")
    mhw.say_name("Bob")

    mhw2 = MyHelloWorld()
    mhw2.say()
    mhw2.say_name("Charlie")
    mhw2.say_name("David")
    