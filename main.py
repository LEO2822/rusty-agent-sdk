from rusty_agent_sdk import greet


def main():
    name = input("Enter your name: ")
    print(greet(name))


if __name__ == "__main__":
    main()
