from agent_v1 import agent_respond

def main():
    print("Welcome to the Healthcare Assistant!")
    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = agent_respond(question)
        print(f"\Answer:\n{answer}")

if __name__ == "__main__":
    main()
