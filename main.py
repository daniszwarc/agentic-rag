from dotenv import load_dotenv
load_dotenv()

from graph.graph import app



def main():
    print("Hello from agentic-rag!")
    print(app.invoke(input={"question": "What is the best acoustic guitar?"}))


if __name__ == "__main__":
    main()
