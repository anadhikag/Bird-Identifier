from src.rag import BirdChat

chat = BirdChat()

answer = chat.ask(
    species="001.Black_footed_Albatross",
    question="Black-footed Albatross: Where does this bird migrate?"
)

print("\nAnswer:\n")
print(answer)