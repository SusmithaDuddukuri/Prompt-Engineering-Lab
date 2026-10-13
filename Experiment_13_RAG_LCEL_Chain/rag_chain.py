import os
import requests
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Load the API key from the root .env file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL = "gemini-flash-lite-latest"


def call_gemini(prompt_value):
    """Send the formatted LCEL prompt to the Gemini API."""
    if not API_KEY:
        raise ValueError(
            "GEMINI_API_KEY was not found in the project root .env file."
        )

    prompt_text = prompt_value.to_string()

    response = requests.post(
        API_URL,
        params={"key": API_KEY},
        json={
            "model": MODEL,
            "input": prompt_text,
            "generation_config": {"temperature": 0},
        },
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()

    # Extract the final text response
    for step in reversed(result.get("steps", [])):
        for item in reversed(step.get("content", [])):
            if item.get("type") == "text" and item.get("text"):
                return item["text"].strip()

    raise ValueError("No text answer was returned by Gemini.")


def format_documents(documents):
    """Format retrieved documents into prompt context."""
    return "\n\n".join(document.page_content for document in documents)


def main():
    print("=" * 65)
    print("EXPERIMENT 13: RAG CHAIN USING LCEL")
    print("=" * 65)
    print(f"LLM: {MODEL}")
    print("Embeddings: all-MiniLM-L6-v2 (local)")
    print("Retriever: FAISS, top 2 chunks")

    if not API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing. Check the root .env file."
        )

    # Reuse the same climate-change text from Experiment 12
    sample_text = """
    Climate change refers to long-term shifts in temperatures and weather
    patterns. These shifts can occur naturally, but human activities have
    been the main driver of climate change since the nineteenth century.
    Burning fossil fuels such as coal, oil, and gas releases greenhouse
    gases that trap heat in the Earth's atmosphere.

    Rising global temperatures cause glaciers and polar ice sheets to melt.
    This contributes to rising sea levels, which threaten coastal cities
    and low-lying communities. Flooding can damage homes, roads, farmland,
    and other important infrastructure.

    Climate change also affects weather patterns. Many regions experience
    more intense heat waves, droughts, and heavy rainfall. Some areas face
    stronger storms and an increased risk of wildfires. These events can
    damage ecosystems and make daily life more difficult.

    Wildlife and natural habitats are also affected. Some species must move
    to cooler regions to survive, while others face a greater risk of
    extinction. Warmer oceans can harm coral reefs and disrupt marine life.

    Climate change can affect human health and food production. Heat waves
    increase the risk of heat-related illnesses, while droughts and floods
    can reduce crop yields. Changes in rainfall and temperature can also
    affect water supplies.

    Reducing greenhouse gas emissions is important for limiting future
    warming. Renewable energy sources such as solar and wind power can
    replace some fossil fuels. Improving energy efficiency, protecting
    forests, and using public transportation can also help reduce emissions.
    """

    # 1. Split the document into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
    )
    documents = splitter.create_documents([sample_text])
    print(f"\nNumber of chunks created: {len(documents)}")

    # 2. Generate local embeddings and create the FAISS index
    print("Loading embeddings and creating FAISS index...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 3. Create a retriever that returns the top 2 chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 4. Define a context-grounded prompt
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful question-answering assistant.
Answer the question using ONLY the context provided below.
If the answer is not contained in the context, say "I don't know".
Do not add unsupported facts.

Context:
{context}

Question:
{question}

Answer:"""
    )

    # 5. Build the LCEL chain
    llm = RunnableLambda(call_gemini)

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_documents),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Test with two questions and display the retrieved sources
    questions = [
        "What causes climate change?",
        "What are the effects of climate change?",
    ]

    results_path = os.path.join(
        os.path.dirname(__file__),
        "output",
        "rag_results.txt",
    )

    with open(results_path, "w", encoding="utf-8") as output_file:
        output_file.write("EXPERIMENT 13: RAG CHAIN USING LCEL\n")
        output_file.write(f"LLM: {MODEL}\n\n")

        for question in questions:
            print("\n" + "=" * 65)
            print(f"Question: {question}")

            # Retrieve chunks for transparent source display
            retrieved_docs = retriever.invoke(question)

            # Run the LCEL chain
            answer = rag_chain.invoke(question)

            print(f"\nAnswer:\n{answer}")
            print("\nRetrieved Chunks:")

            output_file.write(f"Question: {question}\n")
            output_file.write(f"Answer:\n{answer}\n")
            output_file.write("\nRetrieved Chunks:\n")

            for index, doc in enumerate(retrieved_docs, start=1):
                print(f"\nChunk {index}: {doc.page_content}")
                output_file.write(
                    f"\nChunk {index}: {doc.page_content}\n"
                )

            output_file.write("\n" + "-" * 65 + "\n\n")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()