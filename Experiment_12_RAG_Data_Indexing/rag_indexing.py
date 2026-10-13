from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def main():
    print("=" * 60)
    print("EXPERIMENT 12: RAG DATA INDEXING AND RETRIEVAL")
    print("=" * 60)

    # Step 1: Define the sample text
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

    # Step 2: Split the text into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    documents = splitter.create_documents([sample_text])

    print(f"\nNumber of chunks created: {len(documents)}")

    # Step 3: Load a local embedding model
    print("\nLoading the local embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Step 4: Create the FAISS vector store
    print("Creating the FAISS vector store...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Step 5: Search for relevant chunks
    query = "What are the effects of climate change?"

    print(f"\nQuery: {query}")
    print("\nTop 2 retrieved chunks:\n")

    results = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=2
    )

    for rank, (document, score) in enumerate(results, start=1):
        print(f"Result {rank}")
        print(f"Score: {score:.4f}")
        print(f"Chunk: {document.page_content}")
        print("-" * 60)

    # Step 6: Save the results to a text file
    output_path = (
        "Experiment_12_RAG_Data_Indexing/output/retrieval_results.txt"
    )

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("EXPERIMENT 12: RAG DATA INDEXING AND RETRIEVAL\n")
        output_file.write(f"Number of chunks created: {len(documents)}\n")
        output_file.write(f"Query: {query}\n\n")
        output_file.write("Top 2 retrieved chunks:\n\n")

        for rank, (document, score) in enumerate(results, start=1):
            output_file.write(f"Result {rank}\n")
            output_file.write(f"Score: {score:.4f}\n")
            output_file.write(f"Chunk: {document.page_content}\n")
            output_file.write("-" * 60 + "\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()