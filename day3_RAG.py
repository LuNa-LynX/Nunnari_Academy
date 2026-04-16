import os
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
PDF_1 = r"C:\Users\Swathi Krishna\Downloads\Gopalswamy_Doraiswamy_Naidu.pdf"
PDF_2 = r"C:\Users\Swathi Krishna\Downloads\A._P._J._Abdul_Kalam.pdf"
CHROMA_DIR = "./chroma_db"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:1.5b"

def load_documents():
    print("Loading PDF documents...")

    if not os.path.exists(PDF_1):
        raise FileNotFoundError(f"File not found: {PDF_1}")
    if not os.path.exists(PDF_2):
        raise FileNotFoundError(f"File not found: {PDF_2}")

    loader1 = PyPDFLoader(PDF_1)
    loader2 = PyPDFLoader(PDF_2)

    documents = loader1.load() + loader2.load()

    print(f"Loaded {len(documents)} pages")
    return documents

def split_documents(documents):
    print("Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")
    return chunks

def add_metadata(chunks):
    print("Adding metadata...")

    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        filename = os.path.basename(source).lower()

        chunk.metadata["filename"] = filename
        chunk.metadata["page_number"] = chunk.metadata.get("page", 0)
        chunk.metadata["upload_date"] = datetime.now().strftime("%Y-%m-%d")

    print("Metadata added")
    return chunks
def test_embedding_model():
    print("Testing embedding model...")
    embedding = OllamaEmbeddings(model=EMBED_MODEL)

    vec = embedding.embed_query("Who is G D Naidu?")
    print(f"Embedding model works. Vector length: {len(vec)}")

    return embedding
def create_vector_store(chunks, embedding):
    print("Preparing vector store...")

    # If DB already exists, load it instead of recreating every time
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("Existing Chroma DB found. Loading it...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedding
        )
        print("Loaded existing Chroma DB")
        return vectorstore

    print(f"Creating new Chroma DB for {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_DIR
    )

    print("Stored in Chroma DB")
    return vectorstore
def create_retriever(vectorstore):
    print("Creating retriever...")
    return vectorstore.as_retriever(search_kwargs={"k": 3})
def load_llm():
    print("Loading LLM...")
    llm = ChatOllama(model=LLM_MODEL)
    print("LLM loaded")
    return llm
def ask_question(llm, retriever, question):
    print(f"\nRetrieving context for: {question}")
    docs = retriever.invoke(question)

    print("\nRetrieved chunks:")
    for i, doc in enumerate(docs):
        print(f"\nChunk {i + 1}:")
        print(doc.page_content[:300])

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question using only the context below.
If the answer is not found in the context, say "Answer not found in the given context."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    # response may be an AIMessage object
    if hasattr(response, "content"):
        return response.content
    return str(response)

if __name__ == "__main__":
    try:
        documents = load_documents()
        chunks = split_documents(documents)
        chunks = add_metadata(chunks)

        embedding = test_embedding_model()
        vectorstore = create_vector_store(chunks, embedding)
        retriever = create_retriever(vectorstore)

        llm = load_llm()

        questions = [
            "who is g d naidu?",
            "what are abdul kalam's achievements?",
            "what is abdul kalam known for?"
        ]

        for q in questions:
            print("\n" + "=" * 40)
            print("Question:", q)

            answer = ask_question(llm, retriever, q)

            print("\nAnswer:")
            print(answer)

    except Exception as e:
        print("\nERROR:")
        print(str(e))

  """
  Loading PDF documents...
Loaded 34 pages
Splitting documents into chunks...
Created 149 chunks
Adding metadata...
Metadata added
Testing embedding model...
Embedding model works. Vector length: 768
Preparing vector store...
Existing Chroma DB found. Loading it...
C:\Users\Swathi Krishna\AppData\Local\Temp\ipykernel_17612\3748087370.py:8: LangChainDeprecationWarning: The class `Chroma` was deprecated in LangChain 0.2.9 and will be removed in 1.0. An updated version of the class exists in the `langchain-chroma package and should be used instead. To use it run `pip install -U `langchain-chroma` and import as `from `langchain_chroma import Chroma``.
  vectorstore = Chroma(
Loaded existing Chroma DB
Creating retriever...
Loading LLM...
LLM loaded

========================================
Question: who is g d naidu?

Retrieving context for: who is g d naidu?

Retrieved chunks:

Chunk 1:
Kalam delivering a speech in 2010
transport and infrastructure for all parts of the
country; and (5) self-reliance in critical
technologies. These five areas are closely inter-
related and if advanced in a coordinated way, will
lead to food, economic and national security.
Kalam described a "transfo

Chunk 2:
dia.com/199284/gd-naidu-coimbatore-tamil-nadu-edison-inventor-electric-motor-india/).
thebetterindia.com. Retrieved 12 October 2025.
6. "G D Naidu - "Edison Of India" (Gopalaswamy Doraiswamy Naidu)- DailyList" (https://dailylis
t.in/g-d-naidu-edison-of-india/). 5 March 2021.
Legacy
In popular cultur

Chunk 3:
clothing, a compact disc player and a laptop. He left no will, and his possessions went to his eldest
brother after his death.[124][125]
Kalam set a target of interacting with 100,000 students during the two years after his resignation from the
post of scientific adviser in 1999. He explained, "I fe

Answer:
G D Naidu, whose full name is Gopalaswamy Doraiswamy Naidu, is widely recognized as the "Edison of India." This title refers to his remarkable contributions to the field of science and technology in India. He is celebrated for inventing electric motors, which are considered a significant achievement in electromechanical engineering.

Naidu has been involved in various activities related to innovation and education. His efforts have included working with high school students through interactions that aim to ignite their imagination and prepare them for contributing to the development of India. Additionally, he was deeply committed to promoting open-source technology over proprietary software systems, which he foresaw would benefit from widespread adoption leading to increased efficiency and accessibility in information usage.

Despite leaving behind no will upon his death, G D Naidu's legacy is enduring. He inspired numerous people across generations with his pioneering spirit and dedication to making India a better place through technological advancement.

========================================
Question: what are abdul kalam's achievements?

Retrieving context for: what are abdul kalam's achievements?

Retrieved chunks:

Chunk 1:
Kalam delivering a speech in 2010
transport and infrastructure for all parts of the
country; and (5) self-reliance in critical
technologies. These five areas are closely inter-
related and if advanced in a coordinated way, will
lead to food, economic and national security.
Kalam described a "transfo

Chunk 2:
clothing, a compact disc player and a laptop. He left no will, and his possessions went to his eldest
brother after his death.[124][125]
Kalam set a target of interacting with 100,000 students during the two years after his resignation from the
post of scientific adviser in 1999. He explained, "I fe

Chunk 3:
dia.com/199284/gd-naidu-coimbatore-tamil-nadu-edison-inventor-electric-motor-india/).
thebetterindia.com. Retrieved 12 October 2025.
6. "G D Naidu - "Edison Of India" (Gopalaswamy Doraiswamy Naidu)- DailyList" (https://dailylis
t.in/g-d-naidu-edison-of-india/). 5 March 2021.
Legacy
In popular cultur

Answer:
Kalam's achievements include being a recipient of the Bharat Ratna, an Indian highest civilian award; serving as the chairman of several prestigious bodies such as the National Academy of Engineering and the National Council for Scientific & Technological Development; leading the development of biotechnology in India through the creation of the NIV (National Institute of Virology) and NISCAV (National Institute of Science Communication); establishing 52 science schools throughout India, including those with "sky-high" goals and ambitions to motivate students; authoring books on scientific research such as "Developments in Fluid Mechanics and Space Technology" and "India 2020: A Vision for the New Millennium"; promoting open-source technology over proprietary software; initiating a target of interacting with 100,000 students after his resignation from being the scientific adviser to help them prepare for development in India.

========================================
Question: what is abdul kalam known for?

Retrieving context for: what is abdul kalam known for?

Retrieved chunks:

Chunk 1:
Kalam delivering a speech in 2010
transport and infrastructure for all parts of the
country; and (5) self-reliance in critical
technologies. These five areas are closely inter-
related and if advanced in a coordinated way, will
lead to food, economic and national security.
Kalam described a "transfo

Chunk 2:
dia.com/199284/gd-naidu-coimbatore-tamil-nadu-edison-inventor-electric-motor-india/).
thebetterindia.com. Retrieved 12 October 2025.
6. "G D Naidu - "Edison Of India" (Gopalaswamy Doraiswamy Naidu)- DailyList" (https://dailylis
t.in/g-d-naidu-edison-of-india/). 5 March 2021.
Legacy
In popular cultur

Chunk 3:
clothing, a compact disc player and a laptop. He left no will, and his possessions went to his eldest
brother after his death.[124][125]
Kalam set a target of interacting with 100,000 students during the two years after his resignation from the
post of scientific adviser in 1999. He explained, "I fe

Answer:
Kalam was known for his contributions to science, education, and national development in India. Some of his notable achievements include:

- Delivering speeches advocating for sustainable development and self-reliance in critical technologies.
- Describing a "transformative moment" when asked by Swami on how India might realise his vision of development.
- Developing faith in God and spirituality as a sixth area to overcome crime and corruption, adding it to the five areas he described earlier.
- Setting a target of interacting with 100,000 students during the two years after his resignation from the post of scientific adviser.
"""
