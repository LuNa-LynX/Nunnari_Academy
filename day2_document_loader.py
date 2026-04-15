# day2_document_loader.py
import os
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama


#loading 
def load_documents():
    file1 = r"C:\Users\Swathi Krishna\Downloads\Gopalswamy_Doraiswamy_Naidu.pdf"
    file2 = r"C:\Users\Swathi Krishna\Downloads\A._P._J._Abdul_Kalam.pdf"

    loader1 = PyPDFLoader(file1)
    loader2 = PyPDFLoader(file2)

    docs1 = loader1.load()
    docs2 = loader2.load()

    documents = docs1 + docs2

    print(f"Loaded {len(documents)} pages from PDFs")

    return documents


#Split chunks
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    return chunks


#metadata
def add_metadata(chunks):
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        filename = os.path.basename(source).lower()

        chunk.metadata["filename"] = filename
        chunk.metadata["page_number"] = chunk.metadata.get(
            "page", chunk.metadata.get("page_number", 0)
        )
        chunk.metadata["upload_date"] = datetime.now().strftime("%Y-%m-%d")

        if "naidu" in filename:
            chunk.metadata["source_type"] = "paper"
        elif "kalam" in filename:
            chunk.metadata["source_type"] = "notes"
        else:
            chunk.metadata["source_type"] = "textbook"

    print("Metadata attached successfully")

    return chunks


#Filter function 
def filter_chunks(chunks, **filters):
    result = []

    for chunk in chunks:
        match = True

        for key, value in filters.items():
            chunk_value = chunk.metadata.get(key)

            if isinstance(chunk_value, str) and isinstance(value, str):
                if chunk_value.lower() != value.lower():
                    match = False
                    break
            else:
                if chunk_value != value:
                    match = False
                    break

        if match:
            result.append(chunk)

    return result


#Qwen
def init_llm():
    return Ollama(model="qwen:0.5b")


#question
def ask_question(llm, chunks, query):
    if not chunks:
        return "No relevant chunks found."

    context = "\n\n".join([c.page_content for c in chunks[:3]])

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""

    return llm.invoke(prompt)


#test block
if __name__ == "__main__":

    documents = load_documents()

    chunks = split_documents(documents)

    chunks = add_metadata(chunks)

    #debug
    print("\nAvailable filenames:")
    print(set([c.metadata["filename"] for c in chunks]))

    print("\nSample metadata:")
    print(chunks[0].metadata)

    #filter
    filtered = filter_chunks(
        chunks,
        filename="gopalswamy_doraiswamy_naidu.pdf"
    )

    print(f"\nFiltered chunks count: {len(filtered)}\n")

    for i, chunk in enumerate(filtered[:2]):
        print(f"--- Chunk {i+1} ---")
        print("Metadata:", chunk.metadata)
        print("Preview:", chunk.page_content[:150])
        print()

    #Qwen test
    llm = init_llm()

    answer = ask_question(
        llm,
        filtered,
        "Give a short summary of this document"
    )

    print("\nQwen Response:\n")
    print(answer)




"""
OUTPUT:

Loaded 34 pages from PDFs
Created 149 chunks
Metadata attached successfully

Available filenames:
{'a._p._j._abdul_kalam.pdf', 'gopalswamy_doraiswamy_naidu.pdf'}

Sample metadata:
{'producer': 'Skia/PDF m145', 'creator': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/145.0.0.0 Safari/537.36', 'creationdate': '2026-03-09T06:40:02+00:00', 'title': 'Gopalswamy Doraiswamy Naidu - Wikipedia', 'moddate': '2026-03-09T06:40:02+00:00', 'source': 'C:\\Users\\Swathi Krishna\\Downloads\\Gopalswamy_Doraiswamy_Naidu.pdf', 'total_pages': 4, 'page': 0, 'page_label': '1', 'filename': 'gopalswamy_doraiswamy_naidu.pdf', 'page_number': 0, 'upload_date': '2026-04-15', 'source_type': 'paper'}

Filtered chunks count: 13

--- Chunk 1 ---
Metadata: {'producer': 'Skia/PDF m145', 'creator': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/145.0.0.0 Safari/537.36', 'creationdate': '2026-03-09T06:40:02+00:00', 'title': 'Gopalswamy Doraiswamy Naidu - Wikipedia', 'moddate': '2026-03-09T06:40:02+00:00', 'source': 'C:\\Users\\Swathi Krishna\\Downloads\\Gopalswamy_Doraiswamy_Naidu.pdf', 'total_pages': 4, 'page': 0, 'page_label': '1', 'filename': 'gopalswamy_doraiswamy_naidu.pdf', 'page_number': 0, 'upload_date': '2026-04-15', 'source_type': 'paper'}
Preview: Gopalaswamy Doraiswamy Naidu
Gopalaswamy Doraiswamy Naidu
Born 23 March 1893
Kalangal, Coimbatore District,
Madras Presidency, British India
(now Tami

--- Chunk 2 ---
Metadata: {'producer': 'Skia/PDF m145', 'creator': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/145.0.0.0 Safari/537.36', 'creationdate': '2026-03-09T06:40:02+00:00', 'title': 'Gopalswamy Doraiswamy Naidu - Wikipedia', 'moddate': '2026-03-09T06:40:02+00:00', 'source': 'C:\\Users\\Swathi Krishna\\Downloads\\Gopalswamy_Doraiswamy_Naidu.pdf', 'total_pages': 4, 'page': 0, 'page_label': '1', 'filename': 'gopalswamy_doraiswamy_naidu.pdf', 'page_number': 0, 'upload_date': '2026-04-15', 'source_type': 'paper'}
Preview: contributions were primarily industrial but also
spanned the fields of electrical, mechanical,
agricultural (hybrid cultivation) and automobile
engine


Qwen Response:

This is an article about an Indian technical redesigner and industrial pioneer who transformed imported technologies into practical and affordable innovations for India. The article also mentions the importance of these innovations in modern Indian society and economic growth.

"""
