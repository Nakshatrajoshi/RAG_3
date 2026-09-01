import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, DATA_DIR, TOP_K
from langchain_community.document_loaders import WebBaseLoader


class HybridRetriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = None
        self.bm25 = None
        self.chunks = None

    def load_and_index(self):
        """Load PDFs, chunk them, build both indexes"""
        print("📂 Loading PDF documents...")
        print(f"   Looking in: {DATA_DIR}")

        pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
        print(f"   PDF files found: {pdf_files}")

        if len(pdf_files) == 0:
            raise ValueError(
                f"No PDF files found in {DATA_DIR}. "
                f"Copy your PDFs there first."
            )

        documents = []
        for filename in pdf_files:
            filepath = os.path.join(DATA_DIR, filename)
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
            print(f"   ✅ Loaded: {filename} ({len(docs)} pages)")

        print(f"✅ Loaded {len(documents)} pages total")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.chunks = splitter.split_documents(documents)
        print(f"✅ Created {len(self.chunks)} chunks")

        self.vector_store = FAISS.from_documents(self.chunks, self.embeddings)
        print("✅ Semantic index built")

        tokenized_chunks = [chunk.page_content.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        print("✅ BM25 index built\n")

    def _semantic_search(self, query: str, k: int):
        results = self.vector_store.similarity_search_with_score(query, k=k)
        ranked = []
        for doc, score in results:
            for i, chunk in enumerate(self.chunks):
                if chunk.page_content == doc.page_content:
                    ranked.append(i)
                    break
        return ranked

    def _bm25_search(self, query: str, k: int):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked_indices[:k]

    def hybrid_search(self, query: str, top_k: int = TOP_K, rrf_k: int = 60):
        semantic_ranked = self._semantic_search(query, k=top_k * 2)
        bm25_ranked = self._bm25_search(query, k=top_k * 2)

        rrf_scores = {}
        for rank, idx in enumerate(semantic_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank + 1)
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank + 1)

        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        top_indices = sorted_indices[:top_k]

        return [self.chunks[i] for i in top_indices]

    def load_website(self, url: str):
        """ 
        Loads a webpage, chunks it, and ADDS it to the existing index 
        """
        print(f"🌐 Loading website: {url}")
        
        try:
            loader = WebBaseLoader(url)
            web_docs = loader.load()
            print(f"   ✅ Fetched {len(web_docs)} page(s)")
        except Exception as e:
            print(f"   ❌ Failed to load website: {e}")
            return False

        # Check if we actually got meaningful content
        total_chars = sum(len(doc.page_content) for doc in web_docs)
        print(f"   📏 Total content length: {total_chars} characters")
        
        if total_chars < 500:
            print(f"   ⚠️  Warning: Page content is very short ({total_chars} chars).")
            print(f"   ⚠️  This site may be JavaScript-rendered or blocked scraping.")
            print(f"   ⚠️  Content NOT added to knowledge base.\n")
            return False

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        new_chunks = splitter.split_documents(web_docs)
        print(f"   ✅ Created {len(new_chunks)} new chunks")

        self.chunks.extend(new_chunks)
        self.vector_store = FAISS.from_documents(self.chunks, self.embeddings)
        tokenized_chunks = [chunk.page_content.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        print(f"   ✅ Knowledge base updated: {len(self.chunks)} total chunks\n")
        return True

    def preview_website(self, url: str):
        """Fetches a URL and shows content length/preview WITHOUT adding to knowledge base"""
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            total_chars = sum(len(doc.page_content) for doc in docs)
            
            print(f"\n📏 Content length: {total_chars} characters")
            print(f"📄 First 300 chars preview:")
            print("-" * 60)
            preview = docs[0].page_content[:300] if docs else "(empty)"
            print(preview)
            print("-" * 60)
            
            if total_chars < 500:
                print("⚠️  This looks like a blocked/JS-rendered page (too little content)\n")
            else:
                print("✅ This page looks loadable\n")
                
        except Exception as e:
            print(f"❌ Failed to fetch: {e}\n")