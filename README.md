# 📚 Adaptive RAG & Multi-Agent Assistant

An intelligent, interactive multi-agent assistant built with **CrewAI**, **Gradio**, and **Groq** (`llama-3.3-70b-versatile`). The system employs an advanced architecture utilizing **Adaptive RAG**, automated **Intent Routing**, and dynamic **Tool Calling** to answer document queries, search the web, or extract live job market analytics.

🔗 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Ruthwursun/MultiAgentRag_chatbot)**

---

## 🤖 System Architecture & Orchestration

The application processes user queries through a **Conditional Router** powered by LLM-driven zero-shot classification. Depending on the detected intent, it instantiates one of three isolated multi-agent workflows:

```mermaid
graph TD
    User([User Query]) --> Router{Intent Classification}
    
    %% Routes
    Router -->|document| Crew1[Document Analysis Crew]
    Router -->|job| Crew2[Job Discovery Crew]
    Router -->|general| Crew3[General Intelligence Crew]
    
    %% Crew 1 Details
    subgraph Crew 1: Document Processing
        Crew1 --> Agent1[Document Retriever Agent]
        Agent1 -->|Runs Local Embeddings Search| Tool1[(DocumentRetrieverTool)]
        Tool1 -->|Passes Retrieved Context| Agent2[Web Research Specialist]
        Agent2 -->|If Info Missing| Tool2[SerperDevTool]
    end
    
    %% Crew 2 Details
    subgraph Crew 2: Job Search
        Crew2 --> Agent3[Job Search Specialist]
        Agent3 -->|Queries Live Market Data| Tool3[JSearchTool]
    end
    
    %% Crew 3 Details
    subgraph Crew 3: General Knowledge
        Crew3 --> Agent4[Web Research Specialist]
        Agent4 -->|Fetches Real-Time Updates| Tool2
    end
    
    %% UI Output
    Tool2 --> UI[Gradio UI Markdown Output]
    Tool3 --> UI
    Agent2 --> UI
    
    %% Styling
    style User fill:#f9f,stroke:#333,stroke-width:2px
    style Router fill:#ff9,stroke:#333,stroke-width:2px
    style Tool1 fill:#bbf,stroke:#333,stroke-width:2px
    style UI fill:#8f8,stroke:#333,stroke-width:2px



Document Analysis Crew ("document"):  Document Retriever Agent: Conducts localized semantic vector searches via a custom DocumentRetrieverTool on RagNotes.pdf.  Web Research Specialist: Evaluates context completeness sequentially. If the document database lacks sufficient depth, it fires the SerperDevTool to cross-reference live web sources and explicitly attributes the final output by data origin.  Job Discovery Crew ("job"):  Job Search Specialist: Translates raw developer backgrounds or local parameters into optimized search arguments via a tailored JSearchTool utilizing Google for Jobs data payloads.  General Intelligence Crew ("general"):  Web Research Specialist: Directly leverages Google search capabilities to synthesize real-time, current global updates with complete URL citation tracking.  🛠️ Tech Stack & Key ParadigmsMulti-Agent Orchestration Framework: CrewAI  Core LLM Engine: Groq Cloud Engine (llama-3.3-70b-versatile)  Vector Search Engine: ChromaDB  Embedding Transformer Models: Hugging Face sentence-transformers/all-mpnet-base-v2  UI Framework Layer: Gradio  Data Chunking Strategy: LangChain RecursiveCharacterTextSplitter (chunk_size: 100, overlap: 0)  
