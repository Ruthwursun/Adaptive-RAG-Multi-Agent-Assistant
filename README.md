# 📚 Adaptive RAG & Multi-Agent Assistant

An intelligent, interactive multi-agent assistant built with **CrewAI**, **Gradio**, and **Groq** (`llama-3.3-70b-versatile`). The system employs an advanced architecture utilizing **Adaptive RAG**, automated **Intent Routing**, and dynamic **Tool Calling** to answer document queries, search the web, or extract live job market analytics.

🔗 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Ruthwursun/MultiAgentRag_chatbot)**

---

## 🤖 System Architecture & Orchestration

The application processes user queries through a **Conditional Router** powered by LLM-driven zero-shot classification. Depending on the detected intent, it instantiates one of three isolated multi-agent workflows:
