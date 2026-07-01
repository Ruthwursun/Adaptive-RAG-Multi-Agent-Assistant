import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


from langchain_community.document_loaders import PyPDFLoader
import os
import requests
import gradio as gr
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from crewai import Agent, LLM, Task, Crew, Process
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool

# Patch CrewAI cache breakpoint (kept from original notebook)
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

# ---------------------------------------------------------------------------
# Secrets — set these under Space Settings > Variables and secrets
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("Groq_api_key")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
os.environ["RAPIDAPI_KEY"] = RAPIDAPI_KEY or ""

# ---------------------------------------------------------------------------
# 1. Load + embed the PDF (runs once at Space startup)
# ---------------------------------------------------------------------------
PDF_PATH = "RagNotes.pdf"          # place this file next to app.py in the repo
PERSIST_DIR = "./chroma_db"        # local persistent store inside the Space

loader = PyPDFLoader(PDF_PATH)
doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(doc)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = Chroma(
    collection_name="research_collection",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

if vector_store._collection.count() == 0:
    vector_store.add_documents(documents=texts)


def retrieve_Context(query: str, k: int):
    retrieved_doc = vector_store.similarity_search(query, k=k)
    docs_content = ""
    for d in retrieved_doc:
        docs_content += f"source{d.metadata}\n"
        docs_content += f"Content{d.page_content}"
    return docs_content, retrieved_doc


# ---------------------------------------------------------------------------
# 2. Tools
# ---------------------------------------------------------------------------
class RetrieverInput(BaseModel):
    query: str = Field(..., description="The search query to look up in RagNotes.pdf")
    k: int = Field(default=4, description="Number of relevant chunks to retrieve")


class DocumentRetrieverTool(BaseTool):
    name: str = "Document Retriever Tool"
    description: str = (
        "Searches the embedded RagNotes.pdf document and returns the most "
        "relevant chunks of content along with their source metadata (page number, etc)."
    )
    args_schema: type[BaseModel] = RetrieverInput

    def _run(self, query: str, k: int = 4) -> str:
        docs_content, _ = retrieve_Context(query, k)
        return docs_content


retriever_tool = DocumentRetrieverTool()
search_tool = SerperDevTool(api_key=SERPER_API_KEY)


class JSearchInput(BaseModel):
    query: str = Field(..., description="Job search query, e.g. 'Data Analyst jobs in Chennai'")
    num_pages: int = Field(default=1, description="Number of result pages to fetch (each page ~10 jobs)")


class JSearchTool(BaseTool):
    name: str = "JSearch Job Finder"
    description: str = (
        "Searches for real, current job listings using the JSearch API (Google for Jobs data). "
        "Use this to find open positions matching a role, skill, or location."
    )
    args_schema: type[BaseModel] = JSearchInput

    def _run(self, query: str, num_pages: int = 1) -> str:
        url = "https://jsearch.p.rapidapi.com/search"
        headers = {
            "X-RapidAPI-Key": os.environ["RAPIDAPI_KEY"],
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
        }
        params = {"query": query, "page": "1", "num_pages": str(num_pages)}

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            return f"Error fetching jobs: {response.status_code} - {response.text}"

        data = response.json().get("data", [])
        if not data:
            return "No job listings found for this query."

        results = ""
        for job in data:
            results += f"Title: {job.get('job_title')}\n"
            results += f"Company: {job.get('employer_name')}\n"
            results += f"Location: {job.get('job_city', 'N/A')}, {job.get('job_country', 'N/A')}\n"
            results += f"Employment Type: {job.get('job_employment_type', 'N/A')}\n"
            results += f"Posted: {job.get('job_posted_at_datetime_utc', 'N/A')}\n"
            results += f"Apply Link: {job.get('job_apply_link', 'N/A')}\n"
            results += f"Description: {(job.get('job_description') or '')[:300]}...\n\n"

        return results


jsearch_tool = JSearchTool()

# ---------------------------------------------------------------------------
# 3. LLM + Agents
# ---------------------------------------------------------------------------
llm = LLM(model="groq/llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

retriever_agent = Agent(
    role="Document Retriever",
    goal="Retrieve the most relevant and accurate information from the embedded document to answer user queries",
    backstory=(
        "You are a meticulous research assistant with expertise in information retrieval. "
        "You excel at understanding what the user is actually asking, searching the embedded "
        "document/vector store for the most relevant chunks, synthesizing retrieved passages "
        "into a clear, accurate answer, citing which part of the document the answer came from, "
        "and saying clearly when the document does not contain the answer, rather than guessing. "
        "You never fabricate information that isn't supported by the retrieved content."
    ),
    verbose=True,
    llm=llm,
    tools=[retriever_tool],
    allow_delegation=False,
)

serper_search_agent = Agent(
    role="Web Research Specialist",
    goal="Search the web using Google search results when the internal document doesn't fully answer the user's query",
    backstory=(
        "You are a skilled researcher who specializes in finding reliable information online. "
        "You only step in when the internal document (RagNotes.pdf) lacks sufficient information "
        "to answer the query. You cross-check sources, prefer authoritative and recent results, "
        "and clearly cite the URLs you used. You never present unverified claims as fact."
    ),
    verbose=True,
    llm=llm,
    tools=[search_tool],
    allow_delegation=False,
)

job_search_agent = Agent(
    role="Job Search Specialist",
    goal="Find real, current, relevant job openings that match the user's role, skills, and location preferences",
    backstory=(
        "You are an experienced job search consultant with access to live job listings. "
        "You understand how to translate a candidate's background and target roles into effective "
        "search queries, and you filter results to show only genuinely relevant, recent openings. "
        "You never fabricate job listings — you only report what the tool actually returns."
    ),
    verbose=True,
    llm=llm,
    tools=[jsearch_tool],
    allow_delegation=False,
)

# ---------------------------------------------------------------------------
# 4. Tasks
# ---------------------------------------------------------------------------
task_retrieve = Task(
    description=(
        "The user has asked the following question: {query}"
        "1. Use the Document Retriever Tool to search RagNotes.pdf for content relevant to the query"
        "2. Carefully read the retrieved chunks and their source metadata"
        "3. Synthesize the information into a clear, accurate answer"
        "4. If the retrieved content does not fully answer the query, state clearly what is missing rather than guessing"
        "5. Cite the source metadata (e.g. page number) for each piece of information used"
        "Output format:"
        "## Answer"
        "- Summary: ..."
        "- Details: ..."
        "- Sources: ..."
    ),
    expected_output="A clear markdown answer grounded in the retrieved document content, with sources cited",
    agent=retriever_agent,
)

task_serper_search = Task(
    description=(
        "You are given the result of a document retrieval attempt for this question: {query}"
        "1. Carefully review the retrieved document context provided to you"
        "2. Judge whether it sufficiently and accurately answers the query"
        "3. If it DOES fully answer the query, simply restate that answer — do NOT search the web"
        "4. If it does NOT fully answer the query (missing, vague, or irrelevant content), "
        "use the Serper Search tool to find the missing information online"
        "5. Clearly distinguish which parts of your answer came from the document vs. the web"
        "Output format:"
        "## Final Answer"
        "- Summary: ..."
        "- Details: ..."
        "- Sources: (document page numbers and/or URLs, labeled by origin)"
    ),
    expected_output="A complete markdown answer, using document content when sufficient and web search only when necessary, with sources clearly labeled by origin",
    agent=serper_search_agent,
    context=[task_retrieve],
)

task_general_search = Task(
    description=(
        "The user has asked the following question: {query}"
        "1. Use the Serper Search tool to find accurate, current information online"
        "2. Synthesize the results into a clear, accurate answer"
        "3. Cite the URLs of the sources used"
        "Output format:"
        "## Answer"
        "- Summary: ..."
        "- Details: ..."
        "- Sources: (URLs)"
    ),
    expected_output="A clear markdown answer grounded in web search results, with source URLs cited",
    agent=serper_search_agent,
)

task_job_search = Task(
    description=(
        "The user wants to find job openings based on: {query}"
        "1. Use the JSearch Job Finder tool to search for relevant current job listings"
        "2. Review the results and select the most relevant and recent openings"
        "3. Exclude any listings that are clearly irrelevant or expired"
        "4. Summarize each relevant job clearly with title, company, location, and apply link"
        "Output format:"
        "## Job Search Results"
        "- Job Title: ..."
        "  Company: ..."
        "  Location: ..."
        "  Type: ..."
        "  Posted: ..."
        "  Apply: ..."
    ),
    expected_output="A clear markdown list of relevant, current job openings with company, location, and apply links",
    agent=job_search_agent,
)


# ---------------------------------------------------------------------------
# 5. Router + runner
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 5. Router + runner
# ---------------------------------------------------------------------------
def classify_query(query: str) -> str:
    """Classify the user query into one of: document, job, general"""
    classification_prompt = f"""Classify the following user query into exactly ONE category:
- "document": if it's asking to explain/define/describe a concept likely covered in a study/notes document (e.g. Machine Learning, algorithms, theory)
- "job": if it's asking to find job openings, vacancies, or career opportunities
- "general": if it's a general knowledge or current-events question unrelated to the document or jobs

Query: "{query}"

Respond with ONLY one word: document, job, or general."""

    # Force a clean, string-based classification
    response = llm.call(classification_prompt)
    cleaned_response = response.strip().lower()
    
    if "document" in cleaned_response:
        return "document"
    elif "job" in cleaned_response:
        return "job"
    else:
        return "general"


async def run_query(query: str):
    clean_query = query.strip()
    category = classify_query(clean_query)

    # We ensure inputs are explicitly structured for CrewAI's execution context
    task_inputs = {"query": clean_query}

    if category == "document":
        crew = Crew(
            agents=[retriever_agent, serper_search_agent],
            tasks=[task_retrieve, task_serper_search],
            process=Process.sequential,
            verbose=True,
        )
    elif category == "job":
        crew = Crew(
            agents=[job_search_agent],
            tasks=[task_job_search],
            process=Process.sequential,
            verbose=True,
        )
    else:  # general
        crew = Crew(
            agents=[serper_search_agent],
            tasks=[task_general_search],
            process=Process.sequential,
            verbose=True,
        )

    # Execute the crew layout sequentially
    result = await crew.kickoff_async(inputs=task_inputs)
    
    # CrewAI 0.x+ returns a CrewOutput object. .raw contains the clean final markdown string.
    if hasattr(result, 'raw') and result.raw:
        final_text = result.raw
    else:
        final_text = str(result)
        
    return category, final_text


# ---------------------------------------------------------------------------
# 6. Gradio UI
# ---------------------------------------------------------------------------
async def gradio_query(query):
    if not query.strip():
        return "⚠️ Please enter a valid question."
    
    try:
        category, result = await run_query(query)
        
        # This formatting layout forces Gradio's gr.Markdown component to render clean markdown
        formatted_output = f"""### 🤖 Routing Analysis
* **Category Routed:** `{category.upper()}`

---

{result}
"""
        return formatted_output
        
    except Exception as e:
        return f"❌ **An error occurred during execution:**\n```text\n{str(e)}\n```"


with gr.Blocks(title="RAG Multi-Agent Assistant") as demo:
    gr.Markdown("# 📚 RAG Multi-Agent Assistant")
    gr.Markdown(
        "Ask a question from your document, a general knowledge question, "
        "or search for jobs (e.g. *'Data Analyst jobs in Chennai for freshers'*)."
    )
    with gr.Row():
        query_input = gr.Textbox(
            label="Your question",
            placeholder="e.g. Explain gradient boosting, or: Data Analyst jobs in Chennai",
            scale=4,
        )
        submit_btn = gr.Button("Ask", variant="primary", scale=1)

    # Standardize output assignment structure
    output_display = gr.Markdown(label="Answer")

    submit_btn.click(fn=gradio_query, inputs=query_input, outputs=output_display)
    query_input.submit(fn=gradio_query, inputs=query_input, outputs=output_display)

if __name__ == "__main__":
    demo.launch(share = True)
