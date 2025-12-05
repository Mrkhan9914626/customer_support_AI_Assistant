from agents import function_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import requests
import os
import json
from typing import List
from pathlib import Path


load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = "rag-work-1"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Load and process documents
github_url = "https://raw.githubusercontent.com/Mrkhan9914626/customer_support_AI_Assistant/master/TechFlow_Solutions.pdf"
project_root = Path(__file__).parent
file_path = project_root / "temp_file.pdf"


response = requests.get(github_url , timeout=10)
if response.status_code == 200:
    with open(file_path, "wb") as f:
        f.write(response.content)
else:
    raise Exception(f"Failed to download PDF from GitHub. Status code: {response.status_code}")
loader = PyPDFLoader(str(file_path))
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, 
    chunk_overlap=200, 
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Add documents only if needed
try:
    ids = vector_store.add_documents(documents=all_splits)
except Exception as e:
    print(f"Documents may already exist: {e}")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    api_key=api_key,
    model="gemini-2.5-flash",
    temperature=1.2,  # Lower temperature for more consistent retrieval
    top_p=0.95,
    max_output_tokens=1600,
)

# Store all document content for keyword search
ALL_DOCUMENTS = [doc.page_content for doc in all_splits]

# Document metadata extraction
DOCUMENT_SECTIONS = {
    "pricing": ["pricing", "plan", "cost", "payment", "subscription", "tier"],
    "features": ["feature", "capability", "functionality", "kanban", "gantt", "workflow"],
    "support": ["support", "contact", "help", "email", "phone", "response time"],
    "faq": ["faq", "question", "how do i", "can i", "what is"],
    "policies": ["policy", "refund", "privacy", "terms", "shipping", "return"],
    "technical": ["api", "integration", "browser", "system", "security", "compliance"],
    "troubleshooting": ["error", "issue", "problem", "fix", "troubleshoot", "bug"]
}


def calculate_relevance_score(doc_content: str, query: str) -> float:   #Too identify 
    """Calculate a simple relevance score based on keyword overlap"""
    query_terms = set(query.lower().split())
    doc_terms = set(doc_content.lower().split())
    
    if not query_terms:
        return 0.0
    
    overlap = len(query_terms.intersection(doc_terms))
    return overlap / len(query_terms)


def identify_query_category(query: str) -> List[str]:
    """Identify which document categories are relevant to the query"""
    query_lower = query.lower()
    relevant_categories = []
    
    for category, keywords in DOCUMENT_SECTIONS.items():
        if any(keyword in query_lower for keyword in keywords):
            relevant_categories.append(category)
    
    return relevant_categories if relevant_categories else ["general"]


@function_tool
def semantic_search(query: str, num_results: int = 5) -> str:
    """
    Perform semantic similarity search in the documentation database.
    Best for conceptual queries and finding related information.
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 5, max: 10)
    
    Returns:
        JSON string with search results and metadata
    """
    try:
        num_results = min(num_results, 10)  # Cap at 10
        results = vector_store.similarity_search_with_score(query, k=num_results)
        
        if not results:
            return json.dumps({
                "status": "no_results",
                "message": "No relevant documents found. Try reformulating your query or using keyword_search.",
                "suggestion": "Consider using more specific terms or searching for related concepts."
            })
        
        formatted_results = []
        for i, (doc, score) in enumerate(results, 1):
            relevance_score = calculate_relevance_score(doc.page_content, query)
            formatted_results.append({
                "rank": i,
                "content": doc.page_content,
                "similarity_score": float(1 - score),  # Convert distance to similarity
                "relevance_score": float(relevance_score),   #The similarity_score is based on vector distance, while relevance_score is based on keyword overlap(means how many words of the query appears in the document.)
                "length": len(doc.page_content)
            })
        
        # Calculate average relevance
        avg_relevance = sum(r["relevance_score"] for r in formatted_results) / len(formatted_results)
        # To understand the average relevance look at the above 'results' variable this variable return a tuple of document and its similarity score.Then a for loop is iterated over this tuple and each document is assigned a relevance score based on keyword overlap with the query. Finally, the average relevance score is calculated across all returned documents.
        return json.dumps({
            "status": "success",
            "num_results": len(formatted_results),
            "average_relevance": float(avg_relevance),
            "query_categories": identify_query_category(query),
            "results": formatted_results,
            "suggestion": "Low relevance? Try keyword_search or refine your query." if avg_relevance < 0.3 else ""
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Search failed: {str(e)}"
        })


@function_tool
def keyword_search(query: str, num_results: int = 5) -> str:
    """
    Perform exact keyword/phrase search in the documentation.
    Best for finding specific terms, names, or exact phrases.
    
    Args:
        query: The search query (can include exact phrases)
        num_results: Number of results to return (default: 5)
    
    Returns:
        JSON string with search results
    """
    try:
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        matches = []
        for doc_content in ALL_DOCUMENTS:
            doc_lower = doc_content.lower()
            
            # Check for exact phrase match
            if query_lower in doc_lower:
                score = 1.0
            else:
                # Calculate term match score
                term_matches = sum(1 for term in query_terms if term in doc_lower)
                score = term_matches / len(query_terms) if query_terms else 0
            
            if score > 0:
                matches.append({
                    "content": doc_content,
                    "match_score": float(score),
                    "length": len(doc_content)
                })
        
        # Sort by score and take top results
        matches.sort(key=lambda x: x["match_score"], reverse=True)
        top_matches = matches[:num_results]
        
        if not top_matches:
            return json.dumps({
                "status": "no_results",
                "message": f"No documents contain the keywords: '{query}'",
                "suggestion": "Try semantic_search for conceptual matches or use different keywords."
            })
        
        return json.dumps({
            "status": "success",
            "num_results": len(top_matches),
            "query": query,
            "results": top_matches
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Keyword search failed: {str(e)}"
        })


@function_tool
def get_document_section(section_name: str) -> str:
    """
    Retrieve specific sections of documentation by category.
    
    Available sections:
    - pricing: Pricing plans and billing information
    - features: Product features and capabilities
    - support: Contact information and support channels
    - faq: Frequently asked questions
    - policies: Company policies (refund, privacy, terms)
    - technical: API, integrations, technical specifications
    - troubleshooting: Error messages and solutions
    
    Args:
        section_name: The section category to retrieve
    
    Returns:
        JSON string with section content
    """
    try:
        section_lower = section_name.lower()
        
        if section_lower not in DOCUMENT_SECTIONS:
            return json.dumps({
                "status": "invalid_section",
                "message": f"Section '{section_name}' not found",
                "available_sections": list(DOCUMENT_SECTIONS.keys())
            })
        
        keywords = DOCUMENT_SECTIONS[section_lower]
        relevant_docs = []
        
        for doc_content in ALL_DOCUMENTS:
            doc_lower = doc_content.lower()
            if any(keyword in doc_lower for keyword in keywords):
                relevant_docs.append(doc_content)
        
        if not relevant_docs:
            return json.dumps({
                "status": "no_results",
                "message": f"No documents found in section: {section_name}"
            })
        
        return json.dumps({
            "status": "success",
            "section": section_name,
            "num_documents": len(relevant_docs),
            "content": relevant_docs[:5]  # Return first 5 docs
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Section retrieval failed: {str(e)}"
        })


@function_tool
def hybrid_search(query: str, num_results: int = 5) -> str:
    """
    Perform both semantic and keyword search, then combine results.
    Best for comprehensive information retrieval.
    
    Args:
        query: The search query
        num_results: Number of results to return from each method
    
    Returns:
        JSON string with combined search results
    """
    try:
        # Get semantic results
        semantic_results = vector_store.similarity_search_with_score(query, k=num_results)
        
        # Get keyword results
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        keyword_matches = []
        for doc_content in ALL_DOCUMENTS:
            doc_lower = doc_content.lower()
            term_matches = sum(1 for term in query_terms if term in doc_lower)
            score = term_matches / len(query_terms) if query_terms else 0
            
            if score > 0:
                keyword_matches.append({
                    "content": doc_content,
                    "score": score
                })
        
        keyword_matches.sort(key=lambda x: x["score"], reverse=True)
        top_keyword = keyword_matches[:num_results]
        
        # Combine and deduplicate
        combined_docs = {}
        
        # Add semantic results
        for doc, score in semantic_results:
            content_hash = hash(doc.page_content)
            if content_hash not in combined_docs:
                combined_docs[content_hash] = {
                    "content": doc.page_content,
                    "semantic_score": float(1 - score),
                    "keyword_score": 0.0,
                    "source": "semantic"
                }
        
        # Add keyword results
        for match in top_keyword:
            content_hash = hash(match["content"])
            if content_hash in combined_docs:
                combined_docs[content_hash]["keyword_score"] = match["score"]
                combined_docs[content_hash]["source"] = "both"
            else:
                combined_docs[content_hash] = {
                    "content": match["content"],
                    "semantic_score": 0.0,
                    "keyword_score": match["score"],
                    "source": "keyword"
                }
        
        # Calculate combined score and sort
        results = []
        for doc_data in combined_docs.values():
            combined_score = (doc_data["semantic_score"] * 0.6 + doc_data["keyword_score"] * 0.4)
            results.append({
                **doc_data,
                "combined_score": float(combined_score)
            })
        
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return json.dumps({
            "status": "success",
            "num_results": len(results),
            "query": query,
            "results": results[:num_results * 2]  # Return more results since it's hybrid
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Hybrid search failed: {str(e)}"
        })


@function_tool
def answer_with_context(query: str, context: str) -> str:
    """
    Generate an answer using provided context from previous searches.
    Use this after gathering sufficient information from search tools.
    
    Args:
        query: The user's original question
        context: The relevant context from previous searches
    
    Returns:
        Generated answer based on the context
    """
    try:
        custom_template = """You are a helpful assistant for TechFlow Solutions.

Based ONLY on the following context, answer the user's question accurately and concisely.

If the context doesn't contain enough information to answer completely, acknowledge what you can answer and what information is missing.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt_template = ChatPromptTemplate.from_template(custom_template)
        formatted_input = prompt_template.invoke({
            "context": context,
            "question": query
        })
        
        response = llm.invoke(formatted_input)
        
        return json.dumps({
            "status": "success",
            "answer": response.content,
            "context_length": len(context)
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Answer generation failed: {str(e)}"
        })        