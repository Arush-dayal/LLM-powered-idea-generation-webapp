import arxiv
import numpy as np
import faiss
import openai
from google.genai import Client, types
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity


class ResearchPaperIdea(BaseModel):
    title: str
    brief_description: str

def get_embedding(client: Client, text, model="gemini-embedding-exp-03-07"):
    text = text.replace("\n", " ")
    embeddings = client.models.embed_content(contents = text, model=model).embeddings
    return [embedding.values for embedding in embeddings][0]


def fetch_arxiv_papers(query, max_results=50):

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "abstract": result.summary,
            "url": result.entry_id
        })
    return papers


def embed_papers(client, papers):
    embeddings = []
    for paper in papers[:2]:
        text = paper["title"] + "\n" + paper["abstract"]
        embedding = get_embedding(client, text)
        embeddings.append(np.array(embedding, dtype=np.float32))
    return np.array(embeddings)


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def semantic_search(client, user_query, index, paper_texts, top_k=5):
    query_embedding = get_embedding(client, user_query)
    D, I = index.search(np.array([query_embedding], dtype=np.float32), top_k)
    output = [paper_texts[i] for i in I[0]]
    print(output)
    return output



def generate_ideas_with_gemini(user_input, context_papers, client):
    context_text = "\n\n".join([f"Title: {p['title']}\nAbstract: {p['abstract']}" for p in context_papers])
    
    prompt = f"""
    You are an expert research assistant. Based on the following recent papers and the user's research interest, generate 3–5 novel, relevant, and researchable ideas.

    User interest: {user_input}

    Recent relevant papers:
    {context_text}

    Respond with clearly formatted ideas (titles + brief description) in the specified output format.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        config={
        "response_mime_type": "application/json",
        "response_schema": list[ResearchPaperIdea],
        },
        contents=prompt
    )

    output: list[ResearchPaperIdea] = response.parsed

    return output

def generate_details_with_gemini(title, description, client):

    prompt = f"""
    You are an expert research assistant. Based on the following title and description for a novel experiment, give appropriate datasets that can be used
    and detail the experiment setup/design.

    title: {title},
    description: {description}

    Respond with clear content in formatted markdown.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=prompt
    )

    return response.text

