# AI Research Workbench

An AI-powered research assistant designed to help users explore academic papers, retrieve relevant information, understand research concepts, and investigate claims using large language models and retrieval-augmented generation (RAG).

The project combines document processing, semantic search, LLM-based question answering, and agentic workflows to make interacting with research literature more efficient.

## Overview

AI Research Workbench allows users to upload research papers and interact with their contents through natural-language questions. It uses a retrieval-augmented generation pipeline to identify relevant information from uploaded documents and generate context-aware responses.

The system also supports web-based information retrieval and research claim verification, allowing users to explore information beyond the documents they have uploaded.

## Features

- **Research Paper Interaction:** Upload and query research papers in PDF, TXT, and Markdown formats.
- **Web and ArXiv Support:** Load research material through web URLs and ArXiv paper identifiers.
- **Retrieval-Augmented Generation:** Retrieve relevant document passages to provide context for LLM-generated answers.
- **Semantic Search:** Use sentence-transformer embeddings and vector search to identify relevant document content.
- **Context-Aware Question Answering:** Ask questions about uploaded research papers and receive answers based on retrieved information.
- **Web Search:** Retrieve external information when a question requires information beyond the uploaded documents.
- **Claim Verification:** Investigate research claims using available web and ArXiv sources.
- **Session-Based Document Management:** Maintain separate document collections for different user sessions.
- **Evaluation Framework:** Evaluate response quality and retrieval performance using DeepEval metrics.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| LLM Orchestration | LangChain |
| Agent Workflow | LangGraph |
| Language Model | Groq API |
| Embeddings | Sentence Transformers |
| Vector Database | Qdrant |
| External Search | Tavily |
| Evaluation | DeepEval |

## System Workflow

1. The user uploads a research paper or provides a supported source.
2. The document is processed and divided into smaller chunks.
3. The chunks are converted into embeddings.
4. The embeddings are stored in a session-specific Qdrant collection.
5. The user's question is processed through the application workflow.
6. Depending on the question, the system performs document retrieval, web search, or claim verification.
7. Retrieved information is passed to the language model to generate a response.

## Evaluation

The project includes an evaluation workflow using DeepEval to assess different aspects of RAG performance.

The evaluation metrics include:

- Contextual Precision
- Contextual Recall
- Contextual Relevancy
- Answer Relevancy
- Faithfulness

Evaluation experiments are used to identify retrieval weaknesses and assess the quality and reliability of generated responses.

## Future Improvements

- Improve retrieval relevance and ranking.
- Investigate retrieval failures and context-selection issues.
- Evaluate performance under missing, irrelevant, and conflicting retrieval contexts.
- Compare baseline and improved RAG configurations.
- Explore additional agentic capabilities and research-oriented workflows.

## Development

This is an actively developed project focused on understanding, evaluating, and improving LLM-based research assistance systems, particularly retrieval quality and response reliability.

