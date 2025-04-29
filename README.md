---
title: MedTranscript QA Agent
emoji: 🩺
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.27.1
app_file: app.py
pinned: false
---
# MedTranscript QA Agent

A medical transcript Q&A system that intelligently routes queries between document retrieval and web search to provide accurate healthcare information. Currently works on MIMIC III transcripts.

## Overview

This agent leverages two main tools:
1. **Document Retrieval**: Uses vector similarity search over medical transcripts
2. **Web Search**: Queries the public web for general healthcare information

The agent makes intelligent decisions about which tool to use based on the query type. Clinical questions about procedures and diagnoses are typically routed to the document database, while general medical information and recovery timelines are routed to web search.

## Features

- **Smart Routing**: Uses Claude 3.7 Sonnet to decide between document search and web search
- **Vector Similarity Search**: Efficient document retrieval using FAISS and sentence embeddings
- **Web Search Integration**: DuckDuckGo search for up-to-date medical information
- **Gradio UI**: User-friendly interface for interacting with the agent
- **Debug Mode**: Detailed information about tool selection and search results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/atreyee-m/medTranscript_QA_agent.git
cd medTranscript_QA_agent
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```

3. Create a `.env` file with your API key:
```
ANTHROPIC_API_KEY=your_api_key_here
```

4. Place your medical transcript data in `data/mtsamples_surgery.csv`

## Usage

Run the application:
```bash
python app.py
```

This will start the Gradio interface, accessible at http://127.0.0.1:7860 in your browser.

## File Structure

- **agent.py**: Main agent logic for routing queries and combining results
- **tools/retriever_tool.py**: Vector similarity search for document retrieval
- **tools/search_tool.py**: Web search functionality using DuckDuckGo
- **app.py**: Gradio UI for the agent
- **data/mtsamples_surgery.csv**: Medical transcription samples dataset

## Technical Details

### Document Retrieval

The document retrieval system uses:
- **SentenceTransformers**: For creating embeddings of both documents and queries
- **FAISS**: For efficient similarity search
- **Vector Similarity**: Cosine similarity with a threshold of 0.2-0.6 (adjustable)

### Web Search

The web search component uses:
- **DuckDuckGo Search API**: For querying the public web
- **Result Formatting**: Structured presentation of search results with source links

### LLM Tool Selection

The agent uses Claude 3.7 Sonnet to:
- Determine if a query needs document retrieval, web search, or both
- Route queries appropriately based on content type
- Format results into a coherent response

## How It Works

1. User submits a question through the Gradio interface
2. Claude analyzes the query to determine the appropriate tool(s)
3. Query is sent to selected tool(s) (document retrieval, web search, or both)
4. Results are formatted and returned to the user
5. If debug mode is enabled, additional information about the process is displayed

## Requirements

- Python 3.8+
- Anthropic Claude API access
- FAISS
- SentenceTransformers
- DuckDuckGo Search
- Gradio
- Pandas
- NumPy

## Future Improvements

- TBD next: Provide a better summary for the retrieved results
- Implement a hybrid search approach that blends results from both tools
- Improve document chunking for more precise retrieval
- Implement a feedback mechanism to improve tool selection over time

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Medical transcription samples from [MTSamples](https://www.mtsamples.com/)
- Built with Claude 3.7 Sonnet by Anthropic