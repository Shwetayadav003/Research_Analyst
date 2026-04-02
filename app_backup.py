import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import warnings

# Suppress warnings to keep terminal clean
warnings.filterwarnings('ignore')

load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("⚠️ Please add your Gemini API key to the .env file")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# Page config
st.set_page_config(
    page_title="Research Analyst AI",
    page_icon="🔬",
    layout="wide"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 500;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        transition: 0.2s;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .tree-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .section-item {
        margin: 0.5rem 0;
        padding: 0.3rem;
        border-left: 3px solid #667eea;
        padding-left: 1rem;
    }
    details {
        margin: 0.3rem 0;
    }
    summary {
        cursor: pointer;
        margin: 0.3rem 0;
        padding: 0.3rem;
        background: #f8f9fa;
        border-radius: 5px;
        font-weight: 500;
    }
    summary:hover {
        background: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Beautiful header
st.markdown('<div class="main-header">🔬 Research Analyst AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your Intelligent Research Assistant | Powered by Google Gemini</div>', unsafe_allow_html=True)

# Instructions expander
with st.expander("📖 How to Use", expanded=False):
    st.markdown("""
    **Follow these simple steps:**
    1. 📄 **Upload** research papers (PDF format) using the sidebar
    2. ⚙️ **Process** documents by clicking "Process Documents"
    3. 📊 **View** document structure and statistics in the sidebar
    4. 💬 **Ask** questions about your research in the chat below
    5. 📚 **Get** answers with exact page citations!
    
    **Example Questions:**
    - "What is the main finding of this research?"
    - "Summarize the methodology used"
    - "What are the key conclusions?"
    """)

# Initialize session state
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'sources' not in st.session_state:
    st.session_state.sources = []
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'sections' not in st.session_state:
    st.session_state.sections = []

# Extract sections from PDF
def extract_sections(chunks, sources):
    """Extract section headings and create structure"""
    sections = []
    section_patterns = [
        r'^(Abstract|Introduction|Methods|Methodology|Results|Discussion|Conclusion|References|Acknowledgements)$',
        r'^[IVX]+\.\s+',
        r'^\d+\.\s+',
        r'^\d+\.\d+\.\s+',
    ]
    
    for i, chunk in enumerate(chunks):
        lines = chunk.split('\n')
        for line in lines[:3]:
            line = line.strip()
            if line and len(line) < 80 and not line.endswith('.'):
                for pattern in section_patterns:
                    if re.match(pattern, line, re.IGNORECASE):
                        sections.append({
                            'title': line[:60],
                            'page': sources[i]['page'],
                            'source': sources[i]['file'],
                        })
                        break
    
    seen = set()
    unique_sections = []
    for s in sections:
        key = f"{s['title']}_{s['page']}"
        if key not in seen:
            seen.add(key)
            unique_sections.append(s)
    
    return unique_sections[:20]

# Generate document statistics
def generate_stats_html(chunks, sources, sections):
    unique_files = set(s['file'] for s in sources)
    total_pages = 0
    for s in sources:
        if str(s['page']).isdigit():
            total_pages = max(total_pages, int(s['page']))
    
    return f"""
    <div class="stat-card">
        <h4>📊 Document Statistics</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr><td style="padding: 5px 0;">📄 Files:</td><td><strong>{len(unique_files)}</strong></td></tr>
            <tr><td style="padding: 5px 0;">📑 Chunks:</td><td><strong>{len(chunks)}</strong></td></tr>
            <tr><td style="padding: 5px 0;">📖 Pages:</td><td><strong>{total_pages}</strong></td></tr>
            <tr><td style="padding: 5px 0;">🔍 Sections Found:</td><td><strong>{len(sections)}</strong></td></tr>
        </table>
    </div>
    """

# Generate interactive tree with expandable branches
def generate_interactive_tree(sections, filename):
    """Create interactive tree with expandable/collapsible branches"""
    if not sections:
        return '<div class="stat-card"><p>No sections detected. Try uploading a structured PDF.</p></div>'
    
    # Group sections by level
    main_sections = []
    current_main = None
    
    for sec in sections[:20]:
        title = sec['title']
        
        # Check if this is a main section
        is_main = False
        if title.startswith(('I.', 'II.', 'III.', 'IV.', 'V.', 'VI.', 'VII.')):
            is_main = True
        elif title in ['Abstract', 'Introduction', 'Methods', 'Methodology', 'Results', 'Discussion', 'Conclusion', 'References']:
            is_main = True
        elif re.match(r'^\d+\.\s+', title):
            is_main = True
        
        if is_main:
            current_main = {'title': title, 'page': sec['page'], 'subsections': []}
            main_sections.append(current_main)
        elif current_main:
            current_main['subsections'].append({'title': title, 'page': sec['page']})
    
    # If no main sections detected, create flat list
    if not main_sections:
        for sec in sections[:15]:
            main_sections.append({'title': sec['title'], 'page': sec['page'], 'subsections': []})
    
    # Generate HTML with collapsible sections
    html = f"""
    <div class="tree-card">
        <h4>📑 Document Structure</h4>
        <p style="color: #666; font-size: 0.85rem;">📄 {filename}</p>
        <div style="margin-top: 0.5rem;">
    """
    
    for main in main_sections[:10]:
        if main['subsections']:
            # Section has subsections - make it collapsible
            html += f"""
            <details>
                <summary>📁 <strong>{main['title']}</strong> <span style="color: #888;">(p.{main['page']})</span></summary>
                <div style="margin-left: 1.5rem;">
            """
            for sub in main['subsections'][:6]:
                html += f"""
                <div style="margin: 0.3rem 0;">
                    📄 {sub['title']} <span style="color: #888;">(p.{sub['page']})</span>
                </div>
                """
            if len(main['subsections']) > 6:
                html += f"<div style='color: #888; font-size: 0.8rem;'>... and {len(main['subsections']) - 6} more</div>"
            html += """
                </div>
            </details>
            """
        else:
            # Simple section
            html += f"""
            <div style="margin: 0.5rem 0;">
                📄 <strong>{main['title']}</strong> <span style="color: #888;">(p.{main['page']})</span>
            </div>
            """
    
    html += """
        </div>
        <p style="color: #888; font-size: 0.8rem; margin-top: 0.5rem;">
            💡 Click on 📁 sections to expand/collapse
        </p>
    </div>
    """
    return html

# Extract key terms from document
def extract_key_terms(chunks, top_n=10):
    """Extract important keywords from the document"""
    all_text = " ".join(chunks[:20])
    
    # Common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                  'this', 'that', 'these', 'those', 'it', 'they', 'we', 'you', 'he', 'she'}
    
    # Split into words and count
    words = all_text.lower().split()
    word_counts = {}
    
    for word in words:
        word = word.strip('.,;:()[]{}"\'-')
        if len(word) > 3 and word not in stop_words and not word.isdigit():
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Get top terms
    top_terms = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return [term for term, count in top_terms]

# Fast keyword search
def fast_search(query, chunks, sources, top_k=3):
    query_words = set(query.lower().split())
    
    scores = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk.lower().split())
        overlap = len(query_words & chunk_words)
        if overlap > 0:
            scores.append((i, overlap))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for i, score in scores[:top_k]:
        results.append({
            'content': chunks[i],
            'source': sources[i]['file'],
            'page': sources[i]['page'],
            'score': score
        })
    
    return results

# Answer function
def answer_with_gemini(question, context):
    prompt = f"""You are a research analyst. Use the following context from research papers to answer the question.
Always cite the source document and page number when possible.

Context from research papers:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the context above
2. Cite sources like [Source: filename, page X]
3. If the answer isn't in the context, say "I couldn't find this information"
4. Be concise and professional

Answer:
"""
    
    try:
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e):
            return "⚠️ Rate limit reached. Please wait 30 seconds and try again."
        return f"Error: {str(e)}"

# Sidebar
with st.sidebar:
    st.markdown("## 📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload research papers (PDF)",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.markdown(f"**Files ready:** {len(uploaded_files)}")
        for file in uploaded_files:
            st.markdown(f"✅ {file.name}")
    
    if st.button("🚀 Process Documents", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                all_chunks = []
                all_sources = []
                
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(file.getvalue())
                        tmp_path = tmp_file.name
                    
                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = text_splitter.split_documents(documents)
                    
                    for chunk in chunks:
                        all_chunks.append(chunk.page_content)
                        all_sources.append({
                            'file': file.name,
                            'page': chunk.metadata.get('page', 'N/A'),
                            'content': chunk.page_content
                        })
                    
                    os.unlink(tmp_path)
                
                st.session_state.chunks = all_chunks
                st.session_state.sources = all_sources
                st.session_state.processed_files = [f.name for f in uploaded_files]
                st.session_state.sections = extract_sections(all_chunks, all_sources)
                
                st.success(f"✅ Processed {len(uploaded_files)} files, {len(all_chunks)} chunks")
        else:
            st.warning("Please upload files first")
    
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.chunks = []
        st.session_state.sources = []
        st.session_state.messages = []
        st.session_state.processed_files = []
        st.session_state.sections = []
        st.success("Cleared all documents!")
    
    # Display document structure after processing
    if st.session_state.chunks and st.session_state.sections:
        # Show statistics
        stats_html = generate_stats_html(
            st.session_state.chunks, 
            st.session_state.sources, 
            st.session_state.sections
        )
        st.markdown(stats_html, unsafe_allow_html=True)
        
        # Show interactive tree
        tree_html = generate_interactive_tree(
            st.session_state.sections, 
            st.session_state.processed_files[0] if st.session_state.processed_files else "Document"
        )
        st.markdown(tree_html, unsafe_allow_html=True)
        
        # Show key terms
        key_terms = extract_key_terms(st.session_state.chunks, top_n=12)
        st.markdown("""
        <div style="background: white; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; border: 1px solid #e0e0e0;">
            <h4>🔑 Key Terms</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
        """, unsafe_allow_html=True)
        
        for term in key_terms:
            st.markdown(f'<span style="background: #e9ecef; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem;">{term}</span>', unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("⚡ **Fast Mode** | Powered by Google Gemini")

# Main chat interface
st.markdown("## 💬 Ask Questions")

if st.session_state.chunks:
    st.success(f"✅ Ready! {len(st.session_state.processed_files)} documents loaded with {len(st.session_state.chunks)} chunks")
else:
    st.info("📌 Upload documents and click 'Process Documents' to start")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

# Chat input
if prompt := st.chat_input("Ask about your research papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if st.session_state.chunks:
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching and analyzing..."):
                results = fast_search(prompt, st.session_state.chunks, st.session_state.sources, top_k=3)
                
                if results:
                    context = ""
                    sources = []
                    for i, result in enumerate(results, 1):
                        context += f"[Source {i}: {result['source']}, Page {result['page']}]\n{result['content']}\n\n"
                        sources.append(f"{result['source']} (page {result['page']})")
                    
                    answer = answer_with_gemini(prompt, context)
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📚 Sources"):
                            for source in list(set(sources)):
                                st.markdown(f"- {source}")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": list(set(sources))
                    })
                else:
                    st.warning("No relevant information found.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I couldn't find relevant information in the uploaded documents."
                    })
    else:
        with st.chat_message("assistant"):
            st.warning("Please upload and process documents first!")

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Upload research papers, then ask questions. The AI will search and cite sources!")
st.markdown("✨ **100% Free** - Powered by Google Gemini | Hierarchical Document Structure | Key Terms Extraction")