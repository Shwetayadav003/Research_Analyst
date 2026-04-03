import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import warnings
import sqlite3
import pandas as pd
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("⚠️ Please add your Gemini API key to the .env file")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# Database setup
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('research_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        question TEXT,
        answer TEXT,
        sources TEXT,
        bookmarked INTEGER DEFAULT 0
    )''')
    conn.commit()
    conn.close()

def save_query(question, answer, sources):
    """Save a query to database"""
    conn = sqlite3.connect('research_history.db')
    c = conn.cursor()
    c.execute('''INSERT INTO queries (timestamp, question, answer, sources, bookmarked)
                 VALUES (?, ?, ?, ?, ?)''',
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
               question, answer, str(sources), 0))
    conn.commit()
    conn.close()

def get_query_history(limit=20):
    """Get recent queries"""
    conn = sqlite3.connect('research_history.db')
    c = conn.cursor()
    c.execute('''SELECT id, timestamp, question, answer, sources, bookmarked 
                 FROM queries ORDER BY timestamp DESC LIMIT ?''', (limit,))
    results = c.fetchall()
    conn.close()
    return results

def get_bookmarked_queries():
    """Get all bookmarked queries"""
    conn = sqlite3.connect('research_history.db')
    c = conn.cursor()
    c.execute('''SELECT id, timestamp, question, answer, sources 
                 FROM queries WHERE bookmarked = 1 ORDER BY timestamp DESC''')
    results = c.fetchall()
    conn.close()
    return results

def toggle_bookmark(query_id):
    """Bookmark/unbookmark a query"""
    conn = sqlite3.connect('research_history.db')
    c = conn.cursor()
    c.execute('UPDATE queries SET bookmarked = NOT bookmarked WHERE id = ?', (query_id,))
    conn.commit()
    conn.close()

def delete_query(query_id):
    """Delete a query from history"""
    conn = sqlite3.connect('research_history.db')
    c = conn.cursor()
    c.execute('DELETE FROM queries WHERE id = ?', (query_id,))
    conn.commit()
    conn.close()

def clear_all_history():
    """Clear all query history"""
    conn = sqlite3.connect('research_history.db')
    c = conn.cursor()
    c.execute('DELETE FROM queries')
    conn.commit()
    conn.close()

def export_to_csv():
    """Export all queries to CSV"""
    conn = sqlite3.connect('research_history.db')
    df = pd.read_sql_query("SELECT id, timestamp, question, answer, sources, bookmarked FROM queries", conn)
    conn.close()
    return df.to_csv(index=False)

# Initialize database on startup
init_db()

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
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #4a5568;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        transition: 0.2s;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stat-box {
        background: #f7fafc;
        border-radius: 8px;
        padding: 0.8rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        border-left: 3px solid #2a5298;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔬 Research Analyst AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Research Assistant | Powered by Google Gemini</div>', unsafe_allow_html=True)

# Instructions
with st.expander("📖 How to Use", expanded=False):
    st.markdown("""
    **Steps:**
    1. 📄 **Upload** research papers (PDF)
    2. ⚙️ **Process** documents
    3. 💬 **Ask** questions
    4. 📚 **Get** answers with page citations
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
if 'quick_question' not in st.session_state:
    st.session_state.quick_question = None
if 'show_history' not in st.session_state:
    st.session_state.show_history = False
if 'show_bookmarks' not in st.session_state:
    st.session_state.show_bookmarks = False

# Extract sections from PDF
def extract_sections(chunks, sources):
    """Extract section headings"""
    sections = []
    section_patterns = [
        r'^(Abstract|Introduction|Methods|Methodology|Results|Discussion|Conclusion|References|Acknowledgements|Summary)$',
        r'^[IVX]+\.\s+',
        r'^\d+\.\s+',
        r'^\d+\.\d+\.\s+',
    ]
    
    for i, chunk in enumerate(chunks):
        lines = chunk.split('\n')
        for line in lines[:3]:
            line = line.strip()
            if line and 3 < len(line) < 80 and not line.endswith('.'):
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
    
    return unique_sections[:25]

# Generate tree diagram
def generate_tree_diagram(chunks, sources):
    """Generate a tree diagram of the document structure"""
    sections = extract_sections(chunks, sources)
    
    if not sections:
        return "No clear document structure detected."
    
    main_sections = []
    current_main = None
    
    for sec in sections[:25]:
        title = sec['title']
        page = sec['page']
        
        is_main = False
        if title.startswith(('I.', 'II.', 'III.', 'IV.', 'V.', 'VI.', 'VII.')):
            is_main = True
        elif title.startswith(('1.', '2.', '3.', '4.', '5.')):
            is_main = True
        elif title in ['Abstract', 'Introduction', 'Methods', 'Methodology', 'Results', 'Discussion', 'Conclusion', 'References', 'Summary']:
            is_main = True
        
        if is_main:
            current_main = {'title': title, 'page': page, 'subsections': []}
            main_sections.append(current_main)
        elif current_main:
            if title != current_main['title']:
                current_main['subsections'].append({'title': title, 'page': page})
    
    if not main_sections:
        for sec in sections[:15]:
            main_sections.append({'title': sec['title'], 'page': sec['page'], 'subsections': []})
    
    tree_lines = []
    tree_lines.append("")
    tree_lines.append("```")
    tree_lines.append("DOCUMENT STRUCTURE")
    tree_lines.append("=" * 40)
    tree_lines.append("")
    
    filename = sources[0]['file'] if sources else "Document"
    tree_lines.append(f"📄 {filename}")
    
    if main_sections:
        tree_lines.append("│")
        
        for i, main in enumerate(main_sections):
            prefix = "├── " if i < len(main_sections) - 1 else "└── "
            tree_lines.append(f"{prefix}📁 {main['title']} (p.{main['page']})")
            
            if main['subsections']:
                for j, sub in enumerate(main['subsections'][:8]):
                    sub_prefix = "│   " if i < len(main_sections) - 1 else "    "
                    sub_prefix += "├── " if j < len(main['subsections']) - 1 else "└── "
                    tree_lines.append(f"{sub_prefix}📄 {sub['title']} (p.{sub['page']})")
                
                if len(main['subsections']) > 8:
                    tree_lines.append(f"{'│   ' if i < len(main_sections) - 1 else '    '}└── ... and {len(main['subsections']) - 8} more")
            
            if i < len(main_sections) - 1:
                tree_lines.append("│")
    else:
        for i, sec in enumerate(sections[:15]):
            prefix = "├── " if i < len(sections) - 1 else "└── "
            tree_lines.append(f"{prefix}📄 {sec['title']} (p.{sec['page']})")
    
    tree_lines.append("```")
    tree_lines.append("")
    tree_lines.append(f"**Summary:** {len(sections)} sections found")
    
    return "\n".join(tree_lines)

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

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the context
2. Cite sources like [Source: filename, page X]
3. If not found, say "Information not found in documents"

Answer:
"""
    
    try:
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e):
            return "Rate limit reached. Please wait 30 seconds."
        return f"Error: {str(e)}"

# Sidebar
with st.sidebar:
    st.markdown("## Document Upload")
    uploaded_files = st.file_uploader(
        "Upload research papers (PDF)",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.markdown(f"**Files ready:** {len(uploaded_files)}")
        for file in uploaded_files:
            st.markdown(f"✅ {file.name}")
    
    if st.button("Process Documents", type="primary", use_container_width=True):
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_chunks = []
            all_sources = []
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}... ({idx + 1}/{len(uploaded_files)})")
                progress_bar.progress(idx / len(uploaded_files))
                
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
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("Building search index...")
            
            st.session_state.chunks = all_chunks
            st.session_state.sources = all_sources
            st.session_state.processed_files = [f.name for f in uploaded_files]
            st.session_state.sections = extract_sections(all_chunks, all_sources)
            
            progress_bar.empty()
            status_text.empty()
            
            # Document stats
            total_size = sum(file.size for file in uploaded_files) / 1024
            total_pages = 0
            for source in all_sources:
                if str(source['page']).isdigit():
                    total_pages = max(total_pages, int(source['page']))
            
            st.markdown(f"""
            <div class="stat-box">
                <small>📊 <strong>Document Stats:</strong> {len(uploaded_files)} file(s) | 
                📄 {total_pages} pages | 
                💾 {total_size:.1f} KB | 
                📑 {len(all_chunks)} chunks</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.success(f"Processed {len(uploaded_files)} files, {len(all_chunks)} chunks")
            st.toast("Ready for questions", icon="✅")
        else:
            st.warning("Please upload files first")
    
    st.markdown("---")
    st.markdown("### Quick Questions")
    
    if st.button("What is this paper about?", use_container_width=True):
        st.session_state.quick_question = "What is this research paper about?"
    
    if st.button("What methods were used?", use_container_width=True):
        st.session_state.quick_question = "What methods were used in this study?"
    
    if st.button("What are the main findings?", use_container_width=True):
        st.session_state.quick_question = "What are the main findings of this research?"
    
    if st.button("Show tree diagram", use_container_width=True):
        st.session_state.quick_question = "Convert this PDF into a tree diagram"
    
    st.markdown("---")
    st.markdown("### Query History")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("All Queries", use_container_width=True):
            st.session_state.show_history = not st.session_state.show_history
            st.session_state.show_bookmarks = False
    with col2:
        if st.button("Bookmarks", use_container_width=True):
            st.session_state.show_bookmarks = not st.session_state.show_bookmarks
            st.session_state.show_history = False
    
    if st.session_state.show_history:
        history = get_query_history(limit=15)
        if history:
            for q in history[:10]:
                q_id, timestamp, question, answer, sources, bookmarked = q
                with st.expander(f"📝 {question[:40]}..."):
                    st.caption(f"🕐 {timestamp}")
                    st.write(f"**Answer:** {answer[:100]}...")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        star_label = "⭐ Bookmarked" if bookmarked else "☆ Bookmark"
                        if st.button(star_label, key=f"bookmark_{q_id}"):
                            toggle_bookmark(q_id)
                            st.rerun()
                    with col_b:
                        if st.button("Delete", key=f"delete_{q_id}"):
                            delete_query(q_id)
                            st.rerun()
                    with col_c:
                        if st.button("Re-ask", key=f"reask_{q_id}"):
                            st.session_state.quick_question = question
                            st.rerun()
        else:
            st.info("No queries yet")
    
    if st.session_state.show_bookmarks:
        bookmarks = get_bookmarked_queries()
        if bookmarks:
            for q in bookmarks[:10]:
                q_id, timestamp, question, answer, sources = q
                with st.expander(f"⭐ {question[:40]}..."):
                    st.caption(f"🕐 {timestamp}")
                    st.write(f"**Answer:** {answer[:100]}...")
                    if st.button(f"Re-ask", key=f"reask_bookmark_{q_id}"):
                        st.session_state.quick_question = question
                        st.rerun()
        else:
            st.info("No bookmarked queries")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export CSV", use_container_width=True):
            csv_data = export_to_csv()
            st.download_button(
                label="Download",
                data=csv_data,
                file_name=f"research_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_csv"
            )
    with col2:
        if st.button("Clear History", use_container_width=True):
            clear_all_history()
            st.success("History cleared")
            st.rerun()
    
    if st.button("Clear Documents", use_container_width=True):
        st.session_state.chunks = []
        st.session_state.sources = []
        st.session_state.messages = []
        st.session_state.processed_files = []
        st.session_state.sections = []
        st.session_state.quick_question = None
        st.success("Documents cleared")
    
    st.markdown("---")
    st.markdown("⚡ Google Gemini | Fast Mode")

# Main chat interface
st.markdown("## Ask Questions")

if st.session_state.chunks:
    st.success(f"Ready: {len(st.session_state.processed_files)} documents, {len(st.session_state.chunks)} chunks")
else:
    st.info("Upload documents and click Process Documents to start")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

# Chat input
if st.session_state.quick_question:
    prompt = st.session_state.quick_question
    st.session_state.quick_question = None
else:
    prompt = st.chat_input("Ask about your research papers...")

if prompt:
    tree_keywords = ['tree diagram', 'tree structure', 'document structure', 'show structure', 'convert to tree', 'make a tree']
    
    if any(keyword in prompt.lower() for keyword in tree_keywords):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if st.session_state.chunks:
                with st.spinner("Generating tree diagram..."):
                    tree_diagram = generate_tree_diagram(st.session_state.chunks, st.session_state.sources)
                    st.markdown(tree_diagram)
                    st.session_state.messages.append({"role": "assistant", "content": tree_diagram})
                    save_query(prompt, tree_diagram, "Tree diagram generated")
            else:
                response = "Please upload and process a document first"
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if st.session_state.chunks:
            with st.chat_message("assistant"):
                with st.spinner("Searching and analyzing..."):
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
                            with st.expander("Sources"):
                                for source in list(set(sources)):
                                    st.markdown(f"- {source}")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": list(set(sources))
                        })
                        save_query(prompt, answer, str(list(set(sources))))
                    else:
                        st.warning("No relevant information found")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "No relevant information found in the documents"
                        })
        else:
            with st.chat_message("assistant"):
                st.warning("Please upload and process documents first")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Please upload and process documents first"
                })

# Footer
st.markdown("---")
st.markdown("✨ Google Gemini | All queries saved to database")