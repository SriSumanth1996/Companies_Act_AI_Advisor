import streamlit as st
import os
import re
import json
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.callbacks.base import BaseCallbackHandler
import logging
import regex as re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legal_advisor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CompaniesActParser class for hierarchical parsing
class CompaniesActParser:
    def __init__(self, pdf_path: str = "data/Companies Act 2013.pdf"):
        self.pdf_path = pdf_path
        self.pages: List[str] = []
        self._load_pdf()

    def _load_pdf(self):
        """Load the PDF and extract pages individually using multiple strategies."""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        try:
            loader = PDFMinerLoader(self.pdf_path)
            documents = loader.load()
            if len(documents) == 1 and documents[0].page_content:
                full_text = documents[0].page_content
                page_patterns = [r'\n\s*\d+\s*\n', r'\f', r'Page \d+', r'\n\s*\n\s*\d+\s*\n\s*\n']
                for pattern in page_patterns:
                    potential_pages = re.split(pattern, full_text)
                    if len(potential_pages) > 1:
                        self.pages = [self._clean_text(page) for page in potential_pages if page.strip()]
                        logger.info(f"Split into {len(self.pages)} pages using pattern matching")
                        return
                if len(full_text) > 10000:
                    estimated_page_length = len(full_text) // max(1, len(full_text) // 3000)
                    for i in range(0, len(full_text), estimated_page_length):
                        page_text = full_text[i:i + estimated_page_length]
                        if page_text.strip():
                            self.pages.append(self._clean_text(page_text))
                    logger.info(f"Split into {len(self.pages)} pages using length estimation")
                else:
                    self.pages = [self._clean_text(full_text)]
            else:
                for doc in documents:
                    text = self._clean_text(doc.page_content)
                    self.pages.append(text)
            logger.info("Loaded using PDFMinerLoader")
        except Exception as e:
            logger.error(f"PDFMinerLoader failed: {str(e)}")
            raise Exception("Failed to load PDF with any available method.")

    def _remove_footnotes_and_page_numbers(self, text: str) -> str:
        """Remove footnotes and page numbers from text."""
        if not text:
            return ""
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line or re.match(r'^\d+$', line):
                continue
            footnote_pattern = r'^\d+\.\s+.*?\(w\.e\.f\.\s+\d{1,2}-\d{1,2}-\d{4}\)\.'
            if re.match(footnote_pattern, line):
                continue
            cleaned_lines.append(line)
        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = re.sub(r'\d+\.\s+.*?\(w\.e\.f\.\s*\d{1,2}\s*-\s*\d{1,2}\s*-\s*\d{4}\)\.', '', cleaned_text)
        return cleaned_text

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        text = self._remove_footnotes_and_page_numbers(text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'-\n(\w)', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        return text.strip()

    def find_all_sections(self) -> List[tuple]:
        """Find all sections in the document."""
        section_pattern = r'(\d+[A-Z]*)\.\s+([^\n—]+?)\.\s*—'
        all_sections = []
        for page_num, page_text in enumerate(self.pages, 1):
            matches = re.finditer(section_pattern, page_text)
            for match in matches:
                section_number = match.group(1)
                section_title = match.group(2).strip()
                start_pos = match.start()
                all_sections.append((section_number, page_num, start_pos, section_title))
        return all_sections

    def extract_section(self, section_number: str) -> Optional[str]:
        """Extract a complete section by its number."""
        all_sections = self.find_all_sections()
        target_section = None
        target_index = -1
        for i, (sec_num, page_num, start_pos, title) in enumerate(all_sections):
            if sec_num == section_number:
                target_section = (sec_num, page_num, start_pos, title)
                target_index = i
                break
        if target_section is None:
            logger.warning(f"Section {section_number} not found")
            return None
        next_section = all_sections[target_index + 1] if target_index + 1 < len(all_sections) else None
        target_page = target_section[1]
        target_start_pos = target_section[2]
        if next_section is None:
            if target_page == len(self.pages):
                page_text = self.get_page_text(target_page)
                return page_text[target_start_pos:].strip()
            else:
                result = [self.get_page_text(target_page)[target_start_pos:].strip()]
                for page_num in range(target_page + 1, len(self.pages) + 1):
                    page_text = self.get_page_text(page_num)
                    if page_text:
                        result.append(page_text)
                return "\n\n".join(result)
        else:
            next_page = next_section[1]
            next_start_pos = next_section[2]
            if target_page == next_page:
                page_text = self.get_page_text(target_page)
                raw_section_text = page_text[target_start_pos:next_start_pos]
                return self._trim_to_last_full_stop(raw_section_text)
            else:
                result = [self.get_page_text(target_page)[target_start_pos:].strip()]
                for page_num in range(target_page + 1, next_page):
                    page_text = self.get_page_text(page_num)
                    if page_text:
                        result.append(page_text)
                last_page_text = self.get_page_text(next_page)
                if last_page_text:
                    result.append(last_page_text[:next_start_pos].strip())
                raw_section_text = "\n\n".join(result)
                return self._trim_to_last_full_stop(raw_section_text)

    def _trim_to_last_full_stop(self, text: str) -> str:
        """Trim text to end at the last full stop."""
        if not text:
            return text
        last_full_stop = text.rfind('.')
        if last_full_stop == -1:
            return text.strip()
        text_up_to_last = text[:last_full_stop]
        second_last_full_stop = text_up_to_last.rfind('.')
        remaining_text = text[last_full_stop + 1:].strip()
        if len(remaining_text) <= 50:
            if (remaining_text.upper().startswith(('PART ', 'CHAPTER ')) or
                    re.match(r'^\s*\d+[A-Z]*\.', remaining_text)):
                return text[:last_full_stop + 1].strip()
        if second_last_full_stop != -1:
            between_stops = text[second_last_full_stop + 1:last_full_stop + 1].strip()
            after_last_stop = text[last_full_stop + 1:].strip()
            if (after_last_stop.upper().startswith(('PART ', 'CHAPTER ')) or
                    re.match(r'^\s*\d+[A-Z]*\.', after_last_stop)):
                return text[:last_full_stop + 1].strip()
            if len(between_stops) <= 30:
                return text[:second_last_full_stop + 1].strip()
        return text[:last_full_stop + 1].strip()

    def get_page_text(self, page_number: int) -> Optional[str]:
        """Get raw text of a specific page (1-indexed)."""
        if 1 <= page_number <= len(self.pages):
            return self.pages[page_number - 1]
        return None

    def _normalize_section_text_for_parsing(self, text: str) -> str:
        """
        Normalize section text by removing page breaks, footnotes, and other artifacts
        that can disrupt hierarchy parsing, while preserving the logical structure.
        """
        if not text:
            return ""

        # Remove footnote references and content that appear between logical elements
        # These patterns typically appear between subsections and their explanations
        footnote_patterns = [
            # Remove footnote numbers with insertion references
            r'\d+\.\s*Ins\.\s*by\s*Act\s*\d+\s*of\s*\d+[^.]*\.',
            # Remove standalone page numbers that might appear between content
            r'\n\s*\d{1,3}\s*\n',
            # Remove "next page" markers and similar artifacts
            r'[-\s]*next\s+page\s*[->\s]*',
            # Remove excessive whitespace that might separate related content
            r'\n\s*\n\s*\n+',  # Replace multiple newlines with double newline
        ]

        normalized_text = text
        for pattern in footnote_patterns:
            normalized_text = re.sub(pattern, ' ', normalized_text, flags=re.IGNORECASE)

        # Normalize whitespace while preserving paragraph structure
        normalized_text = re.sub(r'\s+', ' ', normalized_text)
        normalized_text = re.sub(r'\s*\.\s*', '. ', normalized_text)  # Normalize periods

        # Ensure proper spacing around key structural elements
        # Fix cases where explanations might be separated from their subsections
        normalized_text = re.sub(r'(\(\d+\))\s*([A-Z])', r'\1 \2', normalized_text)  # Fix subsection spacing
        normalized_text = re.sub(r'([.:])\s*(Explanation)', r'\1 \2', normalized_text)  # Fix explanation spacing
        normalized_text = re.sub(r'([.:])\s*(Provided)', r'\1 \2', normalized_text)  # Fix proviso spacing

        return normalized_text.strip()

    def _parse_subsection(self, content: str, subsection_num: str) -> Dict:
        """Parse a subsection and its nested elements with enhanced cross-page content handling."""
        subsection = {
            'number': subsection_num,
            'content': '',
            'clauses': {},
            'provisos': [],
            'explanations': {}
        }

        # Further normalize content for better parsing
        content = self._normalize_section_text_for_parsing(content)

        # Split content by major markers, but be more flexible about spacing and formatting
        # First, extract main content before any clauses or provisos
        main_content_match = re.match(r'^(.*?)(?=\([a-z]\)|Provided|Explanation)', content, re.DOTALL | re.IGNORECASE)
        if main_content_match:
            subsection['content'] = main_content_match.group(1).strip()
        else:
            # If no clauses/provisos found, check if entire content is main content
            if not re.search(r'\([a-z]\)|Provided|Explanation', content, re.IGNORECASE):
                subsection['content'] = content.strip()

        # Extract clauses (a), (b), (c), etc. - more flexible pattern
        clause_pattern = r'\(([a-z])\)\s+(.*?)(?=\([a-z]\)|Provided|Explanation|$)'
        clause_matches = re.finditer(clause_pattern, content, re.DOTALL | re.IGNORECASE)

        for match in clause_matches:
            clause_letter = match.group(1)
            clause_content = match.group(2).strip()
            # Remove trailing semicolon or "and" if present
            clause_content = re.sub(r';\s*(and\s*)?$', '', clause_content, flags=re.IGNORECASE)
            subsection['clauses'][clause_letter] = clause_content

        # Extract provisos with more flexible pattern
        proviso_pattern = r'Provided\s+(?:that|further\s+that|also\s+that)\s*(.*?)(?=Provided|Explanation|$)'
        proviso_matches = re.finditer(proviso_pattern, content, re.DOTALL | re.IGNORECASE)

        for match in proviso_matches:
            proviso_content = match.group(1).strip()
            # Clean up the proviso content
            proviso_content = re.sub(r':\s*$', '', proviso_content)  # Remove trailing colon
            if proviso_content:
                subsection['provisos'].append(proviso_content)

        # Extract explanations with enhanced pattern to catch cross-page content
        # FIXED: Use a single comprehensive pattern and avoid duplicates
        explanation_pattern = r'Explanation\s*[.:]?\s*[—-]?\s*(.*?)(?=Explanation|$)'
        explanation_matches = list(re.finditer(explanation_pattern, content, re.DOTALL | re.IGNORECASE))

        # Deduplicate explanations by content
        seen_explanations = set()
        explanation_count = 1

        for match in explanation_matches:
            explanation_content = match.group(1).strip()

            if not explanation_content or len(explanation_content) < 10:  # Skip empty or very short matches
                continue

            # Create a normalized version for duplicate detection
            normalized_explanation = re.sub(r'\s+', ' ', explanation_content.lower()).strip()

            if normalized_explanation in seen_explanations:
                continue  # Skip duplicate

            seen_explanations.add(normalized_explanation)

            # Try to extract individual explanation items (a), (b), etc.
            item_pattern = r'\(([a-z])\)\s+the\s+expression\s+"([^"]+)"\s+means\s+(.*?)(?=\([a-z]\)|$)'
            item_matches = re.finditer(item_pattern, explanation_content, re.DOTALL | re.IGNORECASE)

            explanation_items = {}
            for item_match in item_matches:
                item_letter = item_match.group(1)
                term = item_match.group(2)
                definition = item_match.group(3).strip()
                # Clean up definition
                definition = re.sub(r';\s*$', '', definition)
                explanation_items[item_letter] = {'term': term, 'definition': definition}

            if explanation_items:
                subsection['explanations'][f'explanation_{explanation_count}'] = explanation_items
            else:
                # If no structured items found, store the entire explanation
                # Clean up any remaining artifacts
                cleaned_explanation = re.sub(r'\s+', ' ', explanation_content).strip()
                if cleaned_explanation and len(cleaned_explanation) > 10:  # Only store substantial content
                    subsection['explanations'][f'explanation_{explanation_count}'] = cleaned_explanation

            explanation_count += 1

        return subsection

    def _parse_section_hierarchy(self, section_text: str, section_number: str) -> Dict:
        """Parse the hierarchical structure of a section."""
        # Normalize the text to handle cross-page artifacts
        normalized_text = self._normalize_section_text_for_parsing(section_text)

        hierarchy = {
            'section_number': section_number,
            'title': '',
            'content': '',
            'subsections': {}
        }

        # Extract section title (between section number and first subsection or content)
        title_match = re.match(
            rf'{re.escape(section_number)}\.\s+([^.]+)\.\s*—\s*(.*?)(?=\(\d+\)|$)',
            normalized_text,
            re.DOTALL
        )
        if title_match:
            hierarchy['title'] = title_match.group(1).strip()
            initial_content = title_match.group(2).strip()
            if initial_content and not re.match(r'^\(\d+\)', initial_content):
                hierarchy['content'] = initial_content

        # Find all subsections marked with (1), (2), etc.
        subsection_pattern = r'\((\d+)\)\s+(.*?)(?=\(\d+\)|$)'
        subsection_matches = re.finditer(subsection_pattern, normalized_text, re.DOTALL)

        for match in subsection_matches:
            subsection_num = match.group(1)
            subsection_content = match.group(2).strip()
            hierarchy['subsections'][subsection_num] = self._parse_subsection(subsection_content, subsection_num)

        return hierarchy

    def extract_section_with_hierarchy(self, section_number: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Extract a section and return both raw text and hierarchical structure."""
        raw_text = self.extract_section(section_number)
        if not raw_text:
            return None, None
        hierarchy = self._parse_section_hierarchy(raw_text, section_number)
        return raw_text, hierarchy

@dataclass
class QueryAnalysis:
    """Structure for query analysis results"""
    query_type: str
    confidence: float
    relevant_sections: List[str]
    complexity_score: int
    keywords: List[str]

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for Streamlit streaming"""
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

class EnhancedLegalAdvisor:
    """Enhanced Legal Advisor with hierarchical parsing and robust error handling"""
    def __init__(self):
        self.groq_api_key = self._load_api_key()
        self.parser = CompaniesActParser()
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.rag_chain = None
        self.retriever = None
        self.query_cache = {}
        self.setup_complete = False

    def _load_api_key(self) -> str:
        """Securely load API key with multiple fallbacks"""
        try:
            if "groq_api_key" in st.secrets:
                return st.secrets["groq_api_key"]
            elif "GROQ_API_KEY" in os.environ:
                return os.environ["GROQ_API_KEY"]
            else:
                st.error("❌ GROQ API key not found. Please set it in secrets.toml or environment variables.")
                st.stop()
        except Exception as e:
            st.error(f"❌ Error loading API key: {str(e)}")
            st.stop()

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze user query to determine type, complexity, and relevant sections"""
        query_lower = query.lower()
        definition_patterns = ['what is', 'define', 'meaning of', 'explain', 'elaborate', 'describe']
        procedure_patterns = ['how to', 'procedure', 'process', 'steps', 'method', 'way to', 'approach']
        compliance_patterns = ['penalty', 'fine', 'non-compliance', 'violation', 'punishment', 'imprisonment',
                              'offence', 'default', 'breach', 'prosecution', 'liable', 'liability']
        comparison_patterns = ['difference', 'versus', 'vs', 'compare', 'distinguish', 'relative to', 'contrast']
        role_patterns = ["duties", "responsibilities", "powers", "liabilities", "obligations", "authority", "role of",
                         "function of"]
        filing_patterns = ['registrar of companies', "roc", "form mgt", "form aoc", "filing deadline", "annual return",
                           "statutory filing", "reporting", "submission", "form filing"]
        query_type = "general"
        confidence = 0.5
        if any(pattern in query_lower for pattern in definition_patterns):
            query_type = "definition"
            confidence = 0.8
        elif any(pattern in query_lower for pattern in procedure_patterns):
            query_type = "procedure"
            confidence = 0.8
        elif any(pattern in query_lower for pattern in compliance_patterns):
            query_type = "compliance"
            confidence = 0.9
        elif any(pattern in query_lower for pattern in comparison_patterns):
            query_type = "comparison"
            confidence = 0.7
        elif any(pattern in query_lower for pattern in role_patterns):
            query_type = "role"
            confidence = 0.8
        elif any(pattern in query_lower for pattern in filing_patterns):
            query_type = "filing"
            confidence = 0.8
        keywords = re.findall(r'\b[a-zA-Z]{3,}\b', query_lower)
        keywords = [word for word in keywords if word not in ['what', 'how', 'the', 'and', 'are', 'for']]
        complexity_score = min(len(keywords) + len(query.split()), 10)
        relevant_sections = []
        section_pattern = r'\bsection\s+(\d+[A-Z]*)\b'
        section_matches = re.findall(section_pattern, query_lower, re.IGNORECASE)
        if section_matches:
            relevant_sections = section_matches
        else:
            section_keywords = {
                'related party transactions': ['188'],
                'director duties': ['166', '149'],
                'annual return': ['92'],
                'incorporation': ['7'],
                'private company': ['2(68)'],
                'public company': ['2(71)'],
                'auditor appointment': ['139'],
                'share capital': ['61'],
                'company name change': ['13'],
                'penalties': ['188', '92', '134'],  # Added for penalty-related queries
                'violation': ['188', '92', '134'],
                'non-compliance': ['188', '92', '134']
            }
            for key_phrase, sections in section_keywords.items():
                if key_phrase in query_lower:
                    relevant_sections.extend(sections)
                    confidence = max(confidence, 0.9)  # Boost confidence for strong matches
                    break
            if not relevant_sections and any(k in query_lower for k in ['penalty', 'violation', 'non-compliance']):
                relevant_sections = ['188']  # Default to 188 for penalty-related queries about transactions
        return QueryAnalysis(
            query_type=query_type,
            confidence=confidence,
            relevant_sections=list(set(relevant_sections)),  # Remove duplicates
            complexity_score=complexity_score,
            keywords=keywords[:5]
        )

    def format_hierarchy(self, hierarchy: Dict) -> str:
        """Format the hierarchical structure into a readable string."""
        if not hierarchy:
            return ""
        result = [f"Section {hierarchy['section_number']}: {hierarchy['title']}"]
        if hierarchy['content']:
            result.append(f"Content: {hierarchy['content']}")
        for subsection_num, subsection in sorted(hierarchy['subsections'].items(), key=lambda x: int(x[0])):
            result.append(f"  Subsection ({subsection_num}) {subsection['content']}")
            if subsection['clauses']:
                result.append("    Clauses:")
                for clause_letter, clause_content in sorted(subsection['clauses'].items()):
                    result.append(f"      ({clause_letter}) {clause_content}")
            if subsection['provisos']:
                result.append("    Provisos:")
                for i, proviso in enumerate(subsection['provisos'], 1):
                    result.append(f"      Proviso {i}: {proviso}")
            if subsection['explanations']:
                result.append("    Explanations:")
                for exp_key, exp_content in sorted(subsection['explanations'].items()):
                    if isinstance(exp_content, dict):
                        for item_key, item_data in sorted(exp_content.items()):
                            result.append(f"      ({item_key}) {item_data['term']}: {item_data['definition']}")
                    else:
                        result.append(f"      {exp_key}: {exp_content}")
        return "\n".join(result)

    def extract_enhanced_layers(self, documents: List[Document]) -> List[Document]:
        """Extract enhanced document layers with hierarchy."""
        layered_docs = []
        try:
            for doc_idx, doc in enumerate(documents):
                text = doc.page_content
                metadata = doc.metadata.copy()
                if not text.strip():
                    continue
                layered_docs.append(Document(
                    page_content=text,
                    metadata={**metadata, "layer": "document", "doc_index": doc_idx}
                ))
                all_sections = self.parser.find_all_sections()
                for section_number, page_num, start_pos, section_title in all_sections:
                    if page_num == metadata.get('page', 0):
                        raw_text, hierarchy = self.parser.extract_section_with_hierarchy(section_number)
                        if raw_text and hierarchy:
                            layered_docs.append(Document(
                                page_content=self.format_hierarchy(hierarchy),
                                metadata={
                                    **metadata,
                                    "layer": "section",
                                    "section_number": section_number,
                                    "section_title": section_title,
                                    "doc_index": doc_idx
                                }
                            ))
            logger.info(f"Created {len(layered_docs)} layered documents with hierarchy")
            return layered_docs
        except Exception as e:
            logger.error(f"Error in document layering: {str(e)}")
            return documents

    @st.cache_resource
    def setup_system(_self):
        """Setup the complete RAG system with robust error handling"""
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("📄 Loading Companies Act document...")
            progress_bar.progress(10)
            pdf_path = "data/Companies Act 2013.pdf"
            if not os.path.exists(pdf_path):
                alternative_paths = [
                    "./data/Companies Act 2013.pdf",
                    "../data/Companies Act 2013.pdf",
                    "Companies Act 2013.pdf"
                ]
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        pdf_path = alt_path
                        break
                else:
                    raise FileNotFoundError(f"Companies Act PDF not found in expected locations")
            loader = PDFMinerLoader(pdf_path)
            raw_documents = loader.load()
            progress_bar.progress(25)
            if not raw_documents:
                raise ValueError("No content loaded from PDF")
            status_text.text("✂️ Processing document structure...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=200,
                length_function=len,
                separators=["\nCHAPTER", "\n\n", "\n", ". ", " ", ""]
            )
            documents = text_splitter.split_documents(raw_documents)
            progress_bar.progress(40)
            status_text.text("🔍 Extracting legal structure...")
            layered_docs = _self.extract_enhanced_layers(documents)
            progress_bar.progress(55)
            status_text.text("🧠 Setting up AI embeddings...")
            _self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            progress_bar.progress(70)
            status_text.text("💾 Building knowledge base...")
            persist_directory = "vectorstore/faiss_index_enhanced"
            doc_hash = hashlib.md5(str([doc.page_content[:100] for doc in layered_docs]).encode()).hexdigest()
            hash_file = os.path.join(persist_directory, "doc_hash.txt")
            rebuild_needed = True
            if os.path.exists(persist_directory) and os.path.exists(hash_file):
                try:
                    with open(hash_file, 'r') as f:
                        stored_hash = f.read().strip()
                    if stored_hash == doc_hash:
                        _self.vectorstore = FAISS.load_local(
                            persist_directory,
                            _self.embeddings,
                            allow_dangerous_deserialization=True
                        )
                        rebuild_needed = False
                except Exception as e:
                    logger.warning(f"Error loading existing vectorstore: {e}")
            if rebuild_needed:
                _self.vectorstore = FAISS.from_documents(layered_docs, _self.embeddings)
                os.makedirs(persist_directory, exist_ok=True)
                _self.vectorstore.save_local(persist_directory)
                with open(hash_file, 'w') as f:
                    f.write(doc_hash)
            progress_bar.progress(85)
            status_text.text("🔎 Configuring search system...")
            base_retriever = _self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 12, "fetch_k": 40, "lambda_mult": 0.7}
            )
            _self.llm = ChatGroq(
                groq_api_key=_self.groq_api_key,
                model_name="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=500  # Increased for more detailed responses
            )
            llm_filter = LLMChainFilter.from_llm(_self.llm)
            _self.retriever = ContextualCompressionRetriever(
                base_retriever=base_retriever,
                base_compressor=llm_filter
            )
            progress_bar.progress(95)
            status_text.text("⚖️ Finalizing legal advisor...")
            enhanced_prompt = """You are an expert AI Legal Advisor specializing in the Companies Act of India, 2013.
                INSTRUCTIONS:
                1. Analyze user question carefully.
                2. Understand the hierarchy of the relevant sections involved.
                3. Use the provided context to give accurate and detailed answer.
                4. Always cite relevant sections, subsections, and clauses that are related to the aspect involved in the query.
                5. Structure your response logically with clear headings.
                
                Context from Companies Act 2013:
                {context}
                
                User Question: {question}
                
                Comprehensive Legal Answer:"""
            PROMPT = PromptTemplate(
                template=enhanced_prompt,
                input_variables=["context", "question", "query_type", "complexity_score", "keywords", "relevant_sections"]
            )
            def format_enhanced_docs(docs, hierarchies):
                formatted_context = ""
                for i, doc in enumerate(docs, 1):
                    metadata = doc.metadata
                    layer = metadata.get('layer', 'unknown').title()
                    page = metadata.get('page', 'N/A')
                    header = f"\n--- {layer}"
                    if layer == "Section" and 'section_number' in metadata:
                        header += f" {metadata['section_number']}"
                        if 'section_title' in metadata:
                            header += f": {metadata['section_title']}"
                    header += f" (Page {page}) ---\n"
                    content = doc.page_content[:1200]
                    if len(doc.page_content) > 1200:
                        content += "..."
                    formatted_context += f"{header}{content}\n"
                for section_number, hierarchy in hierarchies.items():
                    formatted_context += f"\n--- Section Hierarchy {section_number} ---\n"
                    formatted_context += _self.format_hierarchy(hierarchy) + "\n"
                return formatted_context
            def enhanced_format_context(input_data):
                if isinstance(input_data, str):
                    question = input_data
                    docs = _self.retriever.invoke(question)
                else:
                    docs = input_data.get("retrieved_docs", [])
                    question = input_data.get("question", "")
                analysis = _self.analyze_query(question)
                hierarchies = {}
                for section_number in analysis.relevant_sections:
                    raw_text, hierarchy = _self.parser.extract_section_with_hierarchy(section_number)
                    if hierarchy:
                        hierarchies[section_number] = hierarchy
                return {
                    "context": format_enhanced_docs(docs, hierarchies),
                    "question": question,
                    "query_type": analysis.query_type,
                    "complexity_score": analysis.complexity_score,
                    "keywords": ", ".join(analysis.keywords),
                    "relevant_sections": ", ".join(analysis.relevant_sections)
                }
            _self.rag_chain = (
                RunnableLambda(enhanced_format_context)
                | PROMPT
                | _self.llm
                | StrOutputParser()
            )
            progress_bar.progress(100)
            status_text.text("✅ Legal Advisor ready!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            _self.setup_complete = True
            logger.info("Enhanced Legal Advisor setup completed successfully")
        except Exception as e:
            logger.error(f"Error in system setup: {str(e)}")
            st.error(f"❌ Setup failed: {str(e)}")
            raise

    def get_cached_response(self, query: str) -> Optional[str]:
        """Get cached response for similar queries"""
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
        return self.query_cache.get(query_hash)

    def cache_response(self, query: str, response: str):
        """Cache response for future use"""
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
        self.query_cache[query_hash] = response
        if len(self.query_cache) > 100:
            keys_to_remove = list(self.query_cache.keys())[:20]
            for key in keys_to_remove:
                del self.query_cache[key]

    def generate_response(self, query: str, use_cache: bool = True) -> Tuple[str, List[Document]]:
        """Generate response with hierarchical context"""
        if not self.setup_complete:
            raise RuntimeError("System not properly initialized")
        if use_cache:
            cached_response = self.get_cached_response(query)
            if cached_response:
                st.info("📋 Using cached response for faster delivery")
                return cached_response, []
        try:
            retrieved_docs = self.retriever.invoke(query)
            response = self.rag_chain.invoke(query)
            if use_cache:
                self.cache_response(query, response)
            return response, retrieved_docs
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            error_response = f"""❌ I encountered an error while processing your question.

**Error Details**: {str(e)}

**Suggestions**:
- Try rephrasing your question
- Check if your question is related to the Companies Act 2013
- Contact support if the issue persists"""
            return error_response, []

def main():
    with st.sidebar:
        st.header("⚖️ Legal Advisor Controls")
        advisor = st.cache_resource(get_legal_advisor)()
        if not advisor.setup_complete:
            with st.spinner("Initializing Legal Advisor..."):
                advisor.setup_system()
        if advisor.setup_complete:
            st.success("✅ System Ready")
        st.subheader("🔧 Settings")
        use_cache = st.checkbox("Use Response Cache", value=True)
        show_sources = st.checkbox("Show Source Documents", value=True)
        max_sources = st.slider("Max Sources to Show", 1, 10, 5)
        st.subheader("📊 Session Stats")
        st.metric("Questions Asked", len([m for m in st.session_state.messages if m["role"] == "user"]))
        st.metric("Cached Responses", len(advisor.query_cache))
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
        if st.session_state.messages and st.button("📥 Export Conversation"):
            conversation_data = {
                "conversation_id": st.session_state.conversation_id,
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.messages
            }
            st.download_button(
                "Download JSON",
                json.dumps(conversation_data, indent=2),
                f"legal_conversation_{st.session_state.conversation_id}.json",
                "application/json"
            )
    st.title("⚖️ Indian Companies Act AI Legal Advisor")
    st.markdown("""
    🎯 **Enhanced Legal Assistant** for the Companies Act of India, 2013
    Get comprehensive, accurate answers with proper legal citations and practical guidance.
    """)
    with st.expander("💡 Example Questions (Click to Use)", expanded=False):
        example_questions = [
            "What is the definition of a private company under the Companies Act 2013?",
            "What are the minimum requirements for incorporating a company?",
            "What is the difference between public and private companies?",
            "What are the statutory duties and liabilities of directors?",
            "What is the procedure for changing the name of a company?",
            "What are the penalties for non-filing of annual returns?",
            "How can a company increase its authorized share capital?",
            "What is the procedure for appointment of auditors?",
            "What are the provisions related to related party transactions?",
            "What is the minimum and maximum number of directors in a company?"
        ]
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            col = cols[i % 2]
            if col.button(f"📝 {question[:50]}...", key=f"example_{i}"):
                st.session_state.example_question = question
                st.rerun()
    if "example_question" in st.session_state:
        user_input = st.session_state.example_question
        del st.session_state.example_question
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                response, retrieved_docs = advisor.generate_response(user_input, use_cache)
                message_placeholder.markdown(response)
                if show_sources and retrieved_docs:
                    with st.expander(f"📚 Legal Sources ({len(retrieved_docs)} documents)", expanded=False):
                        for i, doc in enumerate(retrieved_docs[:max_sources]):
                            metadata = doc.metadata
                            layer = metadata.get('layer', 'Unknown').title()
                            page = metadata.get('page', 'N/A')
                            section = metadata.get('section_number', '')
                            section_title = metadata.get('section_title', '')
                            header = f"**📖 Source {i + 1}: {layer}"
                            if section:
                                header += f" {section}"
                                if section_title:
                                    header += f" - {section_title}"
                            header += f" (Page {page})**"
                            st.markdown(header)
                            content_preview = doc.page_content[:600]
                            if len(doc.page_content) > 600:
                                content_preview += "..."
                            st.code(content_preview, language="text")
                            st.markdown("---")
            except Exception as e:
                error_msg = f"❌ Error processing question: {str(e)}"
                message_placeholder.markdown(error_msg)
                response = error_msg
            st.session_state.messages.append({"role": "assistant", "content": response})
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask a question about the Companies Act 2013..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                with st.spinner("🤔 Analyzing your question..."):
                    response, retrieved_docs = advisor.generate_response(prompt, use_cache)
                message_placeholder.markdown(response)
                if show_sources and retrieved_docs:
                    with st.expander(f"📚 Legal Sources ({len(retrieved_docs)} documents)", expanded=False):
                        for i, doc in enumerate(retrieved_docs[:max_sources]):
                            metadata = doc.metadata
                            layer = metadata.get('layer', 'Unknown').title()
                            page = metadata.get('page', 'N/A')
                            section = metadata.get('section_number', '')
                            section_title = metadata.get('section_title', '')
                            header = f"**📖 Source {i + 1}: {layer}"
                            if section:
                                header += f" {section}"
                                if section_title:
                                    header += f" - {section_title}"
                            header += f" (Page {page})**"
                            st.markdown(header)
                            content_preview = doc.page_content[:600]
                            if len(doc.page_content) > 600:
                                content_preview += "..."
                            st.code(content_preview, language="text")
                            st.markdown("---")
            except Exception as e:
                error_msg = f"❌ Error processing your question: {str(e)}"
                message_placeholder.markdown(error_msg)
                response = error_msg
            st.session_state.messages.append({"role": "assistant", "content": response})

@st.cache_resource
def get_legal_advisor():
    return EnhancedLegalAdvisor()

if __name__ == "__main__":
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    main()
