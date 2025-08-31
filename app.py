from flask import Flask, request, render_template, jsonify, session, redirect, url_for, flash
import os
import re
import json
import PyPDF2
import google.generativeai as genai
from datetime import datetime
from werkzeug.utils import secure_filename
import spacy
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pickle
import uuid
import difflib
from collections import Counter
import re

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Configuration
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Initialize Gemini API
GEMINI_API_KEY = "AIzaSyCuYxsYkfHhJ-9OPWVUjsEmuFELdv639f8"
genai.configure(api_key=GEMINI_API_KEY)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy English model loaded successfully")
except OSError:
    print("⚠️  spaCy English model not found. Some features will be limited.")
    print("To install: python -m spacy download en_core_web_sm")
    nlp = None

def get_session_id():
    """Get or create a session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def save_documents_to_file(session_id, documents):
    """Save documents to file instead of session"""
    file_path = os.path.join(DATA_FOLDER, f"{session_id}.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(documents, f)

def load_documents_from_file(session_id):
    """Load documents from file"""
    file_path = os.path.join(DATA_FOLDER, f"{session_id}.pkl")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

def extract_dates_with_gemini(text):
    """Use Gemini API to extract and identify important dates"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Analyze this legal document and extract all important dates. For each date, identify its significance (e.g., agreement date, commencement date, expiry date, etc.).
        
        Text: {text[:4000]}  # Limit text to avoid token limits
        
        Return the response in JSON format like this:
        {{
            "dates": [
                {{"date": "2023-01-15", "significance": "Agreement Date", "context": "brief context"}},
                {{"date": "2023-02-01", "significance": "Commencement Date", "context": "brief context"}}
            ]
        }}
        """
        
        response = model.generate_content(prompt)
        # Try to extract JSON from response
        response_text = response.text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"dates": []}
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return {"dates": []}

def extract_parties_with_gemini(text):
    """Use Gemini API to extract lessor and lessee information"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Analyze this legal document and extract information about all parties involved, specifically identifying:
        1. Lessor (landlord/owner) - name, address, contact details
        2. Lessee (tenant/renter) - name, address, contact details
        
        Text: {text[:4000]}
        
        Return the response in JSON format like this:
        {{
            "lessor": [
                {{"name": "John Doe", "address": "123 Main St", "details": "additional details"}}
            ],
            "lessee": [
                {{"name": "Jane Smith", "address": "456 Oak Ave", "details": "additional details"}}
            ]
        }}
        """
        
        response = model.generate_content(prompt)
        response_text = response.text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"lessor": [], "lessee": []}
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return {"lessor": [], "lessee": []}

def extract_key_terms_with_gemini(text):
    """Use Gemini API to extract key terms and conditions"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Analyze this legal document and extract key terms including:
        - Rent amount and payment terms
        - Security deposit
        - Duration/term of agreement
        - Important clauses and conditions
        - Property details
        
        Text: {text[:4000]}
        
        Return the response in JSON format like this:
        {{
            "rent": {{"amount": "1000", "frequency": "monthly", "details": "rent details"}},
            "deposit": {{"amount": "2000", "details": "security deposit details"}},
            "duration": {{"period": "12 months", "details": "duration details"}},
            "property": {{"address": "property address", "description": "property description"}},
            "key_clauses": ["clause 1", "clause 2"]
        }}
        """
        
        response = model.generate_content(prompt)
        response_text = response.text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {}
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return {}

def extract_clauses_from_text(text):
    """Extract numbered clauses from legal documents"""
    clauses = []
    
    # Multiple patterns for clause identification
    clause_patterns = [
        # Standard numbered clauses: "1.", "2.", etc.
        r'^\s*(\d+)\.\s+(.+?)(?=^\s*\d+\.|$)',
        # Sub-clauses: "1.1", "1.2", etc.
        r'^\s*(\d+\.\d+)\s+(.+?)(?=^\s*\d+\.(?:\d+\s+|[^\d])|$)',
        # Article format: "Article 1", "ARTICLE I", etc.
        r'^\s*(?:ARTICLE|Article)\s+([IVX]+|\d+)[:\.\s]+(.+?)(?=^\s*(?:ARTICLE|Article)\s+|$)',
        # Section format: "Section 1", "SECTION A", etc.
        r'^\s*(?:SECTION|Section)\s+([A-Z]|\d+)[:\.\s]+(.+?)(?=^\s*(?:SECTION|Section)\s+|$)',
        # Clause format: "Clause 1", "CLAUSE A", etc.
        r'^\s*(?:CLAUSE|Clause)\s+([A-Z]|\d+)[:\.\s]+(.+?)(?=^\s*(?:CLAUSE|Clause)\s+|$)',
        # Paragraph format: "(a)", "(1)", etc.
        r'^\s*\(([a-z]|\d+)\)\s+(.+?)(?=^\s*\([a-z]|\d+\)|$)',
        # Roman numerals: "I.", "II.", etc.
        r'^\s*([IVX]+)\.\s+(.+?)(?=^\s*[IVX]+\.|$)',
        # Letter format: "a)", "b)", etc.
        r'^\s*([a-z])\)\s+(.+?)(?=^\s*[a-z]\)|$)'
    ]
    
    lines = text.split('\n')
    current_clause = None
    clause_content = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        found_clause = False
        
        for pattern in clause_patterns:
            match = re.match(pattern, line, re.MULTILINE | re.DOTALL)
            if match:
                # Save previous clause if exists
                if current_clause and clause_content:
                    clauses.append({
                        'number': current_clause,
                        'content': ' '.join(clause_content).strip(),
                        'position': i - len(clause_content),
                        'type': get_clause_type(current_clause)
                    })
                
                # Start new clause
                current_clause = match.group(1)
                clause_content = [match.group(2)]
                found_clause = True
                break
        
        if not found_clause and current_clause:
            # Continue adding to current clause
            clause_content.append(line)
    
    # Add the last clause
    if current_clause and clause_content:
        clauses.append({
            'number': current_clause,
            'content': ' '.join(clause_content).strip(),
            'position': len(lines) - len(clause_content),
            'type': get_clause_type(current_clause)
        })
    
    # Also extract paragraphs that might be clauses without clear numbering
    additional_clauses = extract_implicit_clauses(text)
    
    # Combine and sort clauses
    all_clauses = clauses + additional_clauses
    
    # Remove duplicates and sort by position
    seen_content = set()
    unique_clauses = []
    for clause in all_clauses:
        content_key = clause['content'][:100].lower()  # First 100 chars for comparison
        if content_key not in seen_content and len(clause['content']) > 50:  # Minimum length filter
            seen_content.add(content_key)
            unique_clauses.append(clause)
    
    unique_clauses.sort(key=lambda x: x['position'])
    
    return unique_clauses

def get_clause_type(number):
    """Determine the type of clause based on its numbering"""
    if re.match(r'^\d+$', str(number)):
        return 'numbered'
    elif re.match(r'^\d+\.\d+$', str(number)):
        return 'sub-clause'
    elif re.match(r'^[IVX]+$', str(number)):
        return 'roman'
    elif re.match(r'^[a-z]$', str(number)):
        return 'lettered'
    elif re.match(r'^\([a-z0-9]+\)$', str(number)):
        return 'parenthetical'
    else:
        return 'other'

def extract_implicit_clauses(text):
    """Extract clauses that might not have clear numbering but are important sections"""
    implicit_clauses = []
    
    # Common legal section headers
    section_headers = [
        r'(?:WHEREAS|Whereas)[,\s]+(.+?)(?=\n\s*(?:WHEREAS|Whereas|NOW\s+THEREFORE)|$)',
        r'(?:NOW\s+THEREFORE|Now\s+Therefore)[,\s]+(.+?)(?=\n\s*(?:WHEREAS|IN\s+WITNESS)|$)',
        r'(?:IN\s+WITNESS\s+WHEREOF|In\s+Witness\s+Whereof)[,\s]+(.+?)(?=\n\s*[A-Z]|$)',
        r'(?:TERMS\s+AND\s+CONDITIONS|Terms\s+and\s+Conditions)[:\s]*\n(.+?)(?=\n\s*[A-Z]|$)',
        r'(?:DEFINITIONS|Definitions)[:\s]*\n(.+?)(?=\n\s*[A-Z]|$)',
        r'(?:RECITALS|Recitals)[:\s]*\n(.+?)(?=\n\s*[A-Z]|$)'
    ]
    
    for i, pattern in enumerate(section_headers):
        matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        for match in matches:
            content = match.group(1).strip()
            if len(content) > 50:  # Minimum content length
                implicit_clauses.append({
                    'number': f'Section-{i+1}',
                    'content': content[:1000],  # Limit content length
                    'position': match.start(),
                    'type': 'section'
                })
    
    return implicit_clauses

def extract_numbered_bullet_clauses(text):
    """Extract sentences that start with numbered bullet points and have 10+ words"""
    numbered_clauses = []
    lines = text.split('\n')
    
    # Pattern to match numbered bullet points at start of sentence
    number_patterns = [
        r'^(\d+)\.\s+(.+)',           # "1. text"
        r'^(\d+)\)\s+(.+)',           # "1) text"  
        r'^\((\d+)\)\s+(.+)',         # "(1) text"
        r'^(\d+)\s*[-–]\s+(.+)',      # "1 - text" or "1 – text"
        r'^(\d+)\s*:\s+(.+)',         # "1: text"
    ]
    
    current_clause_number = None
    current_clause_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        found_number = False
        
        # Check if this line starts a new numbered clause
        for pattern in number_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                # Save previous clause if it exists and has enough words
                if current_clause_number and current_clause_content:
                    full_content = ' '.join(current_clause_content).strip()
                    word_count = len(full_content.split())
                    if word_count >= 10:
                        numbered_clauses.append({
                            'number': current_clause_number,
                            'content': full_content,
                            'word_count': word_count
                        })
                
                # Start new clause
                current_clause_number = match.group(1)
                initial_content = match.group(2).strip()
                current_clause_content = [initial_content] if initial_content else []
                found_number = True
                break
        
        # If not a new numbered clause and we have a current clause, add to content
        if not found_number and current_clause_number:
            # Don't add if it looks like a header or new section
            if (not re.match(r'^[A-Z\s]+:?$', line) and 
                not re.match(r'^\s*Page\s+\d+', line, re.IGNORECASE) and
                not re.match(r'^\s*\d+\s*$', line) and
                len(line) > 3 and
                not re.match(r'^\d+[\.\)\:]', line)):  # Don't add if it's another numbered item
                current_clause_content.append(line)
    
    # Add the last clause
    if current_clause_number and current_clause_content:
        full_content = ' '.join(current_clause_content).strip()
        word_count = len(full_content.split())
        if word_count >= 10:
            numbered_clauses.append({
                'number': current_clause_number,
                'content': full_content,
                'word_count': word_count
            })
    
    return numbered_clauses

def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def build_vocabulary_from_documents(session_id):
    """Build a vocabulary from all uploaded documents"""
    docs = load_documents_from_file(session_id)
    vocabulary = Counter()
    
    for doc in docs:
        # Get full text for vocabulary building
        full_text = doc.get('text', '')
        if 'full_text_path' in doc and os.path.exists(doc['full_text_path']):
            try:
                with open(doc['full_text_path'], 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    full_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            except:
                pass
        
        # Extract words and count them
        words = re.findall(r'\b[a-zA-Z]{3,}\b', full_text.lower())
        vocabulary.update(words)
    
    return vocabulary

def suggest_corrections(query, session_id, max_suggestions=3, max_distance=2):
    """Suggest spelling corrections for search query"""
    vocabulary = build_vocabulary_from_documents(session_id)
    
    if not vocabulary:
        return []
    
    query_words = query.lower().split()
    suggestions = []
    
    for word in query_words:
        if len(word) < 3:  # Skip very short words
            continue
            
        word_suggestions = []
        
        # First try difflib for similar words
        close_matches = difflib.get_close_matches(word, vocabulary.keys(), n=max_suggestions, cutoff=0.6)
        
        if close_matches:
            word_suggestions.extend(close_matches)
        else:
            # If no close matches, use Levenshtein distance
            for vocab_word in vocabulary.keys():
                if abs(len(word) - len(vocab_word)) <= max_distance:
                    distance = levenshtein_distance(word, vocab_word)
                    if distance <= max_distance and distance > 0:
                        word_suggestions.append((vocab_word, distance, vocabulary[vocab_word]))
            
            # Sort by distance and frequency
            word_suggestions.sort(key=lambda x: (x[1], -x[2]))
            word_suggestions = [w[0] for w in word_suggestions[:max_suggestions]]
        
        if word_suggestions:
            suggestions.append({
                'original': word,
                'suggestions': word_suggestions[:max_suggestions]
            })
    
    return suggestions

def search_text(text, query, context_length=150):
    """Search for query in text and return matches with context and navigation"""
    if not query.strip():
        return []
    
    # Clean the text first to handle PDF extraction issues
    cleaned_text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    cleaned_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_text)  # Add space between camelCase
    
    # Try multiple search strategies for better matching
    patterns = [
        re.compile(r'\b' + re.escape(query) + r'\b', re.IGNORECASE),  # Exact word boundary
        re.compile(re.escape(query), re.IGNORECASE),  # Substring match
        re.compile(r'\b' + re.escape(query.lower()) + r'\b', re.IGNORECASE),  # Lowercase word boundary
    ]
    
    matches = []
    match_positions = set()  # To avoid duplicate matches
    
    for pattern in patterns:
        for match_num, match in enumerate(pattern.finditer(cleaned_text)):
            # Skip if we already found a match at this position
            if match.start() in match_positions:
                continue
            match_positions.add(match.start())
            
            start = max(0, match.start() - context_length)
            end = min(len(cleaned_text), match.end() + context_length)
            context = cleaned_text[start:end]
            
            # Further clean up context
            context = re.sub(r'\s+', ' ', context).strip()
            
            # Calculate relative position within context
            relative_start = match.start() - start
            relative_end = match.end() - start
            
            # Ensure we don't go out of bounds
            relative_start = max(0, relative_start)
            relative_end = min(len(context), relative_end)
            
            # Create highlighted context with better highlighting
            if relative_start < len(context) and relative_end <= len(context):
                highlighted_context = (
                    context[:relative_start] + 
                    f"<span class='search-highlight'>{context[relative_start:relative_end]}</span>" + 
                    context[relative_end:]
                )
                
                matches.append({
                    'context': highlighted_context,
                    'position': match.start(),
                    'match_number': len(matches) + 1,
                    'page_estimate': match.start() // 3000 + 1  # Rough page estimation
                })
    
    # Sort matches by position
    matches.sort(key=lambda x: x['position'])
    
    # Update match numbers after sorting
    for i, match in enumerate(matches):
        match['match_number'] = i + 1
        match['total_matches'] = len(matches)
    
    return matches

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    session_id = get_session_id()
    
    # Check if files are present
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'success': False, 'error': 'No files selected'}), 400
    
    processed_files = []
    errors = []
    
    for file in files:
        if file and file.filename and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                # Add timestamp to avoid conflicts
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
                filename = timestamp + filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Extract text from PDF
                text = extract_text_from_pdf(file_path)
                
                if text.strip():
                    # Analyze the document
                    print(f"Analyzing document: {file.filename}")
                    print(f"Text length: {len(text)} characters")
                    
                    dates = extract_dates_with_gemini(text)
                    parties = extract_parties_with_gemini(text)
                    key_terms = extract_key_terms_with_gemini(text)
                    
                    # Enhanced clause extraction with debugging
                    print("Extracting clauses from document...")
                    clauses = extract_clauses_from_text(text)
                    print(f"Found {len(clauses)} clauses")
                    
                    # Extract numbered bullet clauses (10+ words)
                    numbered_bullet_clauses = extract_numbered_bullet_clauses(text)
                    print(f"Found {len(numbered_bullet_clauses)} numbered bullet clauses with 10+ words")
                    
                    # Group clauses by type for better organization
                    clauses_by_type = {}
                    for clause in clauses:
                        clause_type = clause['type']
                        if clause_type not in clauses_by_type:
                            clauses_by_type[clause_type] = []
                        clauses_by_type[clause_type].append(clause)
                    
                    file_data = {
                        'id': str(uuid.uuid4()),
                        'filename': file.filename,
                        'stored_filename': filename,
                        'text': text[:10000],  # Store only first 10k chars to save space
                        'full_text_path': file_path,  # Store path to full text
                        'dates': dates,
                        'parties': parties,
                        'key_terms': key_terms,
                        'clauses': clauses[:50],  # Store top 50 clauses
                        'numbered_bullet_clauses': numbered_bullet_clauses,  # Store numbered bullet clauses
                        'clauses_by_type': clauses_by_type,  # Organized clauses
                        'clause_summary': {
                            'total_clauses': len(clauses),
                            'types_found': len(clauses_by_type),
                            'clause_types': sorted(clauses_by_type.keys())
                        },
                        'word_count': len(text.split()),
                        'upload_time': datetime.now().isoformat(),
                        'file_size': os.path.getsize(file_path)
                    }
                    
                    processed_files.append(file_data)
                    print(f"Successfully processed: {file.filename}")
                else:
                    errors.append(f"No text extracted from: {file.filename}")
                    print(f"No text extracted from: {file.filename}")
            except Exception as e:
                error_msg = f"Error processing {file.filename}: {str(e)}"
                errors.append(error_msg)
                print(error_msg)
                continue
        else:
            errors.append(f"Invalid file type: {file.filename}")
    
    if processed_files:
        # Load existing documents and add new ones
        existing_docs = load_documents_from_file(session_id)
        existing_docs.extend(processed_files)
        save_documents_to_file(session_id, existing_docs)
        
        response_data = {
            'success': True, 
            'files': processed_files, 
            'total': len(existing_docs),
            'processed_count': len(processed_files)
        }
        
        if errors:
            response_data['warnings'] = errors
        
        return jsonify(response_data)
    else:
        return jsonify({
            'success': False, 
            'error': 'No valid PDF files could be processed',
            'details': errors
        }), 400

@app.route('/documents')
def documents():
    """Display uploaded documents with analysis"""
    session_id = get_session_id()
    docs = load_documents_from_file(session_id)
    return render_template('documents.html', documents=docs)

@app.route('/document/<int:doc_index>')
def document_detail(doc_index):
    """Display detailed analysis of a specific document"""
    session_id = get_session_id()
    docs = load_documents_from_file(session_id)
    if 0 <= doc_index < len(docs):
        doc = docs[doc_index]
        
        # Check if numbered_bullet_clauses exist, if not extract them
        if 'numbered_bullet_clauses' not in doc:
            print(f"Extracting numbered bullet clauses for existing document: {doc.get('filename', 'Unknown')}")
            # Read the full text
            if 'full_text_path' in doc and os.path.exists(doc['full_text_path']):
                full_text = extract_text_from_pdf(doc['full_text_path'])
            else:
                full_text = doc.get('text', '')
            
            # Extract numbered bullet clauses
            numbered_bullet_clauses = extract_numbered_bullet_clauses(full_text)
            doc['numbered_bullet_clauses'] = numbered_bullet_clauses
            
            # Save the updated document
            docs[doc_index] = doc
            save_documents_to_file(session_id, docs)
            print(f"Found {len(numbered_bullet_clauses)} numbered bullet clauses with 10+ words")
        
        return render_template('document_detail.html', document=doc, doc_index=doc_index)
    return redirect(url_for('documents'))

@app.route('/remove_document/<int:doc_index>', methods=['POST'])
@app.route('/remove_document/<int:doc_index>', methods=['POST'])
def remove_document(doc_index):
    """Remove a document from the session"""
    session_id = get_session_id()
    docs = load_documents_from_file(session_id)
    
    if 0 <= doc_index < len(docs):
        # Remove the document from the list
        removed_doc = docs.pop(doc_index)
        
        # Save the updated documents list
        save_documents_to_file(session_id, docs)
        
        # Optionally remove the file from disk if it exists
        if 'file_path' in removed_doc and os.path.exists(removed_doc['file_path']):
            try:
                os.remove(removed_doc['file_path'])
            except:
                pass  # File might be in use or already deleted
                
        return jsonify({'success': True})
    
    return jsonify({'success': False, 'error': 'Document not found'})
#             except Exception as e:
#                 print(f"Error removing file: {e}")
        
#         flash(f'Document "{removed_doc.get("filename", "Unknown")}" has been removed successfully.', 'success')
#         return jsonify({'success': True, 'message': 'Document removed successfully'})
    
#     return jsonify({'success': False, 'message': 'Document not found'}), 404

# @app.route('/document/<int:doc_index>')
# def document_detail(doc_index):
#     session_id = get_session_id()
#     docs = load_documents_from_file(session_id)
#     if 0 <= doc_index < len(docs):
#         # Load full text if needed
#         doc = docs[doc_index]
#         if 'full_text_path' in doc and os.path.exists(doc['full_text_path']):
#             with open(doc['full_text_path'], 'rb') as f:
#                 pdf_reader = PyPDF2.PdfReader(f)
#                 doc['full_text'] = "".join([page.extract_text() for page in pdf_reader.pages])
#         return render_template('document_detail.html', document=doc, doc_index=doc_index)
#     return redirect(url_for('documents'))

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        session_id = get_session_id()
        docs = load_documents_from_file(session_id)
        
        # Get spell suggestions for the query
        spell_suggestions = suggest_corrections(query, session_id)
        
        search_results = []
        for i, doc in enumerate(docs):
            # Load full text for search with improved extraction
            full_text = doc.get('text', '')
            if 'full_text_path' in doc and os.path.exists(doc['full_text_path']):
                try:
                    with open(doc['full_text_path'], 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        extracted_pages = []
                        for page in pdf_reader.pages:
                            page_text = page.extract_text()
                            # Clean up common PDF extraction issues
                            page_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', page_text)  # Add space between words
                            page_text = re.sub(r'(\w)(\d)', r'\1 \2', page_text)  # Add space before numbers
                            page_text = re.sub(r'(\d)(\w)', r'\1 \2', page_text)  # Add space after numbers
                            page_text = re.sub(r'\s+', ' ', page_text)  # Normalize whitespace
                            extracted_pages.append(page_text)
                        full_text = " ".join(extracted_pages)
                except Exception as e:
                    print(f"Error extracting text for search: {e}")
                    pass
                    
            matches = search_text(full_text, query)
            if matches:
                search_results.append({
                    'document': doc,
                    'doc_index': i,
                    'matches': matches,
                    'match_count': len(matches)
                })
        
        return render_template('search_results.html', 
                             query=query, 
                             results=search_results,
                             total_matches=sum(r['match_count'] for r in search_results),
                             spell_suggestions=spell_suggestions)
    
    # Handle GET request - show search form
    return render_template('search.html')

@app.route('/document/<int:doc_index>/clauses')
def view_document_clauses(doc_index):
    """View all clauses from a specific document"""
    session_id = session.get('session_id')
    if not session_id:
        flash("No session found", "error")
        return redirect(url_for('index'))
        
    docs = load_documents_from_file(session_id)
    
    if doc_index >= len(docs):
        flash("Document not found", "error")
        return redirect(url_for('index'))
    
    document = docs[doc_index]
    
    # Load full document text for re-analysis if needed
    full_text = document.get('text', '')
    if 'full_text_path' in document and os.path.exists(document['full_text_path']):
        try:
            full_text = extract_text_from_pdf(document['full_text_path'])
        except:
            pass
    
    # Extract clauses from full text for comprehensive view
    all_clauses = extract_clauses_from_text(full_text)
    
    # Group clauses by type
    clauses_by_type = {}
    for clause in all_clauses:
        clause_type = clause['type']
        if clause_type not in clauses_by_type:
            clauses_by_type[clause_type] = []
        clauses_by_type[clause_type].append(clause)
    
    return render_template('document_clauses.html', 
                         document=document,
                         clauses=all_clauses,
                         clauses_by_type=clauses_by_type,
                         doc_index=doc_index)
    
    return render_template('search.html')

@app.route('/api/spell-check')
def spell_check_api():
    """API endpoint for real-time spell checking"""
    query = request.args.get('q', '').strip()
    session_id = get_session_id()
    
    if not query:
        return jsonify({'suggestions': []})
    
    suggestions = suggest_corrections(query, session_id)
    
    return jsonify({
        'query': query,
        'suggestions': suggestions,
        'has_suggestions': len(suggestions) > 0
    })

@app.route('/api/document/<int:doc_index>/full-text')
def get_document_full_text(doc_index):
    """Get full text of a document for enhanced search"""
    session_id = get_session_id()
    docs = load_documents_from_file(session_id)
    
    if 0 <= doc_index < len(docs):
        doc = docs[doc_index]
        
        # Try to get full text from file path first
        full_text = ""
        if 'full_text_path' in doc and os.path.exists(doc['full_text_path']):
            try:
                with open(doc['full_text_path'], 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    full_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            except Exception as e:
                print(f"Error reading PDF: {e}")
        
        if not full_text:
            full_text = doc.get('text', '')
        
        return jsonify({
            'success': True,
            'text': full_text,
            'filename': doc.get('filename', 'Unknown'),
            'word_count': len(full_text.split())
        })
    
    return jsonify({'success': False, 'error': 'Document not found'})

@app.route('/api/search')
def api_search():
    query = request.args.get('q', '').strip()
    session_id = get_session_id()
    docs = load_documents_from_file(session_id)
    
    results = []
    for i, doc in enumerate(docs):
        # Load full text for search
        full_text = doc.get('text', '')
        if 'full_text_path' in doc and os.path.exists(doc['full_text_path']):
            try:
                with open(doc['full_text_path'], 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    full_text = "".join([page.extract_text() for page in pdf_reader.pages])
            except:
                pass
                
        matches = search_text(full_text, query)
        if matches:
            results.append({
                'document_name': doc['filename'],
                'doc_index': i,
                'matches': matches[:5]  # Limit to first 5 matches per document
            })
    
    return jsonify(results)

@app.route('/clear')
def clear_session():
    session_id = get_session_id()
    
    # Clear stored documents
    file_path = os.path.join(DATA_FOLDER, f"{session_id}.pkl")
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Clear uploaded files for this session
    docs = load_documents_from_file(session_id)
    for doc in docs:
        if 'full_text_path' in doc and os.path.exists(doc['full_text_path']):
            try:
                os.remove(doc['full_text_path'])
            except Exception as e:
                print(f"Error deleting file: {e}")
    
    session.clear()
    flash('All documents cleared successfully!', 'info')
    return redirect(url_for('index'))

@app.route('/test-clauses')
def test_clauses():
    """Test route to check if clause extraction is working"""
    test_text = """
    1. This Agreement shall commence on the Effective Date and shall continue for a period of one (1) year.
    
    2. The parties agree to the following terms and conditions:
        a) Payment shall be made within thirty (30) days
        b) All disputes shall be resolved through arbitration
    
    3. Termination of this agreement may occur under the following circumstances:
    
    WHEREAS, the parties wish to enter into this agreement for mutual benefit;
    
    NOW THEREFORE, in consideration of the mutual covenants contained herein, the parties agree as follows:
    
    Article I. Definitions
    For purposes of this Agreement, the following terms shall have the meanings set forth below.
    """
    
    clauses = extract_clauses_from_text(test_text)
    return jsonify({
        'success': True,
        'test_text': test_text,
        'clauses_found': len(clauses),
        'clauses': clauses
    })

# Comprehensive rental agreement clauses database
STANDARD_RENTAL_CLAUSES = {
    "utilities": {
        "electricity": {
            "title": "Electricity Bill Responsibility",
            "description": "Clear specification of who pays electricity bills and utility charges",
            "suggested_text": "The LESSEE shall pay the electricity charges in respect of the schedule premises to the LESSOR as per the share of bill amount divided with other tenants during the period of stay and keep the installation intact.",
            "priority": "high",
            "keywords": ["electricity", "power", "utility", "bill", "charges", "installation", "share", "divided"]
        },
        "water": {
            "title": "Water and Sewerage Charges", 
            "description": "Responsibility for water bills and sewerage charges",
            "suggested_text": "The LESSEE shall pay all water charges, sewerage charges, and municipal taxes related to water supply directly to the concerned authorities during the tenancy period.",
            "priority": "high",
            "keywords": ["water", "sewerage", "municipal", "charges", "arrears"]
        },
        "gas": {
            "title": "Gas Connection and Bills",
            "description": "Gas supply and billing responsibility",
            "suggested_text": "The LESSEE may arrange for LPG/PNG gas connection at their own cost and shall be responsible for all gas bills and related charges during the tenancy.",
            "priority": "medium", 
            "keywords": ["gas", "lpg", "png", "connection", "cylinder", "stove"]
        },
        "internet": {
            "title": "Internet and Cable Connection",
            "description": "Telecommunication services responsibility",
            "suggested_text": "The Lessee may install internet, cable TV, or other telecommunication services at their own expense and shall be responsible for all related bills and charges.",
            "priority": "medium",
            "keywords": ["internet", "cable", "broadband", "wifi", "telecommunication"]
        }
    },
    "maintenance": {
        "minor_repairs": {
            "title": "Minor Repairs and Maintenance", 
            "description": "Responsibility for day-to-day maintenance and minor repairs",
            "suggested_text": "The Lessee shall be responsible for minor repairs and maintenance including but not limited to electrical fittings, plumbing fixtures, door handles, and general upkeep of the premises up to Rs. 2,000 per incident.",
            "priority": "high",
            "keywords": ["repair", "maintenance", "minor", "upkeep"]
        },
        "major_repairs": {
            "title": "Major Repairs and Structural Issues",
            "description": "Responsibility for major structural repairs", 
            "suggested_text": "The Lessor shall be responsible for major structural repairs including foundation, roof leakage, major plumbing issues, and electrical wiring problems that exceed Rs. 5,000 in cost.",
            "priority": "high",
            "keywords": ["major", "structural", "foundation", "roof", "wiring"]
        },
        "wear_tear": {
            "title": "Normal Wear and Tear",
            "description": "Definition of acceptable wear and tear vs damage",
            "suggested_text": "Normal wear and tear due to ordinary use shall be acceptable. However, any damage beyond normal wear including holes in walls, broken fixtures, or damaged flooring shall be repaired by the Lessee at their cost.",
            "priority": "medium",
            "keywords": ["wear", "tear", "damage", "ordinary", "use"]
        }
    },
    "legal": {
        "notice_period": {
            "title": "Notice Period for Termination",
            "description": "Required notice period before termination",
            "suggested_text": "Either party may terminate this agreement by giving 30 days written notice to the other party. The notice period shall commence from the date of receipt of the notice.",
            "priority": "high",
            "keywords": ["notice", "termination", "terminate", "period", "days"]
        },
        "lock_in": {
            "title": "Lock-in Period Clause",
            "description": "Minimum tenancy period restriction",
            "suggested_text": "This agreement shall have a lock-in period of 11 months from the commencement date, during which neither party may terminate the agreement except for breach of terms.",
            "priority": "medium",
            "keywords": ["lock", "lock-in", "minimum", "period", "months"]
        },
        "renewal": {
            "title": "Renewal Terms and Conditions",
            "description": "Process for agreement renewal",
            "suggested_text": "This agreement may be renewed for another term by mutual consent of both parties with a rent revision not exceeding 10% of the current rent, subject to a new agreement being executed.",
            "priority": "medium", 
            "keywords": ["renewal", "renew", "extend", "extension", "revision"]
        }
    },
    "security": {
        "deposit_conditions": {
            "title": "Security Deposit Return Conditions",
            "description": "Conditions for security deposit refund with proper acknowledgment",
            "suggested_text": "The LESSOR hereby acknowledges the receipt of security deposit amount which shall be refundable to the LESSEE at the time of vacating the premises without any interest subject to deductions towards arrears of rents, water, electricity, painting damages if any etc.",
            "priority": "high",
            "keywords": ["security", "deposit", "refund", "return", "deduct", "acknowledges", "receipt", "arrears", "damages"]
        },
        "deposit_payment": {
            "title": "Security Deposit Payment Schedule",
            "description": "Detailed payment schedule for security deposit",
            "suggested_text": "The LESSEE has paid a sum of Rs. [Amount] as security deposit with detailed payment schedule specifying dates and amounts paid in installments.",
            "priority": "high",
            "keywords": ["paid", "sum", "rupees", "installments", "schedule", "amount"]
        },
        "inspection": {
            "title": "Property Inspection Rights",
            "description": "Lessor's right to inspect the property with proper notice",
            "suggested_text": "The LESSEE shall permit the LESSOR or his authorized agents/assigns to inspect the schedule premises at any reasonable hours of the day time with prior notice.",
            "priority": "medium",
            "keywords": ["inspection", "inspect", "access", "premises", "notice", "authorized", "agents", "reasonable", "hours"]
        }
    },
    "rental_terms": {
        "rent_amount": {
            "title": "Monthly Rent Payment Terms",
            "description": "Clear specification of rent amount and payment schedule",
            "suggested_text": "The LESSEE shall pay a sum of Rs. [Amount] as rent towards the schedule premises and the same shall be payable to the LESSOR on or before 1st of every month.",
            "priority": "high",
            "keywords": ["rent", "monthly", "payable", "every", "month", "sum", "rupees", "schedule", "premises"]
        },
        "period": {
            "title": "Agreement Period and Tenure",
            "description": "Specific period and effective dates of the agreement",
            "suggested_text": "The period of this agreement shall be for 11 months and the same shall be effective from [start date] and shall terminate on [end date].",
            "priority": "high",
            "keywords": ["period", "agreement", "months", "effective", "commence", "terminate", "tenure"]
        },
        "rent_escalation": {
            "title": "Rent Increase for Extension",
            "description": "Terms for rent increase if tenant continues beyond agreement period",
            "suggested_text": "If the LESSEE desires to continue his/her stay for further period after the expiry of eleven months, the LESSEE shall pay 5% increase in rent for the extended period.",
            "priority": "medium",
            "keywords": ["continue", "stay", "further", "period", "expiry", "increase", "percent", "extended"]
        }
    },
    "usage": {
        "purpose": {
            "title": "Purpose of Use Restriction", 
            "description": "Restrictions on property usage for residential purposes only",
            "suggested_text": "The LESSEE shall use the premises only for the purpose of residence only and no commercial activities shall be permitted.",
            "priority": "high",
            "keywords": ["purpose", "residential", "residence", "commercial", "business", "illegal", "activities"]
        },
        "subletting": {
            "title": "Subletting and Assignment Restrictions",
            "description": "Restrictions on subletting the property to third parties",
            "suggested_text": "The LESSEE shall not sublet, under-let, re-let, part or whole of the premises to third parties during the period of this agreement without the prior permission of the LESSOR.",
            "priority": "high", 
            "keywords": ["sublet", "subletting", "under-let", "re-let", "assign", "transfer", "third", "party", "parties", "permission"]
        },
        "pets": {
            "title": "Pet Policy and Restrictions",
            "description": "Rules regarding keeping pets in the premises",
            "suggested_text": "The LESSEE shall not keep pets in the premises without the written consent of the LESSOR.",
            "priority": "medium",
            "keywords": ["pet", "pets", "animal", "dog", "cat", "consent"]
        },
        "parking": {
            "title": "Parking Space Allocation",
            "description": "Parking facility and rules",
            "suggested_text": "One designated parking space is provided with the premises. The LESSEE shall park only their registered vehicle in the allocated space and shall not obstruct common areas.",
            "priority": "medium",
            "keywords": ["parking", "vehicle", "car", "space", "allocated", "designated"]
        },
        "dangerous_substances": {
            "title": "Prohibition of Dangerous Substances",
            "description": "Restriction on storing inflammable or dangerous materials",
            "suggested_text": "The LESSEE shall not store any inflammable or combustible substance in the premises during the period of stay which will endanger human life and building condition.",
            "priority": "high",
            "keywords": ["inflammable", "combustible", "substance", "store", "endanger", "human", "building", "dangerous"]
        },
        "nuisance": {
            "title": "Nuisance and Disturbance Prevention",
            "description": "Rules to prevent disturbance to neighbors",
            "suggested_text": "The LESSEE shall not cause any nuisance, loud noise, or disturbance to neighbors or engage in illegal activities within the premises.",
            "priority": "medium",
            "keywords": ["nuisance", "noise", "disturbance", "neighbors", "illegal", "activities"]
        }
    },
    "maintenance": {
        "condition": {
            "title": "Property Condition Maintenance",
            "description": "Responsibility to keep premises in good condition",
            "suggested_text": "The LESSEE shall keep the schedule premises in good condition subject to normal wear and tear during the tenancy period.",
            "priority": "high",
            "keywords": ["condition", "good", "premises", "normal", "wear", "tear", "maintain"]
        },
        "minor_repairs": {
            "title": "Minor Repairs and Maintenance", 
            "description": "Lessee's responsibility for minor repairs",
            "suggested_text": "The LESSEE shall attend to the minor repairs to the premises during this period at his own cost including electrical fittings, plumbing fixtures, and general upkeep.",
            "priority": "high",
            "keywords": ["repair", "minor", "repairs", "attend", "own", "cost", "maintenance", "upkeep"]
        },
        "major_repairs": {
            "title": "Major Repairs and Structural Issues",
            "description": "Lessor's responsibility for major structural repairs", 
            "suggested_text": "The LESSOR shall be responsible for major structural repairs including foundation, roof leakage, major plumbing issues, and electrical wiring problems.",
            "priority": "high",
            "keywords": ["major", "structural", "foundation", "roof", "wiring", "lessor", "responsible"]
        },
        "fixtures_fittings": {
            "title": "Protection of Fixtures and Fittings",
            "description": "Responsibility to protect existing fixtures",
            "suggested_text": "The LESSEE shall not damage the existing fittings and fixtures of premises during the period of stay without the prior permission of the Lessor. Existing fittings include fans, gas stove, cylinder, and geyser.",
            "priority": "medium",
            "keywords": ["fixtures", "fittings", "damage", "existing", "fans", "stove", "cylinder", "geyser", "permission"]
        },
        "painting": {
            "title": "Painting and Whitewash Charges",
            "description": "Charges for painting/whitewash upon vacating",
            "suggested_text": "At the time of vacating the premises, the LESSEE shall pay half a month's rent towards whitewash/painting the premises as it was prior to occupation.",
            "priority": "medium",
            "keywords": ["painting", "whitewash", "vacating", "half", "month", "rent", "prior", "occupation"]
        }
    },
    "legal": {
        "notice_period": {
            "title": "Notice Period for Termination",
            "description": "Required notice period before termination by either party",
            "suggested_text": "Either party shall give two months prior notice to each other about vacating or terminating the rental agreement.",
            "priority": "high",
            "keywords": ["notice", "termination", "terminate", "period", "months", "prior", "either", "party", "vacating"]
        },
        "lock_in": {
            "title": "Lock-in Period and Early Termination",
            "description": "Penalties for early termination before minimum period",
            "suggested_text": "In case the LESSEE vacates before 11 months, 1 month's rent will be deducted and adjusted from the security deposit as penalty for early termination.",
            "priority": "high",
            "keywords": ["lock-in", "vacate", "before", "months", "deducted", "adjusted", "penalty", "early", "termination"]
        },
        "renewal": {
            "title": "Agreement Renewal Terms",
            "description": "Process for agreement renewal and extension",
            "suggested_text": "This agreement may be renewed for another term by mutual consent of both parties with appropriate rent revision, subject to a new agreement being executed.",
            "priority": "medium",
            "keywords": ["renewal", "renew", "extend", "extension", "revision", "mutual", "consent"]
        },
        "jurisdiction": {
            "title": "Jurisdiction and Legal Disputes",
            "description": "Court jurisdiction for dispute resolution",
            "suggested_text": "Any dispute arising out of this agreement shall be subject to the exclusive jurisdiction of the courts in [City/State].",
            "priority": "medium",
            "keywords": ["jurisdiction", "dispute", "court", "exclusive", "legal", "arising"]
        },
        "force_majeure": {
            "title": "Force Majeure Clause",
            "description": "Protection against uncontrollable circumstances",
            "suggested_text": "Neither party shall be held liable for non-performance of obligations due to natural calamities, government restrictions, pandemics, or other circumstances beyond their control.",
            "priority": "medium",
            "keywords": ["force", "majeure", "liable", "calamities", "government", "restrictions", "pandemic", "circumstances", "control"]
        },
        "default_payment": {
            "title": "Default in Rent Payment",
            "description": "Consequences of defaulting on rent payments",
            "suggested_text": "If the LESSEE defaults in paying rent for more than two consecutive months, the LESSOR shall have the right to terminate this agreement and adjust dues from the security deposit.",
            "priority": "high",
            "keywords": ["default", "defaults", "paying", "consecutive", "right", "terminate", "adjust", "dues"]
        },
        "alterations": {
            "title": "Alterations and Modifications",
            "description": "Restrictions on structural changes to property",
            "suggested_text": "The LESSEE shall not carry out any structural changes, alterations, or additions to the premises without the prior written consent of the LESSOR.",
            "priority": "medium",
            "keywords": ["alterations", "modifications", "structural", "changes", "additions", "written", "consent"]
        },
        "keys_handover": {
            "title": "Keys and Vacant Possession",
            "description": "Process for handing over keys upon vacating",
            "suggested_text": "On vacating, the LESSEE shall hand over all sets of keys and provide vacant possession of the premises to the LESSOR in good condition.",
            "priority": "medium",
            "keywords": ["keys", "handover", "vacant", "possession", "sets", "good", "condition"]
        },
        "government_dues": {
            "title": "Government Dues and Statutory Charges",
            "description": "Responsibility for taxes and statutory payments",
            "suggested_text": "Property tax, building insurance, and other statutory charges shall be borne by the LESSOR, whereas electricity, water, internet, and gas charges shall be borne by the LESSEE.",
            "priority": "medium",
            "keywords": ["property", "tax", "insurance", "statutory", "charges", "borne", "lessor", "lessee"]
        },
        "breach_termination": {
            "title": "Termination for Breach of Terms",
            "description": "Right to terminate for violation of agreement terms",
            "suggested_text": "If either party violates the terms of this agreement, the other party shall have the right to terminate this agreement by serving a written notice.",
            "priority": "high",
            "keywords": ["breach", "violates", "terms", "violation", "terminate", "written", "notice", "serving"]
        }
    }
}

def calculate_agreement_quality(suggestions, total_clauses=20):
    """Determine agreement quality based on missing clauses"""
    missing_high_priority = sum(1 for s in suggestions if s['priority'] == 'high')
    missing_medium_priority = sum(1 for s in suggestions if s['priority'] == 'medium')
    total_missing = len(suggestions)
    
    coverage_score = ((total_clauses - total_missing) / total_clauses) * 100
    
    if missing_high_priority == 0 and missing_medium_priority <= 2:
        rating = "Excellent"
        description = "Your rental agreement is comprehensive and covers all essential clauses with strong legal protection."
        color = "success"
    elif missing_high_priority <= 1 and missing_medium_priority <= 4:
        rating = "Good" 
        description = "Your agreement covers most important clauses but could benefit from a few additional protections."
        color = "primary"
    elif missing_high_priority <= 3 and missing_medium_priority <= 6:
        rating = "Okay"
        description = "Your agreement has basic coverage but is missing several important clauses that could leave you vulnerable."
        color = "warning" 
    else:
        rating = "Needs Improvement"
        description = "Your agreement is missing many critical clauses and needs significant improvements for proper legal protection."
        color = "danger"
    
    return {
        "rating": rating,
        "description": description,
        "color": color,
        "coverage_score": round(coverage_score, 1),
        "missing_high": missing_high_priority,
        "missing_medium": missing_medium_priority,
        "total_missing": total_missing
    }

def calculate_text_similarity(text1, text2):
    """Calculate simple text similarity using word overlap"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def analyze_agreement_clauses(document_text):
    """Analyze agreement and suggest missing clauses"""
    suggestions = []
    doc_text_lower = document_text.lower()
    
    # Check each category and clause
    for category, clauses in STANDARD_RENTAL_CLAUSES.items():
        for clause_key, clause_data in clauses.items():
            # Check if keywords related to this clause exist in document
            keywords_found = sum(1 for keyword in clause_data['keywords'] 
                                if keyword in doc_text_lower)
            
            # If very few or no keywords found, suggest this clause
            keyword_coverage = keywords_found / len(clause_data['keywords'])
            
            if keyword_coverage < 0.3:  # Less than 30% keywords found
                suggestions.append({
                    'title': clause_data['title'],
                    'description': clause_data['description'], 
                    'suggested_text': clause_data['suggested_text'],
                    'priority': clause_data['priority'],
                    'category': category,
                    'coverage': keyword_coverage
                })
    
    # Sort by priority (high first) and then by coverage (lower coverage first)
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    suggestions.sort(key=lambda x: (priority_order[x['priority']], x['coverage']))
    
    return suggestions

@app.route('/agreement-suggestions')
def agreement_suggestions():
    """Show agreement improvement suggestions page"""
    session_id = get_session_id()
    docs = load_documents_from_file(session_id)
    return render_template('agreement_suggestions.html', documents=docs)

@app.route('/api/analyze-agreement/<int:doc_index>')
def api_analyze_agreement(doc_index):
    """API endpoint to analyze agreement and return suggestions with quality assessment"""
    session_id = get_session_id()
    docs = load_documents_from_file(session_id)
    
    if 0 <= doc_index < len(docs):
        doc = docs[doc_index]
        
        # Get document text
        if 'full_text_path' in doc and os.path.exists(doc['full_text_path']):
            full_text = extract_text_from_pdf(doc['full_text_path'])
        else:
            full_text = doc.get('text', '')
        
        if not full_text:
            return jsonify({'error': 'No text found in document'}), 400
        
        # Analyze and get suggestions
        suggestions = analyze_agreement_clauses(full_text)
        
        # Calculate agreement quality
        quality_assessment = calculate_agreement_quality(suggestions)
        
        return jsonify({
            'success': True,
            'document_name': doc.get('filename', 'Unknown'),
            'suggestions': suggestions,
            'total_suggestions': len(suggestions),
            'quality': quality_assessment
        })
    
    return jsonify({'error': 'Document not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
