# Legal Document Analyzer

A powerful web application for analyzing legal documents using AI-powered text analysis. Extract important dates, parties information (lessor/lessee), key terms, and search through documents with intelligent highlighting.

## Features

### 🎯 Key Capabilities
- **PDF Document Upload**: Upload multiple PDF legal documents
- **AI-Powered Analysis**: Uses Google Gemini API for intelligent text extraction
- **Important Dates Extraction**: Automatically identifies and categorizes important dates
- **Parties Identification**: Finds lessors, lessees, and other parties with contact details
- **Key Terms Analysis**: Extracts rent amounts, deposits, duration, and property details
- **Smart Search Engine**: Search across all documents with context and highlighting
- **Named Entity Recognition**: Uses spaCy for identifying persons, organizations, locations, etc.

### 🔍 Search Features
- **Live Search**: Real-time search results as you type
- **Context Highlighting**: See search terms highlighted with surrounding context
- **Cross-Document Search**: Search across all uploaded documents simultaneously
- **Export Results**: Export search results to text files
- **Copy to Clipboard**: Easy copying of search results and document sections

### 📊 Analysis Features
- **Document Summary**: Quick overview of each document's key information
- **Entity Extraction**: Automatic identification of people, places, organizations, dates, and monetary amounts
- **Financial Information**: Rent amounts, security deposits, and payment terms
- **Timeline Analysis**: Important dates with their significance and context
- **Property Details**: Address and property description extraction

## Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key

### Setup Instructions

1. **Clone or Download** the project files to your local machine

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy Model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Run Setup Script** (Optional but recommended):
   ```bash
   python setup.py
   ```

5. **Configure API Key**:
   - Open `app.py`
   - Update line 28 with your Gemini API key:
     ```python
     GEMINI_API_KEY = "your-api-key-here"
     ```

6. **Run the Application**:
   ```bash
   python app.py
   ```

7. **Access the Application**:
   - Open your web browser
   - Navigate to `http://localhost:5000`

## Usage Guide

### 1. Upload Documents
- Go to the home page
- Click on "Choose Files" or drag and drop PDF files
- Click "Analyze Documents" to process them

### 2. View Analysis Results
- Navigate to "Documents" to see all analyzed documents
- Click "View Full Analysis" on any document for detailed information
- Review extracted dates, parties, and key terms

### 3. Search Documents
- Go to "Search" in the navigation
- Enter search terms in the search box
- View results with highlighted matches and context
- Click on document names to view full documents

### 4. Export and Share
- Use the copy buttons to copy text snippets
- Export search results using the export functionality
- Print or save document analysis results

## API Endpoints

### File Upload
- **POST** `/upload`
  - Upload PDF files for analysis
  - Returns: JSON with analysis results

### Search
- **GET** `/api/search?q={query}`
  - Search across all documents
  - Returns: JSON with search results and matches

### Document Management
- **GET** `/documents`
  - View all analyzed documents
- **GET** `/document/{index}`
  - View specific document details
- **GET** `/clear`
  - Clear all uploaded documents and data

## Technologies Used

### Backend
- **Flask**: Web framework
- **Google Gemini AI**: Advanced text analysis and extraction
- **spaCy**: Named entity recognition and NLP
- **NLTK**: Text processing and tokenization
- **PyPDF2**: PDF text extraction

### Frontend
- **Bootstrap 5**: Responsive UI framework
- **Font Awesome**: Icons and visual elements
- **JavaScript**: Interactive functionality and live search
- **HTML5/CSS3**: Modern web standards

## Configuration

### Environment Variables
You can use environment variables instead of hardcoding the API key:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Then update `app.py`:
```python
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your-fallback-key')
```

### Production Deployment
For production deployment:

1. **Update Secret Key**:
   ```python
   app.secret_key = 'your-secure-secret-key-here'
   ```

2. **Use Environment Variables** for sensitive data

3. **Configure HTTPS** and proper security headers

4. **Use a Production WSGI Server**:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

## File Structure

```
Legal/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── setup.py              # Setup script
├── README.md             # This file
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Home page
│   ├── documents.html    # Documents listing
│   ├── document_detail.html # Document details
│   ├── search.html       # Search page
│   └── search_results.html # Search results
├── static/               # Static files
│   ├── style.css         # Custom CSS
│   └── script.js         # JavaScript functionality
└── uploads/              # Uploaded files (created automatically)
```

## Features in Detail

### Document Analysis
The application uses Google Gemini AI to extract:
- **Dates**: Agreement dates, commencement dates, expiry dates
- **Parties**: Names, addresses, and roles of lessors and lessees
- **Financial Terms**: Rent amounts, security deposits, payment schedules
- **Property Information**: Addresses, descriptions, and property details
- **Legal Clauses**: Important terms and conditions

### Search Functionality
- **Real-time Search**: Results appear as you type
- **Context Aware**: Shows surrounding text for better understanding
- **Multi-document**: Search across all uploaded documents
- **Highlighting**: Visual emphasis on matching terms
- **Export Options**: Save search results for later use

### Security Features
- **File Type Validation**: Only PDF files are accepted
- **File Size Limits**: Maximum 16MB per file
- **Secure Filenames**: Automatic filename sanitization
- **Session Management**: Secure session handling

## Troubleshooting

### Common Issues

1. **spaCy Model Not Found**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Gemini API Errors**:
   - Check your API key is correct
   - Ensure you have API quota available
   - Verify internet connectivity

3. **PDF Processing Errors**:
   - Ensure PDFs contain extractable text
   - Check file size limits (16MB max)
   - Try with different PDF files

4. **Import Errors**:
   - Run the setup script: `python setup.py`
   - Manually install missing packages
   - Check Python version (3.8+ required)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the error messages in the browser console
3. Check the terminal output for detailed error information

## License

This project is for educational and research purposes. Please ensure compliance with all applicable laws and regulations when analyzing legal documents.

---

**Note**: This application is designed to assist with document analysis but should not replace professional legal advice. Always consult with qualified legal professionals for important legal matters.
