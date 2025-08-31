/* JavaScript for Legal Document Analyzer */

// Global variables
let uploadedDocuments = [];
let currentSearchQuery = '';

// Document ready
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Initialize tooltips
    if (typeof bootstrap !== 'undefined') {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Initialize file upload drag and drop
    initializeFileUpload();
    
    // Initialize search functionality
    initializeSearch();
    
    // Initialize copy to clipboard
    initializeCopyToClipboard();
}

function initializeFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.querySelector('.file-upload-area');
    
    if (!fileInput || !uploadArea) return;
    
    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            fileInput.dispatchEvent(new Event('change'));
        }
    });
    
    // File input change
    fileInput.addEventListener('change', function() {
        const files = this.files;
        if (files.length > 0) {
            updateFileInputDisplay(files);
        }
    });
}

function updateFileInputDisplay(files) {
    const fileNames = Array.from(files).map(file => file.name).join(', ');
    const button = document.getElementById('browseButton');
    if (button && files.length > 0) {
        button.innerHTML = `<i class="fas fa-check me-2"></i>Selected: ${files.length} file(s)`;
        button.classList.remove('btn-primary');
        button.classList.add('btn-success');
    }
}

function initializeSearch() {
    // Live search functionality
    const searchInputs = document.querySelectorAll('input[type="search"], .search-input');
    
    searchInputs.forEach(input => {
        let searchTimeout;
        
        input.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            const query = this.value.trim();
            
            if (query.length >= 2) {
                searchTimeout = setTimeout(() => {
                    performLiveSearch(query);
                }, 300);
            } else {
                clearSearchResults();
            }
        });
    });
}

function performLiveSearch(query) {
    if (!query) return;
    
    showSearchLoading();
    
    fetch(`/api/search?q=${encodeURIComponent(query)}`)
        .then(response => {
            if (!response.ok) throw new Error('Search failed');
            return response.json();
        })
        .then(data => {
            displaySearchResults(data, query);
        })
        .catch(error => {
            console.error('Search error:', error);
            showSearchError('Search failed. Please try again.');
        })
        .finally(() => {
            hideSearchLoading();
        });
}

function displaySearchResults(results, query) {
    const container = document.getElementById('searchResults') || document.getElementById('liveSearchResults');
    if (!container) return;
    
    if (results.length === 0) {
        container.innerHTML = `
            <div class="alert alert-info fade-in">
                <i class="fas fa-info-circle me-2"></i>
                No results found for "${escapeHtml(query)}".
            </div>
        `;
        return;
    }
    
    const totalMatches = results.reduce((sum, result) => sum + result.matches.length, 0);
    
    let html = `
        <div class="card fade-in">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">
                    <i class="fas fa-search-plus me-2"></i>
                    Search Results for "${escapeHtml(query)}"
                    <span class="badge bg-light text-primary ms-2">${totalMatches} matches</span>
                </h5>
            </div>
            <div class="card-body">
    `;
    
    results.forEach((result, index) => {
        html += createSearchResultHtml(result, index);
    });
    
    html += '</div></div>';
    container.innerHTML = html;
    
    // Initialize copy buttons for new content
    initializeCopyToClipboard();
}

function createSearchResultHtml(result, index) {
    return `
        <div class="search-result-item mb-4 ${index > 0 ? 'border-top pt-4' : ''}">
            <div class="d-flex justify-content-between align-items-start mb-3">
                <h6 class="text-primary mb-0">
                    <i class="fas fa-file-pdf me-2"></i>
                    ${escapeHtml(result.document_name)}
                </h6>
                <div>
                    <span class="badge bg-primary me-2">${result.matches.length} matches</span>
                    <a href="/document/${result.doc_index}" class="btn btn-sm btn-outline-primary">
                        <i class="fas fa-eye me-1"></i>View Document
                    </a>
                </div>
            </div>
            
            <div class="search-matches">
                ${result.matches.map((match, matchIndex) => `
                    <div class="search-match border-start border-primary border-3 ps-3 mb-3">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <small class="text-muted">Match ${matchIndex + 1} • Position: ${match.position}</small>
                            <button class="btn btn-sm btn-outline-secondary copy-btn" 
                                    data-text="${escapeHtml(stripHtml(match.context))}"
                                    title="Copy to clipboard">
                                <i class="fas fa-copy"></i>
                            </button>
                        </div>
                        <div class="search-context">${match.context}</div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

function initializeCopyToClipboard() {
    document.querySelectorAll('.copy-btn').forEach(button => {
        // Remove existing event listeners
        button.replaceWith(button.cloneNode(true));
    });
    
    // Add new event listeners
    document.querySelectorAll('.copy-btn').forEach(button => {
        button.addEventListener('click', function() {
            const text = this.dataset.text;
            
            if (navigator.clipboard) {
                navigator.clipboard.writeText(text).then(() => {
                    showCopySuccess(this);
                }).catch(err => {
                    console.error('Copy failed:', err);
                    fallbackCopyToClipboard(text);
                });
            } else {
                fallbackCopyToClipboard(text);
            }
        });
    });
}

function showCopySuccess(button) {
    const originalContent = button.innerHTML;
    button.innerHTML = '<i class="fas fa-check text-success"></i>';
    button.disabled = true;
    
    setTimeout(() => {
        button.innerHTML = originalContent;
        button.disabled = false;
    }, 2000);
}

function fallbackCopyToClipboard(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.opacity = '0';
    document.body.appendChild(textArea);
    textArea.select();
    
    try {
        document.execCommand('copy');
        showToast('Text copied to clipboard!', 'success');
    } catch (err) {
        console.error('Fallback copy failed:', err);
        showToast('Copy failed. Please select and copy manually.', 'error');
    }
    
    document.body.removeChild(textArea);
}

function showSearchLoading() {
    const container = document.getElementById('searchResults') || document.getElementById('liveSearchResults');
    if (container) {
        container.innerHTML = `
            <div class="text-center py-4">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Searching...</span>
                </div>
                <p class="mt-2 text-muted">Searching documents...</p>
            </div>
        `;
    }
}

function hideSearchLoading() {
    // Loading will be replaced by results or error message
}

function showSearchError(message) {
    const container = document.getElementById('searchResults') || document.getElementById('liveSearchResults');
    if (container) {
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${escapeHtml(message)}
            </div>
        `;
    }
}

function clearSearchResults() {
    const container = document.getElementById('searchResults') || document.getElementById('liveSearchResults');
    if (container) {
        container.innerHTML = '';
    }
}

function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center text-white bg-${type === 'error' ? 'danger' : type}" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-${type === 'success' ? 'check' : type === 'error' ? 'times' : 'info'}-circle me-2"></i>
                    ${escapeHtml(message)}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    // Initialize and show toast
    if (typeof bootstrap !== 'undefined') {
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, { delay: 3000 });
        toast.show();
        
        // Remove toast element after it's hidden
        toastElement.addEventListener('hidden.bs.toast', function() {
            toastElement.remove();
        });
    } else {
        // Fallback for when Bootstrap is not available
        setTimeout(() => {
            const toastElement = document.getElementById(toastId);
            if (toastElement) {
                toastElement.remove();
            }
        }, 3000);
    }
}

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function stripHtml(html) {
    const div = document.createElement('div');
    div.innerHTML = html;
    return div.textContent || div.innerText || '';
}

function formatFileSize(bytes) {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Export functionality
function exportSearchResults(query, results) {
    const exportData = {
        query: query,
        exportDate: new Date().toISOString(),
        totalMatches: results.reduce((sum, r) => sum + r.matches.length, 0),
        results: results
    };
    
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `search_results_${query.replace(/[^a-z0-9]/gi, '_')}_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    
    URL.revokeObjectURL(link.href);
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+/ or Cmd+/ to focus search
    if ((e.ctrlKey || e.metaKey) && e.key === '/') {
        e.preventDefault();
        const searchInput = document.querySelector('input[name="query"], .search-input');
        if (searchInput) {
            searchInput.focus();
        }
    }
    
    // Escape to clear search
    if (e.key === 'Escape') {
        const searchInput = document.querySelector('input[name="query"], .search-input');
        if (searchInput && searchInput === document.activeElement) {
            searchInput.value = '';
            clearSearchResults();
        }
    }
});

// Progress tracking for file uploads
function trackUploadProgress(files) {
    const progressBar = document.querySelector('.progress-bar');
    if (!progressBar) return;
    
    let loaded = 0;
    const total = files.length;
    
    // Simulate progress (in real implementation, you'd track actual upload progress)
    const interval = setInterval(() => {
        loaded += Math.random() * 20;
        const percent = Math.min((loaded / total) * 100, 95);
        progressBar.style.width = percent + '%';
        
        if (percent >= 95) {
            clearInterval(interval);
        }
    }, 100);
    
    return interval;
}
