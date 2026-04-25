const API_BASE = "http://localhost:8000/api";
let currentSessionId = "session_" + Math.random().toString(36).substr(2, 9);

// DOM Elements
const tabBtns = document.querySelectorAll('.tab-btn');
const views = document.querySelectorAll('.view');
const newChatBtn = document.getElementById('new-chat-btn');

const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatMessages = document.getElementById('chat-messages');

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');
const docList = document.getElementById('doc-list');
const docCount = document.getElementById('doc-count');

const historyList = document.getElementById('history-list');
const clearHistoryBtn = document.getElementById('clear-history-btn');

// --- Navigation ---
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // Update active button
        tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update active view
        const targetViewId = btn.getAttribute('data-tab');
        views.forEach(view => {
            if (view.id === targetViewId) {
                view.classList.remove('hidden');
            } else {
                view.classList.add('hidden');
            }
        });

        // Load data depending on view
        if (targetViewId === 'docs-view') loadDocuments();
        if (targetViewId === 'history-view') loadHistory();
    });
});

newChatBtn.addEventListener('click', () => {
    currentSessionId = "session_" + Math.random().toString(36).substr(2, 9);
    chatMessages.innerHTML = `
        <div class="message assistant">
            <div class="avatar">N</div>
            <div class="message-content">
                New session started. How can I help you with your documents?
            </div>
        </div>
    `;
    // Switch to chat tab
    document.querySelector('[data-tab="chat-view"]').click();
});

// --- Chat Logic ---
async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    // Append user message
    appendMessage('user', text);
    chatInput.value = '';

    // Append loading state
    const loadingId = appendMessage('assistant', 'Thinking...', true);

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: currentSessionId,
                question: text
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            updateMessage(loadingId, `Error: ${data.error}`);
        } else {
            updateMessage(loadingId, data.answer, data.sources);
        }
    } catch (err) {
        updateMessage(loadingId, 'Failed to connect to the server. Is the API running?');
    }
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

function appendMessage(role, text, isLoading = false) {
    const id = 'msg_' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    msgDiv.id = id;
    
    const avatar = role === 'assistant' ? 'N' : 'U';
    
    msgDiv.innerHTML = `
        <div class="avatar">${avatar}</div>
        <div class="message-content" style="${isLoading ? 'opacity: 0.7;' : ''}">
            ${escapeHTML(text)}
        </div>
    `;
    
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
}

function updateMessage(id, text, sources = []) {
    const msgDiv = document.getElementById(id);
    if (!msgDiv) return;
    
    let contentHtml = escapeHTML(text).replace(/\n/g, '<br>');
    
    if (sources && sources.length > 0) {
        contentHtml += `<div class="sources-box">`;
        // Deduplicate sources by document and page
        const uniqueSources = [];
        const seen = new Set();
        sources.forEach(s => {
            const key = `${s.document}_${s.page}`;
            if (!seen.has(key)) {
                seen.add(key);
                uniqueSources.push(s);
            }
        });
        
        uniqueSources.forEach((s, idx) => {
            contentHtml += `<span class="source-tag">[${idx + 1}] ${s.document} (p. ${s.page})</span>`;
        });
        contentHtml += `</div>`;
    }
    
    const contentDiv = msgDiv.querySelector('.message-content');
    contentDiv.innerHTML = contentHtml;
    contentDiv.style.opacity = '1';
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// --- Documents Logic ---

async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE}/documents`);
        const data = await response.json();
        
        docCount.textContent = data.total;
        docList.innerHTML = '';
        
        if (data.total === 0) {
            docList.innerHTML = '<p style="color: var(--text-secondary)">No documents found. Upload one above!</p>';
            return;
        }
        
        data.documents.forEach(doc => {
            const item = document.createElement('div');
            item.className = 'doc-item';
            item.innerHTML = `
                <div class="doc-info">
                    <h4>${doc.name}</h4>
                    <p>${doc.pages} pages • ${doc.chunks} indexed chunks</p>
                </div>
                <button class="danger-btn" onclick="deleteDocument('${doc.name}')">Delete</button>
            `;
            docList.appendChild(item);
        });
    } catch (err) {
        console.error("Failed to load documents", err);
    }
}

async function deleteDocument(name) {
    if (!confirm(`Are you sure you want to delete ${name}?`)) return;
    
    try {
        await fetch(`${API_BASE}/documents/${encodeURIComponent(name)}`, {
            method: 'DELETE'
        });
        loadDocuments();
    } catch (err) {
        alert("Failed to delete document");
    }
}

// Drag and drop upload
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFiles(e.dataTransfer.files);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
        handleFiles(fileInput.files);
    }
});

async function handleFiles(files) {
    uploadStatus.classList.remove('hidden');
    uploadStatus.textContent = 'Uploading and indexing...';
    
    for (let file of files) {
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            alert(`Skipping ${file.name}: Only PDFs are supported`);
            continue;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const res = await fetch(`${API_BASE}/documents`, {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
        } catch (err) {
            alert(`Failed to upload ${file.name}: ${err.message}`);
        }
    }
    
    uploadStatus.textContent = 'Upload complete!';
    setTimeout(() => {
        uploadStatus.classList.add('hidden');
        fileInput.value = '';
    }, 3000);
    
    loadDocuments();
}

// --- History Logic ---
async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE}/history/${currentSessionId}`);
        const data = await response.json();
        
        historyList.innerHTML = '';
        
        if (data.length === 0) {
            historyList.innerHTML = '<p style="color: var(--text-secondary)">No chat history in this session yet.</p>';
            return;
        }
        
        for (let i = 0; i < data.length; i += 2) {
            const userMsg = data[i];
            const botMsg = data[i + 1];
            
            if (!userMsg || !botMsg) continue;
            
            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = `
                <div class="history-q">Q: ${escapeHTML(userMsg.content)}</div>
                <div class="history-a">A: ${escapeHTML(botMsg.content)}</div>
            `;
            historyList.appendChild(item);
        }
    } catch (err) {
        console.error("Failed to load history", err);
    }
}

clearHistoryBtn.addEventListener('click', async () => {
    if (!confirm("Clear this session's history?")) return;
    
    try {
        await fetch(`${API_BASE}/history/${currentSessionId}`, { method: 'DELETE' });
        chatMessages.innerHTML = `
            <div class="message assistant">
                <div class="avatar">N</div>
                <div class="message-content">History cleared. How can I help you?</div>
            </div>
        `;
        loadHistory();
    } catch (err) {
        console.error(err);
    }
});

// Utils
function escapeHTML(str) {
    return str.replace(/[&<>'"]/g, 
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag)
    );
}
