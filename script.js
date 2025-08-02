/*
================================================================================
IMPORTANT: HOW TO RUN THIS PROJECT
================================================================================
This project is designed to be run from a web server, not by opening the
index.html file directly. Deploying to a service like GitHub Pages (which you
have done) is the perfect way to run it.

For local testing, you must use a server that sends the correct security headers.

1. Install Node.js and npm from https://nodejs.org/
2. In your project folder, run: npm install -g serve
3. To start the server, run: serve -l 8000 -C
4. Open your browser to: http://localhost:8000
================================================================================
*/
document.addEventListener('DOMContentLoaded', () => {
    // --- CHATBOT LOGIC ---
    const chatDisplay = document.getElementById('chat-display');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const clearChatBtn = document.getElementById('clear-chat-btn');

    // --- ONNX MODEL SETUP (Question-Answering) ---
    let session = null;
    let tokenizer = null;
    let isModelReady = false;

    const context = `
        Paggy Xi Xang is a cutting-edge chatbot designed by a talented team.
        The backend was developed by Alex Martinez and Samira Khan, who focused on API design and database architecture.
        The frontend was engineered by Jordan Lee, who created the user interface and experience.
        The chatbot does not store any personal data or use cookies, ensuring user privacy.
        For legal information, users can consult the Terms of Service and Privacy Policy available in the footer.
        The project is managed under the entity Paggy Inc. and was last updated in July 2025.
    `;

    // --- New function to initialize the model on page load ---
    async function initializeModel() {
        const loadingMessage = document.createElement('div');
        loadingMessage.classList.add('loading-indicator', 'message-bubble');
        loadingMessage.textContent = 'Initializing AI model...';
        chatDisplay.appendChild(loadingMessage);
        userInput.disabled = true; // Disable input while loading

        try {
            // Using a specific, known-stable version
            const { AutoTokenizer } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1');
            
            const modelUrl = 'https://github.com/ohuhih/Skibidi/releases/download/test/transformer_chatbot.onnx';
            const tokenizerName = 'distilbert-base-uncased';

            loadingMessage.textContent = 'Loading tokenizer...';
            tokenizer = await AutoTokenizer.from_pretrained(tokenizerName, { local_files_only: false });
            
            loadingMessage.textContent = 'Loading model library...';
            // This import attaches the `ort` object to the global window scope.
            await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/ort.es6.min.js');
            
            // Configure the ONNX runtime
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/';
            
            // EDITED: Force single-threaded execution to prevent WASM initialization errors.
            // This is a key fix for the "RuntimeError: null function" issue.
            ort.env.wasm.numThreads = 1;

            loadingMessage.textContent = 'Loading model from GitHub...';
            session = await ort.InferenceSession.create(modelUrl, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });

            chatDisplay.removeChild(loadingMessage);
            const readyMessage = document.createElement('div');
            readyMessage.classList.add('initial-message');
            readyMessage.textContent = 'Model loaded. Ask a question!';
            chatDisplay.appendChild(readyMessage);
            
            isModelReady = true;
            userInput.disabled = false; // Re-enable input
            userInput.focus();

        } catch (error) {
            console.error('Failed to initialize the AI model:', error);
            loadingMessage.textContent = 'Error: Could not load the AI model.';
            loadingMessage.style.backgroundColor = '#f87171'; // Red color for error
            loadingMessage.style.color = '#7f1d1d';
        }
    }

    // This is the main function that now runs the ONNX model.
    async function getChatbotResponse(userQuestion) {
        if (!isModelReady) {
            appendMessage("The AI model is still initializing, please wait a moment.", 'chatbot');
            return;
        }

        const loadingMessage = document.createElement('div');
        loadingMessage.classList.add('loading-indicator', 'message-bubble');
        loadingMessage.textContent = 'Thinking...';
        chatDisplay.appendChild(loadingMessage);
        chatDisplay.scrollTop = chatDisplay.scrollHeight;

        try {
            const inputs = tokenizer(userQuestion, { text_pair: context, padding: true, truncation: true });
            
            const inputIds = new ort.Tensor('int64', inputs.input_ids.data, inputs.input_ids.dims);
            const attentionMask = new ort.Tensor('int64', inputs.attention_mask.data, inputs.attention_mask.dims);
            const tokenTypeIds = new ort.Tensor('int64', inputs.token_type_ids.data, inputs.token_type_ids.dims);

            const feeds = {
                input_ids: inputIds,
                attention_mask: attentionMask,
                token_type_ids: tokenTypeIds,
            };

            const outputs = await session.run(feeds);
            const startLogits = outputs.start_logits.data;
            const endLogits = outputs.end_logits.data;

            let bestStart = -1, bestEnd = -1, maxScore = -Infinity;
            for (let i = 0; i < startLogits.length; i++) {
                for (let j = i; j < endLogits.length; j++) {
                    if (startLogits[i] + endLogits[j] > maxScore) {
                        maxScore = startLogits[i] + endLogits[j];
                        bestStart = i;
                        bestEnd = j;
                    }
                }
            }
            
            let chatbotReply = "Sorry, I couldn't find an answer in my knowledge base.";
            if (bestStart !== -1 && maxScore > 0) { // Add a score threshold
                const answerTokens = inputs.input_ids.data.slice(bestStart, bestEnd + 1);
                chatbotReply = tokenizer.decode(answerTokens, { skip_special_tokens: true });
            }
            
            chatDisplay.removeChild(loadingMessage);
            appendMessage(chatbotReply, 'chatbot');

        } catch (error) {
            if (loadingMessage.parentNode === chatDisplay) {
                chatDisplay.removeChild(loadingMessage);
            }
            console.error('Error running the chatbot model:', error);
            appendMessage('Error: Could not run the model. Please check the console.', 'chatbot');
        }
    }

    // --- EXISTING HELPER FUNCTIONS ---

    function appendMessage(text, sender) {
        const messageWrapper = document.createElement('div');
        messageWrapper.classList.add('message-wrapper', `${sender}-wrapper`);

        const messageElement = document.createElement('div');
        messageElement.classList.add('message-bubble', `${sender}-message`);
        messageElement.textContent = text;

        const timestamp = document.createElement('span');
        timestamp.classList.add('timestamp');
        const now = new Date();
        const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        timestamp.textContent = timeString;

        const copyBtn = document.createElement('button');
        copyBtn.classList.add('copy-btn');
        copyBtn.innerHTML = '<i class="far fa-copy"></i>';
        copyBtn.title = 'Copy message';

        copyBtn.addEventListener('click', () => {
            const textToCopy = messageElement.textContent;
            const textArea = document.createElement('textarea');
            textArea.value = textToCopy;
            document.body.appendChild(textArea);
            textArea.select();
            try {
                document.execCommand('copy');
                copyBtn.innerHTML = '<i class="fas fa-check"></i>';
                setTimeout(() => {
                    copyBtn.innerHTML = '<i class="far fa-copy"></i>';
                }, 1500);
            } catch (err) {
                console.error('Failed to copy text: ', err);
            }
            document.body.removeChild(textArea);
        });

        messageWrapper.appendChild(messageElement);
        messageWrapper.appendChild(copyBtn);
        messageWrapper.appendChild(timestamp);

        chatDisplay.appendChild(messageWrapper);
        chatDisplay.scrollTop = chatDisplay.scrollHeight;
    }

    function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;

        appendMessage(message, 'user');
        userInput.value = '';
        getChatbotResponse(message);
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    clearChatBtn.addEventListener('click', () => {
        const messagesToRemove = chatDisplay.querySelectorAll('.message-wrapper, .loading-indicator, .initial-message');
        messagesToRemove.forEach(msg => {
            // Don't remove the very first initial message
            if (!msg.classList.contains('initial-message')) {
                msg.remove();
            }
        });
    });

    // --- MODAL AND BANNER LOGIC ---
    const cookieBanner = document.getElementById('cookie-banner');
    const cookieDismissBtn = document.getElementById('cookie-dismiss-btn');

    if (cookieBanner && cookieDismissBtn) {
        cookieDismissBtn.addEventListener('click', () => {
            cookieBanner.classList.add('hidden');
        });
    }

    const termsModal = document.getElementById('terms-modal');
    const openTermsLink = document.getElementById('open-terms-link');
    const closeTermsBtn = document.getElementById('close-terms-btn');

    if (termsModal && openTermsLink && closeTermsBtn) {
        termsModal.classList.add('visible');
        openTermsLink.addEventListener('click', (e) => {
            e.preventDefault();
            termsModal.classList.add('visible');
        });
        closeTermsBtn.addEventListener('click', () => {
            termsModal.classList.remove('visible');
        });
        termsModal.addEventListener('click', (e) => {
            if (e.target === termsModal) {
                termsModal.classList.remove('visible');
            }
        });
    }

    const privacyModal = document.getElementById('privacy-modal');
    const openPrivacyLink = document.getElementById('open-privacy-link');
    const closePrivacyBtn = document.getElementById('close-privacy-btn');

    if (privacyModal && openPrivacyLink && closePrivacyBtn) {
        openPrivacyLink.addEventListener('click', (e) => {
            e.preventDefault();
            privacyModal.classList.add('visible');
        });
        closePrivacyBtn.addEventListener('click', () => {
            privacyModal.classList.remove('visible');
        });
        privacyModal.addEventListener('click', (e) => {
            if (e.target === privacyModal) {
                privacyModal.classList.remove('visible');
            }
        });
    }
    
    // --- Initialize the model when the page loads ---
    initializeModel();
});
