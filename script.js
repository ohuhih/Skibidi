/*
================================================================================
FINAL WORKING VERSION
================================================================================
This version is designed to work with your specific `src`/`tgt` model. It
manually loads all necessary files from their exact URLs and uses the correct
autoregressive generation loop for your model's architecture.
================================================================================
*/
document.addEventListener('DOMContentLoaded', () => {
    // --- UI ELEMENTS ---
    const chatDisplay = document.getElementById('chat-display');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const clearChatBtn = document.getElementById('clear-chat-btn');

    // --- MODEL AND TOKENIZER SETUP ---
    let session = null;
    let tokenizer = null;
    let isModelReady = false;
    const maxGenerationLength = 50; // Max number of tokens to generate

    // --- New function to initialize the model on page load ---
    async function initializeModel() {
        const loadingMessage = document.createElement('div');
        loadingMessage.classList.add('initial-message'); // Use the centered text style
        loadingMessage.textContent = 'Initializing AI model...';
        chatDisplay.appendChild(loadingMessage);
        userInput.disabled = true;
        sendButton.disabled = true;

        try {
            // EDITED: Import the AutoTokenizer for a more robust loading process.
            const { AutoTokenizer } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1');
            
            // Define the exact URLs for all necessary files.
            const modelUrl = 'https://huggingface.co/Nayusai/chtbot/resolve/main/onnx/model.onnx';
            // This is the base path to the folder containing your tokenizer files.
            const tokenizerPath = 'https://huggingface.co/Nayusai/chtbot/raw/main/';

            loadingMessage.textContent = 'Loading tokenizer...';
            // EDITED: Use AutoTokenizer.from_pretrained to correctly load the tokenizer
            // from your configuration files. This is the fix for the '[UNK]' token issue.
            tokenizer = await AutoTokenizer.from_pretrained(tokenizerPath);
            
            // Configure the ONNX runtime
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/';
            ort.env.wasm.numThreads = 1;

            loadingMessage.textContent = 'Loading model...';
            session = await ort.InferenceSession.create(modelUrl, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });

            chatDisplay.removeChild(loadingMessage);
            const readyMessage = document.createElement('div');
            readyMessage.classList.add('initial-message');
            readyMessage.textContent = 'Model loaded. You can start chatting!';
            chatDisplay.appendChild(readyMessage);
            
            isModelReady = true;
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();

        } catch (error) {
            console.error('Failed to initialize the AI model:', error);
            loadingMessage.textContent = 'Error: Could not load the AI model.';
            loadingMessage.style.color = '#f87171'; // Red color for error
        }
    }
    
    // --- Autoregressive Generation for src/tgt models ---
    async function generateText(prompt) {
        if (!tokenizer || !session) {
            throw new Error("Tokenizer or session not initialized.");
        }

        // 1. Encode the user's prompt to get the `src` tensor
        const srcIds = tokenizer.encode(prompt);
        const srcTensor = new ort.Tensor('int64', BigInt64Array.from(srcIds.map(BigInt)), [1, srcIds.length]);

        // 2. Initialize the `tgt` tensor with the Beginning-Of-Sentence (BOS) token
        const bosTokenId = tokenizer.cls_token_id || 0;
        let tgtIds = [bosTokenId];
        
        // This will store the model's attention cache (past key values)
        let pastKeyValues = null;

        // 3. Autoregressively generate tokens
        for (let i = 0; i < maxGenerationLength; i++) {
            let feeds;
            let currentTgtIds;

            // On the first step, the target sequence is just the BOS token.
            // On subsequent steps, it's only the *last* generated token.
            if (i === 0) {
                currentTgtIds = tgtIds;
            } else {
                currentTgtIds = [tgtIds[tgtIds.length - 1]];
            }

            const tgtTensor = new ort.Tensor('int64', BigInt64Array.from(currentTgtIds.map(BigInt)), [1, currentTgtIds.length]);

            // On the first step, we provide `src`. On later steps, we provide the attention cache.
            if (i === 0) {
                feeds = { src: srcTensor, tgt: tgtTensor };
            } else {
                feeds = { src: srcTensor, tgt: tgtTensor, ...pastKeyValues };
            }
            
            const results = await session.run(feeds);
            const logits = results.output.data;

            // Get the logits for the very last generated token
            const vocabSize = results.output.dims[2];
            const lastTokenLogits = logits.slice((currentTgtIds.length - 1) * vocabSize, currentTgtIds.length * vocabSize);

            // Sample the next token ID (greedy search)
            let maxLogit = -Infinity;
            let nextTokenId = 0;
            for (let j = 0; j < vocabSize; j++) {
                if (lastTokenLogits[j] > maxLogit) {
                    maxLogit = lastTokenLogits[j];
                    nextTokenId = j;
                }
            }
            
            // Stop if we generate the End-Of-Sentence (EOS) token
            const eosTokenId = tokenizer.sep_token_id || 1;
            if (nextTokenId === eosTokenId) {
                break;
            }

            // Add the new token to our generated sequence
            tgtIds.push(nextTokenId);
            
            // Update the attention cache for the next iteration
            pastKeyValues = {};
            for (const key in results) {
                if (key.startsWith('present')) {
                    pastKeyValues[key.replace('present', 'past_key_values')] = results[key];
                }
            }
        }

        // 4. Decode the generated tokens, skipping the initial BOS token
        return tokenizer.decode(tgtIds.slice(1));
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
            const reply = await generateText(userQuestion);
            chatDisplay.removeChild(loadingMessage);
            appendMessage(reply || "...", 'chatbot');

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
