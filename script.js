/*
================================================================================
FINAL WORKING VERSION (with Dual Static Padding)
================================================================================
This version implements a dual static padding strategy for both `src` and `tgt`
tensors to resolve the persistent ONNX Runtime memory allocation error. Both
inputs will now have a fixed size, which is the most robust solution.
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
    const maxSourceLength = 128; // Max number of tokens for the user's input

    // --- New function to initialize the model on page load ---
    async function initializeModel() {
        const loadingMessage = document.createElement('div');
        loadingMessage.classList.add('initial-message'); // Use the centered text style
        loadingMessage.textContent = 'Initializing AI model...';
        chatDisplay.appendChild(loadingMessage);
        userInput.disabled = true;
        sendButton.disabled = true;

        try {
            // Using a specific, known-stable version of the transformers library
            const { BertTokenizer } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1');
            
            // Define the exact URLs for all necessary files.
            const modelUrl = 'https://huggingface.co/Nayusai/chtbot/resolve/main/onnx/model.onnx';
            const tokenizerUrl = 'https://huggingface.co/Nayusai/chtbot/raw/main/tokenizer.json';
            const tokenizerConfigUrl = 'https://huggingface.co/Nayusai/chtbot/raw/main/tokenizer_config.json';

            loadingMessage.textContent = 'Loading tokenizer configuration...';
            // Manually fetch the configuration files to bypass the library's fallback.
            const tokenizerConfigResponse = await fetch(tokenizerConfigUrl);
            if (!tokenizerConfigResponse.ok) throw new Error(`Failed to fetch tokenizer config: ${tokenizerConfigResponse.statusText}`);
            const tokenizerConfig = await tokenizerConfigResponse.json();
            
            const tokenizerResponse = await fetch(tokenizerUrl);
            if (!tokenizerResponse.ok) throw new Error(`Failed to fetch tokenizer: ${tokenizerResponse.statusText}`);
            const tokenizerJson = await tokenizerResponse.json();

            loadingMessage.textContent = 'Creating tokenizer...';
            // Manually create the tokenizer from the fetched configuration.
            tokenizer = new BertTokenizer(tokenizerJson, tokenizerConfig);
            
            // Configure the ONNX runtime
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/';
            ort.env.wasm.numThreads = 1;

            loadingMessage.textContent = 'Loading model...';
            session = await ort.InferenceSession.create(modelUrl, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
                // This flag is kept as a safeguard, but the padding strategy is the primary fix.
                enableMemPattern: false 
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
    
    // --- Autoregressive Generation with Dual Static Padding ---
    async function generateText(prompt) {
        if (!tokenizer || !session) {
            throw new Error("Tokenizer or session not initialized.");
        }

        // Define the padding token ID. For BERT-style models, this is almost always 0.
        const padTokenId = 0;

        // 1. Encode and PAD the user's prompt for the 'src' tensor.
        const srcIds = tokenizer.encode(prompt).filter(id => typeof id === 'number');
        if (srcIds.length > maxSourceLength) {
            srcIds.splice(maxSourceLength); // Truncate if too long
        }
        const paddedSrcIds = new Array(maxSourceLength).fill(padTokenId);
        paddedSrcIds.splice(0, srcIds.length, ...srcIds);
        const srcTensor = new ort.Tensor('int64', BigInt64Array.from(paddedSrcIds.map(BigInt)), [1, maxSourceLength]);


        // 2. Initialize the target sequence with the BOS token.
        const bosTokenId = tokenizer.cls_token_id || 0;
        let tgtIds = [bosTokenId];
        
        // 3. Autoregressive generation loop.
        for (let i = 0; i < maxGenerationLength; i++) {
            // --- PADDING LOGIC for TGT ---
            const paddedTgtIds = new Array(maxGenerationLength).fill(padTokenId);
            paddedTgtIds.splice(0, tgtIds.length, ...tgtIds);
            const tgtTensor = new ort.Tensor('int64', BigInt64Array.from(paddedTgtIds.map(BigInt)), [1, maxGenerationLength]);

            const feeds = {
                src: srcTensor,
                tgt: tgtTensor
            };
            
            const results = await session.run(feeds);
            const logits = results.output; 

            // Get the logits for the *next token to be generated*.
            const vocabSize = logits.dims[2];
            const nextTokenLogits = logits.data.slice(i * vocabSize, (i + 1) * vocabSize);

            // Find the next token with the highest probability.
            let maxLogit = -Infinity;
            let nextTokenId = 0;
            for (let j = 0; j < vocabSize; j++) {
                if (nextTokenLogits[j] > maxLogit) {
                    maxLogit = nextTokenLogits[j];
                    nextTokenId = j;
                }
            }
            
            const eosTokenId = tokenizer.sep_token_id || 1;
            // Stop if we generate the EOS token or a PAD token
            if (nextTokenId === eosTokenId || nextTokenId === padTokenId) {
                break;
            }

            // Add the real generated token to our list.
            tgtIds.push(nextTokenId);
        }

        // 4. Decode the generated tokens, skipping the initial BOS token.
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
