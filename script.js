document.addEventListener('DOMContentLoaded', () => {
    const chatDisplay = document.getElementById('chat-display');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const clearChatBtn = document.getElementById('clear-chat-btn');

    let session = null;
    let tokenizer = null;
    let isModelReady = false;
    const maxGenerationLength = 50;
    const maxSourceLength = 128;

    async function initializeModel() {
        const loadingMessage = document.createElement('div');
        loadingMessage.classList.add('initial-message');
        loadingMessage.textContent = 'Initializing AI model...';
        chatDisplay.appendChild(loadingMessage);
        userInput.disabled = true;
        sendButton.disabled = true;

        try {
            const { BertTokenizer } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1');

            const modelUrl = 'https://huggingface.co/Nayusai/chtbot/resolve/main/onnx/model.onnx';
            const tokenizerUrl = 'https://huggingface.co/Nayusai/chtbot/raw/main/tokenizer.json';
            const tokenizerConfigUrl = 'https://huggingface.co/Nayusai/chtbot/raw/main/tokenizer_config.json';

            loadingMessage.textContent = 'Loading tokenizer configuration...';
            const tokenizerConfigResponse = await fetch(tokenizerConfigUrl);
            const tokenizerConfig = await tokenizerConfigResponse.json();
            const tokenizerResponse = await fetch(tokenizerUrl);
            const tokenizerJson = await tokenizerResponse.json();

            loadingMessage.textContent = 'Creating tokenizer...';
            tokenizer = new BertTokenizer(tokenizerJson, tokenizerConfig);

            ort.env.logLevel = 'verbose';
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.15.0/dist/';
            ort.env.wasm.numThreads = 1;
            ort.env.wasm.simd = false;

            loadingMessage.textContent = 'Loading model...';
            session = await ort.InferenceSession.create(modelUrl, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
                enableMemPattern: false
            });

            // 🔍 Print input/output names
            console.log('Model input names:', session.inputNames);
            console.log('Model output names:', session.outputNames);

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
            loadingMessage.style.color = '#f87171';
        }
    }

    async function generateText(prompt) {
        if (!tokenizer || !session) {
            throw new Error("Tokenizer or session not initialized.");
        }

        const padTokenId = 0;
        const srcIds = tokenizer.encode(prompt).filter(id => typeof id === 'number');
        if (srcIds.length > maxSourceLength) {
            srcIds.splice(maxSourceLength);
        }
        const paddedSrcIds = new Array(maxSourceLength).fill(padTokenId);
        paddedSrcIds.splice(0, srcIds.length, ...srcIds);
        const srcTensor = new ort.Tensor('int64', BigInt64Array.from(paddedSrcIds.map(BigInt)), [1, maxSourceLength]);

        const bosTokenId = tokenizer.cls_token_id || 0;
        let tgtIds = [bosTokenId];

        for (let i = 0; i < maxGenerationLength; i++) {
            const paddedTgtIds = new Array(maxGenerationLength).fill(padTokenId);
            paddedTgtIds.splice(0, tgtIds.length, ...tgtIds);
            const tgtTensor = new ort.Tensor('int64', BigInt64Array.from(paddedTgtIds.map(BigInt)), [1, maxGenerationLength]);

            const feeds = {
                src: srcTensor,
                tgt: tgtTensor
            };

            let results;
            try {
                results = await session.run(feeds);
            } catch (err) {
                console.error('Error running model:', err);
                throw err;
            }

            const logits = results.output;

            const vocabSize = logits.dims[2];
            const nextTokenLogits = logits.data.slice(i * vocabSize, (i + 1) * vocabSize);

            let maxLogit = -Infinity;
            let nextTokenId = 0;
            for (let j = 0; j < vocabSize; j++) {
                if (nextTokenLogits[j] > maxLogit) {
                    maxLogit = nextTokenLogits[j];
                    nextTokenId = j;
                }
            }

            const eosTokenId = tokenizer.sep_token_id || 1;
            if (nextTokenId === eosTokenId || nextTokenId === padTokenId) {
                break;
            }

            tgtIds.push(nextTokenId);
        }

        return tokenizer.decode(tgtIds.slice(1));
    }

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

    initializeModel();
});
