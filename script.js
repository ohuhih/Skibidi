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

            loadingMessage.textContent = 'Loading tokenizer...';
            const tokenizerConfig = await (await fetch(tokenizerConfigUrl)).json();
            const tokenizerJson = await (await fetch(tokenizerUrl)).json();
            tokenizer = new BertTokenizer(tokenizerJson, tokenizerConfig);

            ort.env.logLevel = 'warning';
            ort.env.wasm.numThreads = 1;
            ort.env.wasm.simd = false;

            loadingMessage.textContent = 'Loading model...';
            session = await ort.InferenceSession.create(modelUrl, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
                enableMemPattern: false
            });

            chatDisplay.removeChild(loadingMessage);
            appendMessage('Model loaded. You can start chatting!', 'chatbot');
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
        if (!tokenizer || !session) throw new Error("Tokenizer or session not initialized.");

        const padTokenId = 0;
        const srcIds = tokenizer.encode(prompt).filter(id => typeof id === 'number');
        const paddedSrc = new Array(maxSourceLength).fill(padTokenId);
        paddedSrc.splice(0, srcIds.length, ...srcIds);
        const srcTensor = new ort.Tensor('int64', BigInt64Array.from(paddedSrc.map(BigInt)), [1, maxSourceLength]);

        const bosTokenId = tokenizer.cls_token_id || 0;
        const eosTokenId = tokenizer.sep_token_id || 1;
        let tgtIds = [bosTokenId];

        for (let i = 0; i < maxGenerationLength; i++) {
            const paddedTgt = new Array(maxGenerationLength).fill(padTokenId);
            paddedTgt.splice(0, tgtIds.length, ...tgtIds);
            const tgtTensor = new ort.Tensor('int64', BigInt64Array.from(paddedTgt.map(BigInt)), [1, maxGenerationLength]);

            const results = await session.run({ src: srcTensor, tgt: tgtTensor });
            const logits = results.output;
            const vocabSize = logits.dims[2];
            const nextLogits = logits.data.slice(i * vocabSize, (i + 1) * vocabSize);

            let maxLogit = -Infinity;
            let nextId = 0;
            for (let j = 0; j < vocabSize; j++) {
                if (nextLogits[j] > maxLogit) {
                    maxLogit = nextLogits[j];
                    nextId = j;
                }
            }

            if (nextId === eosTokenId || nextId === padTokenId) break;
            tgtIds.push(nextId);
        }

        return tokenizer.decode(tgtIds.slice(1));
    }

    async function getChatbotResponse(userQuestion) {
        if (!isModelReady) {
            appendMessage("The AI model is still initializing.", 'chatbot');
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
            console.error('Error running model:', error);
            if (chatDisplay.contains(loadingMessage)) chatDisplay.removeChild(loadingMessage);
            appendMessage('Error: Could not run the model.', 'chatbot');
        }
    }

    function appendMessage(text, sender) {
        const wrapper = document.createElement('div');
        wrapper.classList.add('message-wrapper', `${sender}-wrapper`);

        const bubble = document.createElement('div');
        bubble.classList.add('message-bubble', `${sender}-message`);
        bubble.textContent = text;

        const timestamp = document.createElement('span');
        timestamp.classList.add('timestamp');
        timestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        const copyBtn = document.createElement('button');
        copyBtn.classList.add('copy-btn');
        copyBtn.innerHTML = '<i class="far fa-copy"></i>';
        copyBtn.title = 'Copy message';
        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(bubble.textContent);
            copyBtn.innerHTML = '<i class="fas fa-check"></i>';
            setTimeout(() => copyBtn.innerHTML = '<i class="far fa-copy"></i>', 1500);
        });

        wrapper.appendChild(bubble);
        wrapper.appendChild(copyBtn);
        wrapper.appendChild(timestamp);
        chatDisplay.appendChild(wrapper);
        chatDisplay.scrollTop = chatDisplay.scrollHeight;
    }

    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;
        appendMessage(message, 'user');
        userInput.value = '';
        getChatbotResponse(message);
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', e => {
        if (e.key === 'Enter') sendMessage();
    });

    clearChatBtn.addEventListener('click', () => {
        chatDisplay.querySelectorAll('.message-wrapper, .loading-indicator, .initial-message')
            .forEach(msg => msg.remove());
    });

    initializeModel();
});
