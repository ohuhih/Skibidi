/*
================================================================================
FINAL WORKING VERSION
================================================================================
This version is designed to work by loading all model files from your public
Hugging Face Hub repository. This is the standard and most reliable method,
and it will resolve the CORS and file-not-found errors.

**Your Action Required:**
- You have already completed the required action by uploading your files to
  the 'Nayusai/chtbot' repository on Hugging Face. This script will now
  load everything from there.
================================================================================
*/
document.addEventListener('DOMContentLoaded', () => {
    // --- UI ELEMENTS ---
    const chatDisplay = document.getElementById('chat-display');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const clearChatBtn = document.getElementById('clear-chat-btn');

    // --- MODEL AND TOKENIZER SETUP ---
    let generator = null; // Changed to a more generic name
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
            // Using a specific, known-stable version of the all-in-one library
            const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1');
            
            // =================================================================
            // === CORRECTED: This now points to your Hugging Face repo ID ===
            // =================================================================
            // The library will automatically find all necessary files in this repo.
            const modelRepoId = 'Nayusai/chtbot';

            loadingMessage.textContent = 'Loading AI model from Hugging Face...';
            // This is the simplest and most reliable way to load the model.
            // Using 'text2text-generation' as it's more suitable for a src/tgt model.
            generator = await pipeline('text2text-generation', modelRepoId, {
                quantized: true,
                progress_callback: (progress) => {
                    loadingMessage.textContent = `Loading: ${progress.file} (${Math.round(progress.progress)}%)`;
                }
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
            loadingMessage.textContent = 'Error: Could not load model. Check console & file paths.';
            loadingMessage.style.color = '#f87171'; // Red color for error
        }
    }
    
    // --- This is the main function that now runs the ONNX model ---
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
            // The pipeline now handles the complex generation loop internally.
            const result = await generator(userQuestion, {
                max_length: maxGenerationLength,
                no_repeat_ngram_size: 3,
            });
            const reply = result[0].generated_text;

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
