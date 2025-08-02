/*
================================================================================
FINAL SCRIPT (FOR ONNX-ONLY WITH COMPATIBLE CONFIG)
================================================================================
This script is designed to work with your existing .onnx file by pairing it
with a generic text-generation (T5-style) config.json and tokenizer.json
in your Hugging Face repository.

This will solve the loading and execution errors.
*/
document.addEventListener('DOMContentLoaded', () => {
    // --- CHATBOT LOGIC ---
    const chatDisplay = document.getElementById('chat-display');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const clearChatBtn = document.getElementById('clear-chat-btn');

    // --- ONNX MODEL SETUP ---
    let modelPipeline = null;
    let isModelReady = false;

    // This context is now used to build a prompt for the text-generation model
    // You can remove or simplify this if your model doesn't need external context.
    const context = `
        Paggy Xi Xang is a cutting-edge chatbot designed by a talented team.
        The backend was developed by Alex Martinez and Samira Khan, who focused on API design and database architecture.
        The frontend was engineered by Jordan Lee, who created the user interface and experience.
        The chatbot does not store any personal data or use cookies, ensuring user privacy.
        For legal information, users can consult the Terms of Service and Privacy Policy available in the footer.
        The project is managed under the entity Paggy Inc. and was last updated in July 2025.
    `;

    // --- Function to initialize the model on page load ---
    async function initializeModel() {
        const loadingMessage = document.createElement('div');
        loadingMessage.classList.add('loading-indicator', 'message-bubble');
        loadingMessage.textContent = 'Initializing AI model...';
        chatDisplay.appendChild(loadingMessage);
        userInput.disabled = true; // Disable input while loading

        try {
            const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1');
            const modelRepoId = 'Nayusai/chtbot';

            loadingMessage.textContent = 'Loading AI model from Hugging Face...';
            
            // Use the 'text2text-generation' pipeline, which will work with your new T5-style config
            modelPipeline = await pipeline('text2text-generation', modelRepoId, {
                quantized: false, 
                progress_callback: (progress) => {
                    loadingMessage.textContent = `Loading: ${progress.file} (${Math.round(progress.progress)}%)`;
                }
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
            loadingMessage.textContent = 'Error: Could not load model. Check console & file paths.';
            loadingMessage.style.backgroundColor = '#f87171'; // Red color for error
            loadingMessage.style.color = '#7f1d1d';
        }
    }

    // --- Main function to run the model ---
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
            // Create a prompt for the model. You can simplify this if no context is needed.
            const prompt = `Based on the following context, answer the question.\n\nContext: "${context.trim()}"\n\nQuestion: "${userQuestion}"\n\nAnswer:`;

            // Call the text-generation pipeline with the prompt.
            const result = await modelPipeline(prompt, {
                max_new_tokens: 150, // You can adjust the max length of the response
                skip_special_tokens: true,
            });
            
            // The output is an array; the text is in the 'generated_text' property.
            const chatbotReply = result[0]?.generated_text.trim() || "Sorry, I couldn't generate a response.";
            
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

    // --- HELPER & UI FUNCTIONS ---

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
            navigator.clipboard.writeText(messageElement.textContent).then(() => {
                copyBtn.innerHTML = '<i class="fas fa-check"></i>';
                setTimeout(() => {
                    copyBtn.innerHTML = '<i class="far fa-copy"></i>';
                }, 1500);
            }).catch(err => {
                console.error('Failed to copy text: ', err);
            });
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
            // Re-add the initial message after clearing if you want
            if (msg.classList.contains('initial-message')) {
                msg.textContent = 'Chat cleared. Ask me something!';
            } else {
                 msg.remove();
            }
        });
        if (!chatDisplay.querySelector('.initial-message')) {
            const readyMessage = document.createElement('div');
            readyMessage.classList.add('initial-message');
            readyMessage.textContent = 'Chat cleared. Ask me something!';
            chatDisplay.appendChild(readyMessage);
        }
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
        openTermsLink.addEventListener('click', (e) => {
            e.preventDefault();
            termsModal.classList.add('visible');
        });
        closeTermsBtn.addEventListener('click', () => {
            termsModal.classList.remove('visible');
        });
        window.addEventListener('click', (e) => {
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
        window.addEventListener('click', (e) => {
            if (e.target === privacyModal) {
                privacyModal.classList.remove('visible');
            }
        });
    }
    
    // --- Initialize the model when the page loads ---
    initializeModel();
});
