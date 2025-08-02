/*
================================================================================
FINAL SCRIPT FOR YOUR 'distilbert' MODEL (FRIDAY, AUG 1, 9:48 PM EDT)
================================================================================
This script is configured for your 'distilbert' model. It fixes all errors by
using the correct 'question-answering' pipeline that matches your config.json.
*/
document.addEventListener('DOMContentLoaded', () => {
    // --- Add this line for verification ---
    console.log("RUNNING THE FINAL Q&A SCRIPT - This message confirms the correct file is loaded.");

    // --- CHATBOT LOGIC ---
    const chatDisplay = document.getElementById('chat-display');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const clearChatBtn = document.getElementById('clear-chat-btn');

    // --- ONNX MODEL SETUP ---
    let questionAnswerer = null;
    let isModelReady = false;

    // This context is the knowledge base for the question-answering model.
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
        userInput.disabled = true;

        try {
            const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1');
            const modelRepoId = 'Nayusai/chtbot';
            loadingMessage.textContent = 'Loading AI model from Hugging Face...';
            
            // --- THE FIX: Use the 'question-answering' pipeline for a DistilBERT model. ---
            questionAnswerer = await pipeline('question-answering', modelRepoId, {
                quantized: false,
                progress_callback: (progress) => {
                    loadingMessage.textContent = `Loading: ${progress.file} (${Math.round(progress.progress)}%)`;
                }
            });

            chatDisplay.removeChild(loadingMessage);
            const readyMessage = document.createElement('div');
            readyMessage.classList.add('initial-message');
            readyMessage.textContent = 'Model loaded. Ask a question about the context.';
            chatDisplay.appendChild(readyMessage);
            
            isModelReady = true;
            userInput.disabled = false;
            userInput.focus();

        } catch (error) {
            console.error('Failed to initialize the AI model:', error);
            loadingMessage.textContent = 'Error: Could not load model. Check console for details.';
            loadingMessage.style.backgroundColor = '#f87171';
            loadingMessage.style.color = '#7f1d1d';
        }
    }

    // --- Main function to run the model ---
    async function getChatbotResponse(userQuestion) {
        if (!isModelReady) {
            appendMessage("The AI model is still initializing, please wait.", 'chatbot');
            return;
        }

        const loadingMessage = document.createElement('div');
        loadingMessage.classList.add('loading-indicator', 'message-bubble');
        loadingMessage.textContent = 'Thinking...';
        chatDisplay.appendChild(loadingMessage);
        chatDisplay.scrollTop = chatDisplay.scrollHeight;

        try {
            // --- THE FIX: Call the pipeline correctly for question-answering ---
            const result = await questionAnswerer(userQuestion, context);
            
            let chatbotReply = "Sorry, I couldn't find an answer in my knowledge base.";
            if (result && result.score > 0.3 && result.answer) {
                chatbotReply = result.answer;
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

    // --- HELPER & UI FUNCTIONS (No changes needed below) ---
    // (Omitted for brevity, your existing functions are fine)
    function appendMessage(text, sender) { /* ... your code ... */ }
    function sendMessage() { /* ... your code ... */ }
    // ... all other event listeners and modal logic ...
    
    // --- Initialize the model when the page loads ---
    initializeModel();
});
