<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ONNX Generative Chatbot</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 600px; margin: 2em auto; }
  #chat-display { border: 1px solid #ccc; padding: 1em; height: 400px; overflow-y: auto; margin-bottom: 1em; }
  .message-wrapper { margin-bottom: 1em; }
  .user-wrapper .message-bubble { background-color: #d1e7ff; color: #003366; text-align: right; padding: 0.5em 1em; border-radius: 12px; display: inline-block; }
  .chatbot-wrapper .message-bubble { background-color: #eee; color: #222; padding: 0.5em 1em; border-radius: 12px; display: inline-block; }
  #user-input { width: 80%; padding: 0.5em; }
  #send-button { padding: 0.5em 1em; }
</style>
</head>
<body>

<div id="chat-display"></div>
<input type="text" id="user-input" placeholder="Type your message..." />
<button id="send-button">Send</button>

<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
<script type="module">
// Import Hugging Face Tokenizers (must be served from your server or use unpkg if available)
import { Tokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.13.4/dist/tokenizers.esm.min.js';

const modelUrl = 'https://huggingface.co/Nayusai/chtbot/blob/main/onnx/model.onnx';
const tokenizerUrl = 'https://huggingface.co/Nayusai/chtbot/blob/main/tokenizer.json';

let session = null;
let tokenizer = null;
const maxGenerationLength = 50;
const EOS_TOKEN = ''; // Change if your tokenizer uses different EOS token

// Load ONNX model
async function loadModel() {
  session = await ort.InferenceSession.create(modelUrl);
  console.log('ONNX model loaded');
}

// Load tokenizer
async function loadTokenizer() {
  const response = await fetch(tokenizerUrl);
  const tokenizerJson = await response.json();
  tokenizer = await Tokenizer.fromConfig(tokenizerJson);
  console.log('Tokenizer loaded');
}

// Greedy sampling: pick max logit
function sampleFromLogits(logits) {
  let maxIdx = 0;
  let maxVal = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > maxVal) {
      maxVal = logits[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

// Generate text autoregressively
async function generateText(prompt) {
  // Encode input prompt
  let inputIds = tokenizer.encode(prompt).ids;

  // Start generated tokens with input
  let generatedIds = [...inputIds];

  for (let step = 0; step < maxGenerationLength; step++) {
    const inputTensor = new ort.Tensor('int64', Int32Array.from(generatedIds), [1, generatedIds.length]);

    // Run inference
    const feeds = { input_ids: inputTensor }; // Check your ONNX input name!
    const results = await session.run(feeds);

    // Get logits from output
    const logits = results.logits.data; // Check your ONNX output name!

    // Calculate vocab size
    const vocabSize = logits.length / generatedIds.length;
    const lastTokenLogits = logits.slice((generatedIds.length - 1) * vocabSize);

    // Sample next token
    const nextTokenId = sampleFromLogits(lastTokenLogits);

    if (tokenizer.idToToken(nextTokenId) === EOS_TOKEN) {
      break;
    }

    generatedIds.push(nextTokenId);
  }

  // Decode generated tokens except prompt
  const outputIds = generatedIds.slice(inputIds.length);
  return tokenizer.decode(outputIds);
}

// Chat UI helpers
const chatDisplay = document.getElementById('chat-display');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

function appendMessage(text, sender) {
  const wrapper = document.createElement('div');
  wrapper.className = sender + '-wrapper message-wrapper';

  const bubble = document.createElement('div');
  bubble.className = sender + '-message message-bubble';
  bubble.textContent = text;

  wrapper.appendChild(bubble);
  chatDisplay.appendChild(wrapper);
  chatDisplay.scrollTop = chatDisplay.scrollHeight;
}

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  appendMessage(text, 'user');
  userInput.value = '';
  userInput.disabled = true;
  sendButton.disabled = true;

  try {
    const reply = await generateText(text);
    appendMessage(reply, 'chatbot');
  } catch (e) {
    console.error('Generation error:', e);
    appendMessage('Sorry, something went wrong.', 'chatbot');
  } finally {
    userInput.disabled = false;
    sendButton.disabled = false;
    userInput.focus();
  }
}

sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', e => {
  if (e.key === 'Enter') sendMessage();
});

// Initialization
(async () => {
  appendMessage('Loading model and tokenizer...', 'chatbot');
  await loadTokenizer();
  await loadModel();
  appendMessage('Model ready! You can start chatting.', 'chatbot');
  userInput.disabled = false;
  userInput.focus();
})();
</script>

</body>
</html>
