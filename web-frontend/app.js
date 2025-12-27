// ONNX Runtime Autocomplete Model
class AutocompleteModel {
    constructor() {
        this.session = null;
        this.word2id = null;
        this.id2word = null;
        this.maxSeqLen = 30;
        this.padTokenId = 0;
        this.isReady = false;
    }

    async initialize() {
        try {
            // Load vocabulary
            await this.loadVocabulary();

            // Load ONNX model
            await this.loadModel();

            this.isReady = true;
            return true;
        } catch (error) {
            console.error('Initialization error:', error);
            throw error;
        }
    }

    async loadVocabulary() {
        try {
            const response = await fetch('vocabulary.json');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            this.word2id = data.word2id;
            this.id2word = data.id2word;

            console.log(`Vocabulary loaded: ${Object.keys(this.word2id).length} words`);
        } catch (error) {
            throw new Error(`Failed to load vocabulary: ${error.message}`);
        }
    }

    async loadModel() {
        try {
            // Set WebAssembly paths for ONNX Runtime
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/';

            const response = await fetch('model.onnx');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const arrayBuffer = await response.arrayBuffer();
            this.session = await ort.InferenceSession.create(arrayBuffer, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
            });

            console.log('ONNX model loaded successfully');
        } catch (error) {
            throw new Error(`Failed to load ONNX model: ${error.message}`);
        }
    }

    tokenize(text) {
        // Convert text to lowercase and split by spaces/punctuation
        const words = text
            .toLowerCase()
            .replace(/[^\w\s']/g, ' ')
            .split(/\s+/)
            .filter(word => word.length > 0);

        return words;
    }

    wordsToIds(words) {
        return words.map(word => {
            if (word in this.word2id) {
                return this.word2id[word];
            } else {
                // Use UNK token for unknown words
                return this.word2id['<unk>'] || 0;
            }
        });
    }

    padSequence(ids) {
        const padded = new Array(this.maxSeqLen).fill(this.padTokenId);
        const start = Math.max(0, this.maxSeqLen - ids.length);
        const toCopy = ids.slice(Math.max(0, ids.length - this.maxSeqLen));
        toCopy.forEach((id, i) => {
            padded[start + i] = id;
        });
        return padded;
    }

    async predict(inputText) {
        if (!this.isReady) {
            throw new Error('Model not initialized');
        }

        if (!inputText || inputText.trim().length === 0) {
            return [];
        }

        try {
            // Tokenize input
            const words = this.tokenize(inputText);
            const ids = this.wordsToIds(words);
            const paddedIds = this.padSequence(ids);

            // Create input tensor [1, maxSeqLen]
            const inputData = new Int64Array(paddedIds);
            const input = new ort.Tensor('int64', inputData, [1, this.maxSeqLen]);

            // Run inference
            const results = await this.session.run({ input_ids: input });
            const logits = results.logits.data;

            // Get top 5 predictions
            const predictions = this.getTopPredictions(logits, 5);

            return predictions;
        } catch (error) {
            console.error('Prediction error:', error);
            throw error;
        }
    }

    getTopPredictions(logits, topK = 5) {
        const predictions = [];

        // Convert logits to array and get scores
        for (let i = 0; i < logits.length; i++) {
            predictions.push({
                id: i,
                score: logits[i],
                word: this.id2word[i] || `<unk_${i}>`,
            });
        }

        // Sort by score descending
        predictions.sort((a, b) => b.score - a.score);

        // Get softmax scores for display
        const topPredictions = predictions.slice(0, topK);
        const maxScore = Math.max(...topPredictions.map(p => p.score));
        const minScore = Math.min(...topPredictions.map(p => p.score));
        const scoreRange = maxScore - minScore || 1;

        return topPredictions.map((pred, rank) => ({
            rank: rank + 1,
            word: pred.word,
            score: pred.score,
            confidence: ((pred.score - minScore) / scoreRange * 100).toFixed(1),
        }));
    }
}

// UI Controller
class AutocompleteUI {
    constructor() {
        this.model = new AutocompleteModel();
        this.userInput = document.getElementById('userInput');
        this.predictionsList = document.getElementById('predictionsList');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
        this.errorMessage = document.getElementById('errorMessage');
        this.debounceTimer = null;
    }

    async initialize() {
        try {
            await this.model.initialize();
            this.setStatus('ready', 'Model ready - Start typing!');
            this.userInput.disabled = false;
            this.userInput.focus();
        } catch (error) {
            this.setStatus('error', `Error: ${error.message}`);
            this.showError(error.message);
            console.error(error);
        }
    }

    setStatus(state, message) {
        this.statusIndicator.className = `status-indicator ${state}`;
        this.statusText.textContent = message;
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorMessage.classList.add('show');
        setTimeout(() => {
            this.errorMessage.classList.remove('show');
        }, 5000);
    }

    async handleInput() {
        // Clear previous debounce
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }

        // Debounce predictions (300ms)
        this.debounceTimer = setTimeout(() => {
            this.updatePredictions();
        }, 300);
    }

    async updatePredictions() {
        const inputText = this.userInput.value;

        if (inputText.trim().length === 0) {
            this.predictionsList.innerHTML = '<div class="empty-state">Enter text to see predictions</div>';
            return;
        }

        try {
            this.setStatus('loading', 'Generating predictions...');
            const predictions = await this.model.predict(inputText);

            if (predictions.length === 0) {
                this.predictionsList.innerHTML = '<div class="empty-state">No predictions available</div>';
            } else {
                this.renderPredictions(predictions);
            }

            this.setStatus('ready', 'Ready');
        } catch (error) {
            this.setStatus('error', 'Prediction error');
            this.showError(`Prediction failed: ${error.message}`);
            console.error(error);
        }
    }

    renderPredictions(predictions) {
        this.predictionsList.innerHTML = predictions
            .map(
                pred => `
            <div class="prediction-item" onclick="ui.insertWord('${pred.word}')">
                <div class="prediction-rank">#${pred.rank}</div>
                <div class="prediction-word">${this.escapeHtml(pred.word)}</div>
                <div class="prediction-score">${pred.confidence}% confidence</div>
            </div>
        `
            )
            .join('');
    }

    insertWord(word) {
        // Remove special tokens
        if (word.startsWith('<') && word.endsWith('>')) {
            return;
        }

        const currentText = this.userInput.value.trim();
        this.userInput.value = currentText + ' ' + word + ' ';
        this.userInput.focus();
        this.updatePredictions();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global UI instance
let ui;

// Initialize on page load
window.addEventListener('load', async () => {
    ui = new AutocompleteUI();
    await ui.initialize();

    // Setup input listeners
    document.getElementById('userInput').addEventListener('input', () => {
        ui.handleInput();
    });

    // Allow Enter to add a space and trigger predictions
    document.getElementById('userInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            ui.userInput.value += ' ';
            ui.updatePredictions();
        }
    });
});
