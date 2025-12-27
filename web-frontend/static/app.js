class AutocompleteUI {
    constructor() {
        this.userInput = document.getElementById('userInput');
        this.predictionsList = document.getElementById('predictionsList');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
        this.errorMessage = document.getElementById('errorMessage');
        this.debounceTimer = null;
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
        if (this.debounceTimer) clearTimeout(this.debounceTimer);

        this.debounceTimer = setTimeout(() => {
            this.fetchPredictions();
        }, 300); // 300ms delay
    }

    async fetchPredictions() {
        const inputText = this.userInput.value;

        if (inputText.trim().length === 0) {
            this.predictionsList.innerHTML = '<div class="empty-state">Enter text to see predictions</div>';
            this.setStatus('ready', 'Connected to Server');
            return;
        }

        try {
            this.setStatus('loading', 'Fetching from server...');
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: inputText })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            if (data.predictions.length === 0) {
                this.predictionsList.innerHTML = '<div class="empty-state">No predictions available</div>';
            } else {
                this.renderPredictions(data.predictions);
            }

            this.setStatus('ready', 'Ready');
        } catch (error) {
            this.setStatus('error', 'API Error');
            this.showError(`Failed: ${error.message}`);
            console.error(error);
        }
    }

    renderPredictions(predictions) {
        this.predictionsList.innerHTML = predictions.map(pred => `
            <div class="prediction-item" onclick="ui.insertWord('${pred.word}')">
                <div class="prediction-rank">#${pred.rank}</div>
                <div class="prediction-word">${this.escapeHtml(pred.word)}</div>
                <div class="prediction-score">${pred.confidence}% confidence</div>
            </div>
        `).join('');
    }

    insertWord(word) {
        // Remove special tokens logic
        if (word.startsWith('<') && word.endsWith('>')) return;

        const currentText = this.userInput.value.trim();
        this.userInput.value = currentText + ' ' + word + ' ';
        this.userInput.focus();
        this.fetchPredictions();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize
const ui = new AutocompleteUI();

// Event Listeners
document.getElementById('userInput').addEventListener('input', () => ui.handleInput());

document.getElementById('userInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        ui.userInput.value += ' ';
        ui.fetchPredictions();
    }
});
