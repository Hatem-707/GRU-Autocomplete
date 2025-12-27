/**
 * Configuration for the Autocomplete Model
 * Adjust these settings to customize the app behavior
 */

const CONFIG = {
    // Model Configuration
    MODEL: {
        file: 'model.onnx',
        vocabFile: 'vocabulary.json',
        maxSequenceLength: 30,
        padTokenId: 0,
    },

    // Prediction Settings
    PREDICTIONS: {
        topK: 5,  // Number of predictions to show
        debounceMs: 300,  // Wait time before computing predictions
    },

    // UI Settings
    UI: {
        theme: 'dark',  // 'light' or 'dark'
        animationEnabled: true,
        showConfidence: true,
        showRank: true,
    },

    // Performance Settings
    PERFORMANCE: {
        executionProvider: 'wasm',  // WebAssembly provider
        graphOptimization: 'all',  // Optimization level
    },

    // Display Settings
    DISPLAY: {
        placeholderText: "Type something... e.g., 'i want to'",
        emptyStateText: 'Enter text to see predictions',
        loadingText: 'Generating predictions...',
        errorPrefix: 'Error: ',
    },

    // Advanced
    DEBUG: false,  // Enable console logging
};

export default CONFIG;
