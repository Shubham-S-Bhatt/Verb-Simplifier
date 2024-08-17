document.addEventListener('DOMContentLoaded', () => {

    let embeddings = {};
    const embeddingSize = 50; // Assuming each embedding is a 50-dimensional vector

    const trainingData = [
        { input: "obliterates", output: "destroys" },
        { input: "annihilates", output: "destroys" },
        // ... more training data
    ];

    async function loadEmbeddings() {
        console.log("Loading embeddings...");
        try {
            const response = await fetch('embeddings.json');
            if (!response.ok) {
                throw new Error('Network response was not ok: ' + response.statusText);
            }
            embeddings = await response.json();
            console.log("Embeddings loaded successfully", embeddings);
        } catch (error) {
            console.error('Error loading embeddings:', error);
            throw error;
        }
    }

    async function trainModel() {
        console.log("Starting model training...");
        try {
            // Define the model
            const model = tf.sequential();
            model.add(tf.layers.dense({ units: 64, inputShape: [embeddingSize], activation: 'relu' }));
            model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
            model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
            model.add(tf.layers.dense({ units: embeddingSize, activation: 'softmax' }));

            model.compile({
                optimizer: tf.train.adam(),
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });

            const xs = tf.tensor2d(trainingData.map(({ input }) => encodeWord(input)));
            const ys = tf.tensor2d(trainingData.map(({ output }) => encodeWord(output)));

            console.log("Input tensors (xs):", xs.shape);
            console.log("Output tensors (ys):", ys.shape);

            const trainingMessage = document.getElementById('training-message');
            const sentenceInput = document.getElementById('sentence-input');
            const simplifyButton = document.getElementById('simplify-button');

            if (trainingMessage && sentenceInput && simplifyButton) {
                trainingMessage.style.display = 'block'; // Show the training message

                await model.fit(xs, ys, {
                    epochs: 100,
                    callbacks: tf.callbacks.earlyStopping({ monitor: 'loss' })
                });

                console.log('Model training complete!');

                trainingMessage.style.display = 'none'; // Hide the training message
                sentenceInput.disabled = false; // Enable input
                simplifyButton.disabled = false; // Enable button
            } else {
                console.error('One or more elements were not found on the page.');
            }
        } catch (error) {
            console.error('Error during model training:', error);
            throw error;
        }
    }

    function encodeWord(word) {
        const embedding = embeddings[word];
        if (!embedding) {
            console.warn(`Embedding not found for word: ${word}. Returning zero vector.`);
        }
        return embedding || new Array(embeddingSize).fill(0); // Return a zero vector if the word is not found
    }

    async function initialize() {
        console.log("Initializing...");
        try {
            await loadEmbeddings();
            await trainModel();
            console.log('Model is ready!');
        } catch (error) {
            console.error('Initialization failed:', error);
        }
    }

    initialize();

    // Placeholder for other functions like predictVerb, simplifySentence, etc.
    function simplifyInputSentence() {
        console.log("Simplify input sentence function called.");
        // Simplification logic here...
    }

    window.simplifyInputSentence = simplifyInputSentence;
});
