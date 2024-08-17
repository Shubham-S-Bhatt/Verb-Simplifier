document.addEventListener('DOMContentLoaded', () => {

    let embeddings = {};
    const embeddingSize = 50; // Assuming each embedding is a 50-dimensional vector
    let model; // Declare the model variable globally

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
            model = tf.sequential();
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

    function predictVerb(complexVerb) {
        const inputTensor = tf.tensor2d([encodeWord(complexVerb)]);
        const outputTensor = model.predict(inputTensor);
        const predictedWordVector = outputTensor.dataSync();

        // Find the word closest to the predicted embedding
        let bestMatch = '';
        let smallestDistance = Infinity;

        Object.keys(embeddings).forEach(word => {
            const distance = cosineSimilarity(predictedWordVector, embeddings[word]);
            if (distance < smallestDistance) {
                smallestDistance = distance;
                bestMatch = word;
            }
        });

        return bestMatch;
    }

    function cosineSimilarity(a, b) {
        const dotProduct = a.reduce((sum, aValue, i) => sum + aValue * b[i], 0);
        const normA = Math.sqrt(a.reduce((sum, aValue) => sum + aValue * aValue, 0));
        const normB = Math.sqrt(b.reduce((sum, bValue) => sum + bValue * bValue, 0));
        return 1 - (dotProduct / (normA * normB));
    }

    function simplifySentence(sentence) {
        let doc = nlp(sentence);
        let simplified = false;

        doc.verbs().forEach(verb => {
            let text = verb.text();
            if (embeddings[text]) {
                let simplifiedVerb = predictVerb(text);
                doc.replace(text, simplifiedVerb);
                simplified = true;
            }
        });

        if (!simplified) {
            console.warn("No simplification available for some verbs in the sentence.");
        }

        return doc.text();
    }

    function simplifyInputSentence() {
        console.log("Simplify input sentence function called.");
        const sentence = document.getElementById('sentence-input').value;
        document.getElementById('original-sentence').innerText = sentence;
        const simplifiedSentence = simplifySentence(sentence);
        document.getElementById('simplified-sentence').innerText = simplifiedSentence;
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

    window.simplifyInputSentence = simplifyInputSentence;
});
