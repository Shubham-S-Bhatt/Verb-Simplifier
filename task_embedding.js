// #####################################################################################
// ######################### Using Word Embedding / word2vec ###########################
// #####################################################################################

document.addEventListener('DOMContentLoaded', () => {

    let embeddings = {};
    const embeddingSize = 50; // Assuming each embedding is a 50-dimensional vector
    let model;

    const trainingData = [
        { input: "obliterates", output: "destroys" },
        { input: "annihilates", output: "destroys" },
        { input: "promulgates", output: "issues" },
        { input: "delineates", output: "describes" },
        { input: "mitigates", output: "reduces" },
        { input: "revolutionizes", output: "changes" },
        { input: "adjudicates", output: "judges" },
        { input: "elucidates", output: "explains" },
        { input: "endeavors", output: "tries" },
        { input: "organizes", output: "arranges" },
        { input: "explains", output: "clarifies" },
        { input: "tries", output: "attempts" }
    ];

    async function loadEmbeddings() {
        console.log("Loading embeddings...");
        try {
            const response = await fetch('embeddings.json');
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
            model.add(tf.layers.dense({ units: 32, inputShape: [embeddingSize], activation: 'relu' }));
            model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
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

            const trainingMessage = document.getElementById('training-message-embedding');
            const sentenceInput = document.getElementById('sentence-input-embedding');
            const simplifyButton = document.getElementById('simplify-button-embedding');

            if (trainingMessage && sentenceInput && simplifyButton) {
                trainingMessage.style.display = 'block';
                console.log('Model training started...');
                await model.fit(xs, ys, {
                    epochs: 2000,
                    callbacks: tf.callbacks.earlyStopping({ monitor: 'loss' })
                });


                console.log('Model training complete!');
                
                // Hide the training message and enable input and button
                trainingMessage.style.display = 'none'; 
                sentenceInput.disabled = false; 
                simplifyButton.disabled = false;
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
            console.warn(`Embedding not found for word: ${word}`);
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

        return doc.text();
    }

    function simplifyInputSentenceEmbedded() {
        console.log("Simplify input sentence function called.");
        const sentence = document.getElementById('sentence-input-embedding').value;
        document.getElementById('original-sentence-embedding').innerText = sentence;
        console.log("Original sentence:", sentence);
        const simplifiedSentence = simplifySentence(sentence);
        console.log("Simplified sentence:", simplifiedSentence);
        document.getElementById('simplified-sentence-embedding').innerText = simplifiedSentence;
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

    // Exposing the function to the global scope so it can be called by the button
    window.simplifyInputSentenceEmbedded = simplifyInputSentenceEmbedded;
});
