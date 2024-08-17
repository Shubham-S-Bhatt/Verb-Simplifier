document.addEventListener('DOMContentLoaded', () => {
    let embeddings = {};
    const embeddingSize = 50; // Assuming each embedding is a 50-dimensional vector

    // Load the embeddings
    fetch('embeddings.json')
        .then(response => response.json())
        .then(data => {
            embeddings = data;
            trainModel();
        })
        .catch(error => console.error('Error loading embeddings:', error));

    function getEmbedding(word) {
        return embeddings[word] || new Array(embeddingSize).fill(0); // Return a zero vector if the word is not found
    }

    const trainingData = [
            { input: "obliterates", output: "destroys" },
            { input: "annihilates", output: "destroys" },
            { input: "eradicates", output: "removes" },
            { input: "exterminates", output: "kills" },
            { input: "perplexes", output: "confuses" },
            { input: "elevates", output: "raises" },
            { input: "diminishes", output: "reduces" },
            { input: "devastates", output: "ruins" },
            { input: "decimates", output: "reduces" },
            { input: "demolishes", output: "destroys" },
            { input: "disseminates", output: "spreads" },
            { input: "facilitates", output: "helps" },
            { input: "fabricates", output: "makes" },
            { input: "illuminates", output: "lights" },
            { input: "incinerates", output: "burns" },
            { input: "instigates", output: "starts" },
            { input: "intensifies", output: "increases" },
            { input: "magnifies", output: "enlarges" },
            { input: "mitigates", output: "lessens" },
            { input: "mobilizes", output: "activates" },
            { input: "obliterates", output: "destroys" },
            { input: "orchestrates", output: "organizes" },
            { input: "precipitates", output: "causes" },
            { input: "relinquishes", output: "gives up" },
            { input: "repudiates", output: "denies" },
            { input: "resurrects", output: "revives" },
            { input: "reiterates", output: "repeats" },
            { input: "stipulates", output: "specifies" },
            { input: "synthesizes", output: "combines" },
            { input: "transforms", output: "changes" },
            { input: "transmits", output: "sends" },
            { input: "uproots", output: "removes" },
            { input: "vacillates", output: "wavers" },
            { input: "vaporizes", output: "evaporates" },
            { input: "vindicates", output: "justifies" },
            { input: "alleviates", output: "reduces" },
            { input: "articulates", output: "expresses" },
            { input: "aspires", output: "hopes" },
            { input: "calibrates", output: "adjusts" },
            { input: "collaborates", output: "works together" },
            { input: "contaminates", output: "pollutes" },
            { input: "delineates", output: "describes" },
            { input: "depreciates", output: "devalues" },
            { input: "disintegrates", output: "breaks apart" },
            { input: "encompasses", output: "includes" },
            { input: "exaggerates", output: "overstates" },
            { input: "exonerates", output: "clears" },
            { input: "extrapolates", output: "estimates" },
            { input: "fascinates", output: "interests" },
            { input: "formulates", output: "develops" },
            { input: "illuminates", output: "lights" },
            { input: "implicates", output: "involves" },
            { input: "inaugurates", output: "begins" },
            { input: "infuriates", output: "angers" },
            { input: "interrogates", output: "questions" },
            { input: "legitimizes", output: "justifies" },
            { input: "manipulates", output: "controls" },
            { input: "obliterates", output: "destroys" },
            { input: "originates", output: "begins" },
            { input: "permeates", output: "spreads through" },
            { input: "procrastinates", output: "delays" },
            { input: "propagates", output: "spreads" },
            { input: "relinquishes", output: "gives up" },
            { input: "revitalizes", output: "refreshes" },
            { input: "subjugates", output: "dominates" },
            { input: "sublimates", output: "transforms" },
            { input: "submerges", output: "immerses" },
            { input: "validates", output: "confirms" },
            { input: "venerates", output: "respects" },
            { input: "vivifies", output: "enlivens" },
    ];

    function encodeWord(word) {
        return getEmbedding(word);
    }

    const encodedData = trainingData.map(({ input, output }) => ({
        input: encodeWord(input),
        output: encodeWord(output)
    }));

    // Define the model
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 128, inputShape: [embeddingSize], activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: embeddingSize, activation: 'softmax' }));

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    async function trainModel() {
        const xs = tf.tensor2d(encodedData.map(({ input }) => input));
        const ys = tf.tensor2d(encodedData.map(({ output }) => output));

        document.getElementById('training-message').style.display = 'block'; // Show the training message
        await model.fit(xs, ys, {
            epochs: 100,
            callbacks: tf.callbacks.earlyStopping({ monitor: 'loss' })
        });
        document.getElementById('training-message').style.display = 'none'; // Hide the training message
        document.getElementById('sentence-input').disabled = false; // Enable input
        document.getElementById('simplify-button').disabled = false; // Enable button
    }

    function predictVerb(complexVerb) {
        const inputTensor = tf.tensor2d([getEmbedding(complexVerb)]);
        const outputTensor = model.predict(inputTensor);
        const outputIndex = outputTensor.argMax(1).dataSync()[0];
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

    function simplifyInputSentenceEmbedding() {
        const sentence = document.getElementById('sentence-input').value;
        document.getElementById('original-sentence').innerText = sentence;
        const simplifiedSentence = simplifySentence(sentence);
        document.getElementById('simplified-sentence').innerText = simplifiedSentence;
    }

    // Expose the function to the global scope so it can be called by the button
    window.simplifyInputSentenceEmbedding = simplifyInputSentenceEmbedding;
});
