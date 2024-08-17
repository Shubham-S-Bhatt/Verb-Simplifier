document.addEventListener('DOMContentLoaded', () => {
    // Vocabulary and encoding setup
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

    const complexVerbs = trainingData.map(({ input }) => input);
    console.log('Complex verbs:', complexVerbs);

    const vocab = [...new Set(trainingData.flatMap(({ input, output }) => [input, output]))];
    const wordIndex = Object.fromEntries(vocab.map((word, index) => [word, index]));
    console.log('Vocabulary:', vocab);

    function encodeWord(word) {
        const encoded = new Array(vocab.length).fill(0);
        encoded[wordIndex[word]] = 1;
        return encoded;
    }

    const encodedData = trainingData.map(({ input, output }) => ({
        input: encodeWord(input),
        output: encodeWord(output)
    }));
    console.log('Encoded data:', encodedData);

    // Define the model
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 64, inputShape: [vocab.length], activation: 'relu' }));
    model.add(tf.layers.dense({ units: 128, inputShape: [vocab.length], activation: 'relu' }));
    model.add(tf.layers.dense({ units: 128, inputShape: [vocab.length], activation: 'relu' }));
    model.add(tf.layers.dense({ units: 128, inputShape: [vocab.length], activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, inputShape: [vocab.length], activation: 'relu' }));
    model.add(tf.layers.dense({ units: vocab.length, activation: 'softmax' }));

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });



    async function trainModel() {

        const xs = tf.tensor2d(encodedData.map(({ input }) => input));
        const ys = tf.tensor2d(encodedData.map(({ output }) => output));
        
        const trainingMessage = document.getElementById('training-message');
        const sentenceInput = document.getElementById('sentence-input');
        const simplifyButton = document.getElementById('simplify-button');

        if (trainingMessage && sentenceInput && simplifyButton) {
            trainingMessage.style.display = 'block'; // Show the training message
            console.log('Starting model training...');
            await model.fit(xs, ys, {
                epochs: 1000,
                callbacks: tf.callbacks.earlyStopping({ monitor: 'loss' })
            });
            console.log('Model training complete!');
            trainingMessage.style.display = 'none'; // Hide the training message
            sentenceInput.disabled = false; // Enable input
            simplifyButton.disabled = false; // Enable button
        } else {
            console.error('One or more elements were not found on the page.');
        }
    }

    function predictVerb(complexVerb) {
        const inputTensor = tf.tensor2d([encodeWord(complexVerb)]);
        const outputTensor = model.predict(inputTensor);
        const outputIndex = outputTensor.argMax(1).dataSync()[0];
        return vocab[outputIndex];
    }

    function simplifySentence(sentence) {
        let doc = nlp(sentence);

        doc.verbs().forEach(verb => {
            let text = verb.text();
            if (complexVerbs.includes(text)) {
                let simplified = predictVerb(text);
                doc.replace(text, simplified);
            }
        });

        return doc.text();
    }

    // Simplify the sentence provided in the input field
    function simplifyInputSentence() {
        const sentence = document.getElementById('sentence-input').value;
        document.getElementById('original-sentence').innerText = sentence;
        const simplifiedSentence = simplifySentence(sentence);
        document.getElementById('simplified-sentence').innerText = simplifiedSentence;
    }

    // Train the model when the page loads
    trainModel().then(() => {
        console.log('Model is ready!');
    });

    // Expose the function to the global scope so it can be called by the button
    window.simplifyInputSentence = simplifyInputSentence;
});