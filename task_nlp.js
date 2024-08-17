// #####################################################################################
// ############### Using Neural Network and one hot encoding ###########################
// #####################################################################################

document.addEventListener('DOMContentLoaded', () => {

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

    const complexVerbs = trainingData.map(({ input }) => input);
    // console.log('Complex verbs:', complexVerbs);

    const vocab = [...new Set(trainingData.flatMap(({ input, output }) => [input, output]))];
    const wordIndex = Object.fromEntries(vocab.map((word, index) => [word, index]));
    // console.log('Vocabulary:', vocab);

    function encodeWord(word) {
        const encoded = new Array(vocab.length).fill(0);
        encoded[wordIndex[word]] = 1;
        return encoded;
    }

    const encodedData = trainingData.map(({ input, output }) => ({
        input: encodeWord(input),
        output: encodeWord(output)
    }));    
    // console.log('Encoded data:', encodedData);


    // Defining the model
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 32, inputShape: [vocab.length], activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
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

    // Train the model
    trainModel().then(() => {
        console.log('Model is ready!');
    });

    // Exposing the function to the global scope so it can be called by the button
    window.simplifyInputSentence = simplifyInputSentence;
});