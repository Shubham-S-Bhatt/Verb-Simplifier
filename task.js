// Mapping of complex verbs to simpler verbs
const verbMapping = {
    'obliterates': 'destroys',
    'destructs': 'destroys',
    'devastates': 'destroys',
    'annihilates': 'destroys',
    'annihilates': 'destroys',
    'obliterates': 'destroys',
};


function simplifySentence(sentence) {
    // Parse the sentence using compromise
    let doc = nlp(sentence);

    // Replace complex verbs with simpler verbs
    doc.verbs().forEach(verb => {
        console.log(verb)
        let text = verb.text();
        if (verbMapping[text]) {
            doc.replace(text, verbMapping[text]);
        }
    });

    // Return the simplified sentence
    return doc.text();
}


// Example usage
let sentence = 'When the fox touches a rabbit, it obliterates the rabbit.';
let simplifiedSentence = simplifySentence(sentence);
console.log(simplifiedSentence);