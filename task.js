// Mapping of complex verbs to simpler verbs
const verbMapping = {
    'obliterates': 'destroys',
    'annihilates': 'destroys',
    'promulgates': 'issues',
    'delineates': 'describes',
    'mitigates': 'reduces',
    'revolutionizes': 'changes',
    'adjudicates': 'judges',
    'calibrates': 'adjusts',
    'orchestrates': 'organizes',
    'elucidates': 'explains',
    'endeavors': 'tries',
    'organizes': 'arranges',
    'explains': 'clarifies',
    'tries': 'attempts'
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



// Simplify the sentence provided in the input field
function simplifySentenceSimple() {
    const sentence = document.getElementById('sentence-input-simple').value;
    document.getElementById('original-sentence-simple').innerText = sentence;
    const simplifiedSentence = simplifySentence(sentence);
    document.getElementById('simplified-sentence-simple').innerText = simplifiedSentence;
}


// Expose the function to the global scope so it can be called by the button
window.simplifySentenceSimple = simplifySentenceSimple;