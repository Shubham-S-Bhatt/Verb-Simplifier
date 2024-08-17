# Verb-Simplifier

Overview
Verb Simplifier is a web-based application that simplifies complex verbs in sentences using a trained neural network model built with TensorFlow.js. This tool is particularly useful for making text more accessible by replacing less common, complex verbs with simpler, more commonly understood alternatives.

Features
Neural Network Model: Utilizes TensorFlow.js to train and deploy a neural network model directly in the browser.
Real-time Sentence Simplification: Allows users to input sentences and get simplified versions instantly.
Beautiful User Interface: The application features a modern and user-friendly interface built with HTML, CSS, and JavaScript.
Interactive and Responsive Design: The UI is designed to be responsive and interactive, ensuring a smooth user experience across different devices.
How It Works
Training: When the page loads, a neural network model is trained on a dataset of complex verb-to-simple verb mappings.
Simplification: After training, users can input any sentence, and the application will identify and simplify complex verbs within the sentence.
Technologies Used
HTML5: For the structure of the application.
CSS3: For styling and ensuring a visually appealing and responsive design.
JavaScript (ES6+): For the application logic, including interaction with TensorFlow.js and Compromise.js.
TensorFlow.js: For creating and training the neural network model directly in the browser.
Compromise.js: For natural language processing, specifically verb identification in sentences.
Installation
To use this application locally:

Clone the repository:
bash
Copy code
git clone https://github.com/your-username/verb-simplifier.git
Navigate to the project directory:
bash
Copy code
cd verb-simplifier
Open index.html in your preferred web browser.
Usage
Wait for the model to train upon page load.
Enter a sentence in the input field.
Click the "Simplify Sentence" button to get a simplified version of your sentence.
