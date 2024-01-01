function calculateProbability() {
    // Gather user input
    const pclass = document.getElementById('pclass').value;
    const female = document.getElementById('female').value;
    const age = document.getElementById('age').value;
    const fare = document.getElementById('fare').value;

    // Prepare data for the API request
    const requestData = {
        Pclass: parseInt(pclass),
        Female: JSON.parse(female),
        Age: parseInt(age),
        Fare: parseInt(fare),
    };
    // Make the API request
    fetch('https://titanic-survival-probability.onrender.com/:5000/prediction', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        // Display the result to the user
        const resultContainer = document.getElementById('result');
        resultContainer.innerHTML = `<h4>Berechnete Überlebenswahrscheinlichkeit beträgt ${data.prediction * 100}%</h4>`;
    })
    .catch(error => {
        // Display an error message to the user
        const resultContainer = document.getElementById('result');
        resultContainer.innerHTML = '<p>An error occurred while fetching the data. Please try again later.</p>';
    });
}