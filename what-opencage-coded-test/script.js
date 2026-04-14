const customerServiceButton = document.querySelector('.customer-service button');
const customerServiceTextarea = document.querySelector('.customer-service textarea');
const customerServiceResponse = document.querySelector('.customer-service .response');

customerServiceButton.addEventListener('click', () => {
    const question = customerServiceTextarea.value;
    const response = getResponse(question);
    customerServiceResponse.innerText = response;
    customerServiceTextarea.value = '';
});

function getResponse(question) {
    // Mock AI responses
    const responses = {
        'hello': 'Hi! How can I assist you today?',
        'what is your return policy': 'We accept returns within 30 days of purchase.',
        'do you have any discounts': 'Yes, we offer 10% off all orders over $50.',
    };

    return responses[question.toLowerCase()] || 'Sorry, I didn\'t understand your question.';
}