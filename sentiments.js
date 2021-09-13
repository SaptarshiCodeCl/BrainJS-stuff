const brain = require('brain.js');
const fs = require('fs');
const net = new brain.recurrent.LSTM();

const categories = ['Greeting', 'Sad', 'Happy'];

const trainingData = [
	{ input: 'Hello How are you', output: categories[0] },
	{ input: 'Hello Good morning', output: categories[0] },
	{ input: 'Hi!', output: categories[0] },
	{ input: 'Good evening', output: categories[0] },
	{ input: 'The world is a terrible place!', output: categories[1] },
	{ input: 'I dont feel good about this', output: categories[1] },
	{ input: 'I feel terrible', output: categories[1] },
	{ input: 'Thankyou', output: categories[2] },
	{ input: 'That sounds good', output: categories[2] },
	{ input: 'The weather feels beautiful today', output: categories[2] },
];

if (fs.existsSync('sentiments-model.json')) {
	const trainedJson = fs.readFileSync('sentiments-model.json');
	net.fromJSON(JSON.parse(trainedJson));
} else {
	net.train(trainingData, { log: true, iterations: 500 });
	fs.writeFileSync('sentiments-model.json', JSON.stringify(net.toJSON()));
}
console.log(net.run('I feel terrible'));
