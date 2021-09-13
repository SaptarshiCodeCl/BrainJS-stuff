const brain = require('brain.js');

const trainingData = [
	'Jane saw Doug.',
	'Doug saw Jane.',
	'Spot saw Doug and Jane looking at each other.',
	'It was love at first sight, and Spot had a frontrow seat. It was a very special moment for all.',
];

const lstm = new brain.recurrent.LSTM();
const result = lstm.train(trainingData, {
	iterations: 1500,
	//  log: (details) => console.log(details),
	errorThresh: 0.011,
});

console.log(result);

const run1 = lstm.run('Who had a frontrow seat');
console.log('run 1: It' + run1);
