const brain = require('brain.js');
const fs = require('fs');

const net = new brain.recurrent.LSTMTimeStep();

// Same test as previous, but combined on a single set
const trainingData = [
	[1, 4, 5, 6],
	[1, 4, 5, 6],
	[2, 3, 5, 6],
	[1, 2, 6, 4],
	[2, 3, 1, 5],
	[1, 2, 6, 4],
	[2, 3, 1, 5],
];

if (fs.existsSync('trained.json')) {
	const trainedJson = fs.readFileSync('trained.json');
	net.fromJSON(JSON.parse(trainedJson));
} else {
	net.train(trainingData, { log: true, errorThresh: 0.1 });
	fs.writeFileSync('trained.json', JSON.stringify(net.toJSON()));
}

const forecast = net.forecast([1, 3, 2], 2);
normalised = forecast.map((values) => Math.round(values));

console.log('next 2 predictions', normalised);
