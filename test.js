const { brain } = require('brain.js');

const config = {
  binaryThresh: 0.5,
  hiddenLayers: [1], // array of ints for the sizes of the hidden layers in the network
  activation: 'sigmoid', // supported activation types: ['sigmoid', 'relu', 'leaky-relu', 'tanh'],
  leakyReluAlpha: 0.01, // supported for activation type 'leaky-relu'
  iterations: 500,
  log: console.log,
  logPeriod: 30,
};

// create a simple feed forward neural network with backpropagation
const net = new brain.NeuralNetwork(config);

const round = x => Math.round(x * 10000) / 10000; // 0.42851.. >> 0.43
const setExpected = x => round((Math.random() + x) / 2);

const trainArr = [];
for (let i = 0; i < 9; i++) {
  const expected = [Math.floor(Math.random() * 2)]; // config expected result
  expected[1] = expected[0] === 0 ? 1 : 0;
  trainArr.push({
    input: [setExpected(expected[0]), setExpected(expected[1])],
    output: expected,
  });
}
// console.log(trainArr);
// const trainArr2 = [
//   { input: [0, 0], output: [0] },
//   { input: [0, 1], output: [1] },
//   { input: [1, 0], output: [1] },
//   { input: [1, 1], output: [0] },
// ];

debugger;
net.train(trainArr);

const output = net.run([0.2, 0.6]);

debugger;
console.log(output);
