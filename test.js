const { brain } = require('brain.js');
const config = {
  binaryThresh: 0.5,
  hiddenLayers: [1], // array of ints for the sizes of the hidden layers in the network
  activation: 'sigmoid', // supported activation types: ['sigmoid', 'relu', 'leaky-relu', 'tanh'],
  leakyReluAlpha: 0.01, // supported for activation type 'leaky-relu'
  iterations: 1,
  log: console.log,
  logPeriod: 30,
};

// create a simple feed forward neural network with backpropagation
const net = new brain.NeuralNetwork(config);
const round = x => Math.round(x * 10000) / 10000; // 0.42851.. >> 0.43

const setExpected = x => round((Math.random() + x) / 2);
for (let i = 0; i < 4999; i++) {
  let expected = [Math.floor(Math.random() * 2)]; // config expected result
  expected[1] = expected[0] === 0 ? 1 : 0;
  let input = [setExpected(expected[0]), setExpected(expected[1])];
  if (i !== 0) {
    input = [0.2, 0.6];
    expected = [0, 1];
    net.biases = [null, [-0.4424299895763397], [2.8855154514312744, -2.8785271644592285]];
    // //net.outputs = [[0.2, 0.6], [0.808885395526886], [0.09636910259723663, 0.9032278060913086]];
    net.outputs = [null, [], []];
    net.weights = [null, [[-5.431631088256836, 5.519097805023193]], [[-6.339480876922607], [6.325170040130615]]];
    net.errors = [
      [-0.00015359566896222532, -0.0007743531605228782],
      [-0.016961565241217613],
      [-0.5144253969192505, 0.4831986427307129],
    ];
    net.deltas = [
      [-0.000002596405693111592, -0.0001575937494635582],
      [-0.004239736590534449],
      [-0.12849929928779602, 0.12066326290369034],
    ];
    net.changes = [
      null,
      [[-0.00002187704194511752, -0.0009101866744458675]],
      [[-0.01951434090733528], [0.018324334174394608]],
    ];
    debugger;
  }
  net.train([{ input, output: expected }]);
  if (i % 100 === 0) {
    // console.log(i, input, net.run(input));
  }
}

// net.sizes = [2, 1, 2];
// net.outputLayer = 2;

// console.log(`const biases=${JSON.stringify(this.biases)};\nconst outputs=${JSON.stringify(this.outputs)};\nconst weights=${JSON.stringify(this.weights)};\nconst errors=${JSON.stringify(this.errors)};\nconst deltas=${JSON.stringify(this.deltas)};\nconst changes=${JSON.stringify(this.changes)};`)

// sum = bias + ( weight * output ) > sigmoid
const sum = -0.4424299895763397 + -5.431631088256836 * 0.2 + 5.519097805023193 * 0.6;
// 1.782702475786209 > sigmoid > 0.8560302462469835

const outputs2 = [[0.2, 0.6], [0.8560302462469835], [0.09636910259723663, 0.9032278060913086]];

const sum2 = 2.8855154514312744 + -6.339480876922607 * 0.8560302462469835;
// -2.5412719247188287 > sigmoid > 0.07301503790252971
const sum3 = -2.8785271644592285 + 6.325170040130615 * 0.8560302462469835;
// 2.536009702547825 > sigmoid > 0.9266279934372447

const result = [0.07301503790252971, 0.9266279934372447];

const test = () => {
  const biases = [null, [-0.4424299895763397], [2.8840328701795035, -2.8770306276829976]];
  const outputs = [
    [0.20000000298023224, 0.6000000238418579],
    [0.8560302604689196],
    [0.07301503180017563, 0.9266279995532338],
  ];
  const weights = [null, [[-5.431633275961031, 5.519006786355749]], [[-6.342701445428461], [6.328283554314413]]];
  const errors = [[0, 0], [0], [-0.07301503180017563, 0.07337200044676617]];
  const deltas = [[0, 0], [0], [-0.004941937505903706, 0.004988455920769976]];
  const changes = [
    null,
    [[-0.000002187704194511752, -0.00009101866744458676]],
    [[-0.00322056850585349], [0.003113514183797795]],
  ];
};

const t2 = () => {
  // const biases = [null, [-0.4424299895763397], [2.8855154514312744, -2.8785271644592285]];
  // const outputs = [
  //   [0.20000000298023224,  0.6000000238418579 },
  //   [0.8560302604689196],
  //   [0.07301503180017563, 0.9266279995532338],
  // ];
  // const weights = [null, [[-5.431631088256836, 5.519097805023193]], [[-6.339480876922607], [6.325170040130615]]];
  // const errors = [
  //   [-0.00015359566896222532,  -0.0007743531605228782 },
  //   [-0.016961565241217613 },
  //   [-0.5144253969192505,  0.4831986427307129 },
  // ];
  // const deltas = [
  //   [-0.000002596405693111592,  -0.0001575937494635582 },
  //   [-0.004239736590534449 },
  //   [-0.12849929928779602,  0.12066326290369034 },
  // ];
  // const changes = [
  //   null,
  //   [[-0.00002187704194511752,  -0.0009101866744458675 }],
  //   [[-0.01951434090733528 }, [0.018324334174394608 }],
  // ];
};
