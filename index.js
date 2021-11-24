let trainedNet; // pre-trained network example, uncomment bellow:
// trainedNet = [
//   [{ weights: [-9.05941622825273] }, { weights: [8.749511815472376] }],
//   [{ bias: -0.45752638281920205, weights: [-8.196013850956946, 8.174752017015699] }],
//   [{ bias: 3.3835733329507174 }, { bias: -4.372771104374529 }],
// ];

// NETWORK INITIALIZATION:
const layers = [2, 1, 2];
const network = [];
let neuronID = 0;
let weightID = 0;
const getRandom = () => Math.random() * 0.4 - 0.2; // brain.js:1413
layers.forEach((neuronsCountCurrentLayer, layerID) => {
  const neuronsCountPrevLayer = layers[layerID - 1];
  const neuronsCountNextLayer = layers[layerID + 1];
  const layer = [];
  for (let i = 0; i < neuronsCountCurrentLayer; i++) {
    const neuron = { input: [], output: [], id: neuronID++, layerID };
    if (layerID > 0) {
      neuron.value = neuron.delta = neuron.error = 0;
      neuron.bias = trainedNet ? trainedNet[layerID][i].bias : getRandom();
    }
    if (neuronsCountPrevLayer) {
      network[layerID - 1].forEach(neuronPrev => {
        neuronPrev.output.forEach((weight, weightID) => {
          i === weightID && neuron.input.push(weight);
        });
      });
    }
    if (neuronsCountNextLayer) {
      for (let j = 0; j < neuronsCountNextLayer; j++) {
        const weight = { change: 0, id: weightID++ };
        weight.value = trainedNet ? trainedNet[layerID][i].weights[j] : getRandom();
        neuron.output.push(weight);
      }
    }
    layer.push(neuron);
  }
  network.push(layer);
});

const round = x => Math.round(x * 10000) / 10000; // 0.4115000143.. -> 0.4115
const joinString = i => i.join(' | ').slice(0, 90);
const weightString = w => `id:${w.id} value:${round(w.value)} change:${round(w.change)}`;
const layerMap = neuron => ({
  id: neuron.id,
  layer: `L${neuron.layerID}`,
  ['input weights']: joinString(neuron.input.map(weightString)),
  value: round(neuron.value),
  bias: typeof neuron.bias === 'number' ? round(neuron.bias) : '',
  error: typeof neuron.error === 'number' ? round(neuron.error) : '',
  delta: typeof neuron.delta === 'number' ? round(neuron.delta) : '',
  ['output weights']: joinString(neuron.output.map(weightString)),
});
const printNetwork = () => {
  console.table(network.flat(1).map(layerMap));
  const trainedNet = network.map((layer, layerID) => {
    return layer.map(n => {
      const neuronObj = {};
      if (layerID > 0) {
        neuronObj.bias = n.bias;
      }
      if (layerID < network.length - 1) {
        neuronObj.weights = n.output.map(w => w.value);
      }
      return neuronObj;
    });
  });
  console.log(`trainedNet JSON:`, JSON.stringify(trainedNet));
};

// printNetwork();debugger;

const sigmoid = x => 1 / (1 + Math.exp(-x));
const learnRate = 0.7;
const momentum = 0.1;
const run = (input, expected = []) => {
  // SET INITIAL VALUES:
  for (let layerID = 0; layerID < network.length; layerID++) {
    for (let neuronID = 0; neuronID < network[layerID].length; neuronID++) {
      network[layerID][neuronID].value = layerID === 0 ? input[neuronID] : 0;
    }
  }

  // PROJECT VALUES:
  for (let layerID = 0; layerID < network.length; layerID++) {
    const nextLayer = network[layerID + 1] || [];
    for (let neuronID = 0; neuronID < network[layerID].length; neuronID++) {
      const neuron = network[layerID][neuronID];
      for (let weightID = 0; weightID < neuron.output.length; weightID++) {
        nextLayer[weightID].value += neuron.output[weightID].value * neuron.value;
      }
    }
    for (let neuronID = 0; neuronID < nextLayer.length; neuronID++) {
      const neuron = nextLayer[neuronID];
      neuron.value = sigmoid(neuron.value + neuron.bias);
    }
  }

  // TRAIN:
  if (expected.length) {
    // CALCULATE DELTAS:
    for (let layerID = network.length - 1; layerID > 0; layerID--) {
      const layer = network[layerID];
      const nextLayer = network[layerID + 1] || [];
      for (let neuronID = 0; neuronID < layer.length; neuronID++) {
        const neuron = layer[neuronID];
        if (layerID === network.length - 1) {
          neuron.error = expected[neuronID] - neuron.value;
        } else {
          neuron.error = 0;
          for (let nextID = 0; nextID < nextLayer.length; nextID++) {
            const neuronNext = nextLayer[nextID];
            neuron.error += neuronNext.delta * neuronNext.input[neuronID].value;
          }
        }
        neuron.delta = neuron.error * neuron.value * (1 - neuron.value);
      }
    }

    // ADJUST WEIGHTS:
    for (let layerID = 1; layerID < network.length; layerID++) {
      const layer = network[layerID];
      const prevLayer = network[layerID - 1] || [];
      for (let neuronID = 0; neuronID < layer.length; neuronID++) {
        const neuron = layer[neuronID];
        for (let weightID = 0; weightID < neuron.input.length; weightID++) {
          const weight = neuron.input[weightID];
          weight.change = learnRate * neuron.delta * prevLayer[weightID].value + momentum * weight.change;
          weight.value += weight.change;
        }
        neuron.bias += learnRate * neuron.delta;
      }
    }
  }

  return network[network.length - 1].map(i => i.value); // last layer values
};

for (let i = 0; i < 10000; i++) {
  const expected = [Math.floor(Math.random() * 2)];
  expected[1] = expected[0] === 0 ? 1 : 0;
  const input = expected.map(x => round((Math.random() + x) / 2));
  run(input, expected); // train
  if (i % 500 === 0) {
    const result = run([0.2, 0.6]); // expected [0, 1]
    if (result[0] < 0.02) {
      console.log(`iteration: ${i}    \t`, result, `- reached 2% level, training stopped`);
      break;
    } else {
      console.log(`iteration: ${i}    \t`, result);
    }
  }
}

printNetwork();
